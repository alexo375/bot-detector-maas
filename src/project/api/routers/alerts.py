from email.header import Header
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from project.api.schemas.alerts import (
    AlertmanagerWebhookPayload,
    AlertWebhookResponse,
)
from project.core.config import config
from project.core.logging import ProjectLogger
from project.core.urls import service_urls

logger = ProjectLogger.get_logger(__name__)

router = APIRouter(prefix="/alerts", tags=["Alerts"])

_NTFY_PRIORITY_MAP: dict[str, int] = {
    "min": 1,
    "low": 2,
    "default": 3,
    "high": 4,
    "urgent": 5,
    "max": 5,
}


class AlertFormatter:
    """Форматировщик входящего JSON от Alertmanager в Markdown-уведомление для ntfy."""

    @staticmethod
    def _format_extra_alerts(alerts: list[dict[str, Any]]) -> list[str]:
        if len(alerts) <= 1:
            return []
        lines = ["", f"**Additional alerts ({len(alerts) - 1}):**"]
        for extra in alerts[1:10]:
            e_labels = extra.get("labels", {})
            e_ann = extra.get("annotations", {})
            e_name = e_labels.get("alertname", "MLSystemAlert")
            e_sev = str(e_labels.get("severity", "info")).lower()
            e_sum = e_ann.get("summary", "")
            lines.append(f"- `{e_name}` (`{e_sev}`)" + (f": {e_sum}" if e_sum else ""))
        if len(alerts) > 10:
            lines.append(f"- ... and {len(alerts) - 10} more")
        return lines

    @classmethod
    def format(cls, payload: dict[str, Any]) -> tuple[str, str, str, str]:
        """Преобразование Alertmanager JSON в `(title, priority, tags, markdown_body)`."""
        global_status = payload.get("status", "firing").lower()
        alerts = payload.get("alerts", [])
        first_alert = alerts[0] if alerts else {}
        labels = first_alert.get("labels", {}) or payload.get("commonLabels", {})
        annotations = first_alert.get("annotations", {}) or payload.get("commonAnnotations", {})

        alertname = labels.get("alertname", "MLSystemAlert")
        severity = labels.get("severity", "warning").lower()
        summary = annotations.get("summary", "Metric anomaly detected")
        description = annotations.get("description", "")

        if global_status == "resolved":
            title = f"[RESOLVED] {alertname}"
            priority = "default"
            tags = "white_check_mark"
            body_lines = [f"**Resolved:** {summary}", "System metrics returned to normal levels."]
        else:
            is_critical = severity in ("critical", "error")
            title = f"[FIRING] {alertname}"
            priority = "high" if is_critical else "default"
            tags = "rotating_light" if is_critical else "warning"
            body_lines = [f"**{summary}**"]
            if description and description != summary:
                body_lines.append(description)

        body_lines.extend(cls._format_extra_alerts(alerts))
        return title, priority, tags, "\n".join(body_lines)


class NtfyClient:
    """Асинхронный клиент отправки структурированных уведомлений в сервис `ntfy`."""

    @staticmethod
    def build_json_request(
        ntfy_url: str,
        title: str,
        priority: str,
        tags: str,
        markdown_body: str,
    ) -> tuple[str, dict[str, Any] | None]:
        """Разбирает `https://host/<topic>` в нативный JSON publish-запрос `ntfy`."""
        parts = urlsplit(ntfy_url.strip())
        topic = parts.path.strip("/")
        if not parts.scheme or not parts.netloc or not topic:
            return ntfy_url, None

        root_url = f"{parts.scheme}://{parts.netloc}/"
        payload = {
            "topic": topic,
            "title": title,
            "message": markdown_body,
            "priority": _NTFY_PRIORITY_MAP.get(priority.lower(), 3),
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
            "markdown": True,
            "actions": [
                {"action": "view", "label": "Grafana", "url": service_urls.grafana_drift_dashboard, "clear": True},
                {"action": "view", "label": "Prefect", "url": service_urls.prefect_deployments, "clear": True},
                {"action": "view", "label": "MLflow", "url": service_urls.mlflow_models, "clear": True},
            ],
        }
        return root_url, payload

    @classmethod
    async def send(
        cls,
        title: str,
        priority: str,
        tags: str,
        markdown_body: str,
        ntfy_url: str,
    ) -> tuple[bool, str | None]:
        """Асинхронная доставка структурированного уведомления в `ntfy` через JSON API."""
        publish_url, json_payload = cls.build_json_request(ntfy_url, title, priority, tags, markdown_body)
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                if json_payload is not None:
                    resp = await client.post(publish_url, json=json_payload)
                else:
                    actions = (
                        f"view, Grafana, {service_urls.grafana_drift_dashboard}, clear=true; "
                        f"view, Prefect, {service_urls.prefect_deployments}, clear=true; "
                        f"view, MLflow, {service_urls.mlflow_models}, clear=true"
                    )
                    headers = {
                        "Title": Header(title, "utf-8").encode(),
                        "Priority": priority,
                        "Tags": tags,
                        "Markdown": "yes",
                        "Actions": actions,
                    }
                    resp = await client.post(publish_url, content=markdown_body.encode("utf-8"), headers=headers)
                resp.raise_for_status()
                logger.info("Delivered alert '%s' to ntfy (%s, HTTP %d)", title, ntfy_url, resp.status_code)
                return True, None
        except (httpx.HTTPError, UnicodeError, ValueError, OSError) as exc:
            err_detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            logger.warning("Could not forward alert to ntfy (%s): %s", ntfy_url, err_detail)
            return False, err_detail


@router.post(
    "/webhook",
    status_code=status.HTTP_200_OK,
    response_model=AlertWebhookResponse,
    summary="Вебхук приемник алертов от Alertmanager",
    description="Принимает системные алерты от Prometheus Alertmanager, форматирует и отправляет уведомления в ntfy.",
)
async def alertmanager_webhook(
    payload: AlertmanagerWebhookPayload,
) -> AlertWebhookResponse | JSONResponse:
    """Обработчик входящего вебхука от Prometheus Alertmanager."""
    title, priority, tags, markdown_body = AlertFormatter.format(payload.model_dump())
    dispatched, err = await NtfyClient.send(title, priority, tags, markdown_body, config.ntfy_url)
    if not dispatched:
        body = AlertWebhookResponse(status="error", alert=title, dispatched=False, error=err)
        return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content=body.model_dump())
    return AlertWebhookResponse(status="ok", alert=title, dispatched=True)
