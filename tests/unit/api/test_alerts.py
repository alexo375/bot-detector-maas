"""Тесты для роутера стилизации алертов Alertmanager -> ntfy."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from project.api.app import app
from project.api.routers.alerts import AlertFormatter, NtfyClient


def test_format_alert_message_firing() -> None:
    """Проверка форматирования алерта в статусе FIRING."""
    payload = {
        "status": "firing",
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "HighDataDriftDetected",
                    "severity": "warning",
                    "service": "bot-detector-gateway",
                    "instance": "api-gateway:8080",
                },
                "annotations": {
                    "summary": "Feature Data Drift share exceeds 20%",
                    "description": "Statistical data drift detected across input features (share: 64.21%). Model retraining recommended.",
                },
                "startsAt": "2026-11-14T16:27:41.073Z",
            }
        ],
    }

    title, priority, tags, markdown_body = AlertFormatter.format(payload)

    assert title == "[FIRING] HighDataDriftDetected"
    assert priority == "default"
    assert "warning" in tags
    assert "**Feature Data Drift share exceeds 20%**" in markdown_body
    assert "64.21%" in markdown_body
    assert "### [FIRING]" not in markdown_body
    assert "**Links:**" not in markdown_body


def test_format_alert_message_resolved() -> None:
    """Проверка форматирования алерта в статусе RESOLVED."""
    payload = {
        "status": "resolved",
        "alerts": [
            {
                "status": "resolved",
                "labels": {
                    "alertname": "HighDataDriftDetected",
                    "severity": "warning",
                    "service": "bot-detector-gateway",
                },
                "annotations": {
                    "summary": "Data drift returned to normal levels",
                },
            }
        ],
    }

    title, priority, tags, markdown_body = AlertFormatter.format(payload)

    assert title == "[RESOLVED] HighDataDriftDetected"
    assert priority == "default"
    assert "white_check_mark" in tags
    assert "**Resolved:** Data drift returned to normal levels" in markdown_body
    assert "System metrics returned to normal levels." in markdown_body


@pytest.mark.asyncio
async def test_forward_to_ntfy_sends_utf8_json_payload() -> None:
    """Проверка отправки уведомления в нативном JSON-формате ntfy с поддержкой UTF-8 и actions."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"id": "ok"})

    transport = httpx.MockTransport(handler)
    orig_client = httpx.AsyncClient

    def mock_client_factory(*_args: object, **_kwargs: object) -> httpx.AsyncClient:
        return orig_client(transport=transport)

    with patch("project.api.routers.alerts.httpx.AsyncClient", side_effect=mock_client_factory):
        dispatched, err = await NtfyClient.send(
            title="[FIRING] Дрейф признаков (warning)",
            priority="high",
            tags="warning",
            markdown_body="### Обнаружен дрейф\nЗначение: 64.2%",
            ntfy_url="https://ntfy.sh/bot-detector-alerts",
        )

    assert dispatched is True
    assert err is None
    assert captured["url"] == "https://ntfy.sh/"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["topic"] == "bot-detector-alerts"
    assert body["title"] == "[FIRING] Дрейф признаков (warning)"
    assert body["priority"] == 4
    assert body["markdown"] is True
    assert len(body["actions"]) == 3
    prefect_action = next(a for a in body["actions"] if a["label"] == "Prefect")
    assert "/v2/deployments" in prefect_action["url"]


def test_alertmanager_webhook_endpoint() -> None:
    """Проверка приема вебхука API Gateway при успешной и неуспешной доставке в ntfy."""
    client = TestClient(app)
    payload = {
        "receiver": "ntfy",
        "status": "firing",
        "alerts": [
            {
                "status": "firing",
                "labels": {"alertname": "TestAlert", "severity": "warning"},
                "annotations": {"summary": "Test Summary"},
            }
        ],
    }

    with patch("project.api.routers.alerts.NtfyClient.send", new_callable=AsyncMock, return_value=(True, None)):
        resp = client.post("/alerts/webhook", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["dispatched"] is True
        assert "TestAlert" in data["alert"]

    with patch(
        "project.api.routers.alerts.NtfyClient.send",
        new_callable=AsyncMock,
        return_value=(False, "ConnectTimeout"),
    ):
        resp_fail = client.post("/alerts/webhook", json=payload)
        assert resp_fail.status_code == 502
        fail_data = resp_fail.json()
        assert fail_data["status"] == "error"
        assert fail_data["dispatched"] is False
        assert fail_data["error"] == "ConnectTimeout"


def test_alert_schemas_and_openapi_generation() -> None:
    """Проверка генерации Pydantic-схем в OpenAPI спецификации."""
    client = TestClient(app)
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()

    webhook_op = schema["paths"]["/alerts/webhook"]["post"]
    assert "requestBody" in webhook_op
    content = webhook_op["requestBody"]["content"]["application/json"]
    assert "$ref" in content["schema"]
    assert "AlertmanagerWebhookPayload" in content["schema"]["$ref"]

    response_200 = webhook_op["responses"]["200"]["content"]["application/json"]
    assert "$ref" in response_200["schema"]
    assert "AlertWebhookResponse" in response_200["schema"]["$ref"]

    components = schema["components"]["schemas"]
    assert "AlertmanagerWebhookPayload" in components
    assert "AlertItem" in components
    assert "AlertWebhookResponse" in components
