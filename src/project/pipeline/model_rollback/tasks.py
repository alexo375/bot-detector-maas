"""Prefect-задачи и операционные утилиты для ручного отката моделей."""

import datetime as dt
import json
import time

import redis
from mlflow.tracking import MlflowClient
from prefect import task

from project.core.config import config
from project.core.logging import ProjectLogger
from project.drift.data_drift.fetcher import DWHFeatureFetcher
from project.infrastructure.triton.client import TritonEnsembleClient
from project.pipeline.metrics_sync import DriftMetricsSyncService
from project.pipeline.model_rollback.inspector import ModelVersionInspector
from project.pipeline.model_rollback.schemas import ModelVersionCandidate, RollbackMode

logger = ProjectLogger.get_logger(__name__)

EFFECTIVE_SINCE_KEY = "bot_detector:champion_effective_since_ts"


class RollbackAliasManager:
    """Управление алиасами `champion` / `challenger` / `rolled_back` в MLflow при откате."""

    @staticmethod
    def rebalance_after_rollback(
        client: MlflowClient,
        model_name: str,
        target_version: str,
        prev_ver: str | None,
        tracking_uri: str | None,
    ) -> None:
        """Сдвигает или очищает `@challenger` и снимает устаревший `@rolled_back` при откате."""
        excluded = {str(target_version)}
        if prev_ver is not None and prev_ver != str(target_version):
            excluded.add(str(prev_ver))

        rolled_ver = ModelVersionInspector.safe_alias_version(client, model_name, "rolled_back")
        if rolled_ver is not None:
            excluded.add(rolled_ver)

        chall_ver = ModelVersionInspector.safe_alias_version(client, model_name, "challenger")
        if chall_ver is not None and chall_ver not in excluded:
            return

        next_chall: str | None = None
        try:
            candidates = ModelVersionInspector.list_candidates(model_name=model_name, tracking_uri=tracking_uri)
            next_chall = next((c.version for c in candidates if c.version not in excluded), None)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Challenger candidate rebalance lookup skipped: %s", exc)

        if next_chall is not None:
            client.set_registered_model_alias(model_name, "challenger", next_chall)
            logger.info("Shifted %s @challenger to v%s after rollback", model_name, next_chall)
        elif chall_ver is not None:
            try:
                client.delete_registered_model_alias(model_name, "challenger")
                logger.info("Cleared stale %s @challenger (was v%s) after rollback", model_name, chall_ver)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to delete stale @challenger alias: %s", exc)


@task(name="check_no_active_retrain")
def check_no_active_retrain_task() -> None:
    """Проверяет отсутствие активного переобучения в Redis перед откатом."""
    try:
        r_conn = redis.Redis.from_url(config.redis.url, socket_timeout=1.0)
        in_progress = bool(r_conn.get("bot_detector:retrain_in_progress"))
        r_conn.close()
        if in_progress:
            raise RuntimeError(
                "Нельзя выполнить откат во время активного переобучения (bot_detector:retrain_in_progress)."
            )
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.debug("Redis active retrain check skipped during rollback: %s", exc)


@task(name="reset_drift_window_after_rollback")
def reset_window_after_rollback_task(target_version: str, reset_drift_baseline: bool = True) -> None:
    """Сбрасывает окно дрифта и метрики модели после отката."""
    if not reset_drift_baseline:
        return
    try:
        r_conn = redis.Redis.from_url(config.redis.url, socket_connect_timeout=1.0, socket_timeout=1.0)
        r_conn.set(EFFECTIVE_SINCE_KEY, json.dumps({"version": str(target_version), "ts": time.time()}))
        r_conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist %s in Redis: %s", EFFECTIVE_SINCE_KEY, exc)

    DriftMetricsSyncService.reset_baseline(
        DWHFeatureFetcher().get_feature_names(),
        dialogues_count=0,
        model_age_days=0.0,
        retrain_status=0,
        reason="rollback",
    )
    DriftMetricsSyncService.trigger_api_gateway_metrics()


@task(name="resolve_rollback_target")
def resolve_rollback_target_task(
    mode: RollbackMode,
    steps_back: int = 1,
    target_version: str = "",
    candidates: list[ModelVersionCandidate] | None = None,
) -> ModelVersionCandidate:
    """Определяет целевую совместимую версию MLflow по режиму отката."""
    items = candidates if candidates is not None else ModelVersionInspector.list_candidates()
    if not items:
        raise ValueError("В MLflow Model Registry нет совместимых версий 'bot_detector' для отката.")

    if mode == RollbackMode.EXACT_VERSION:
        normalized = target_version.strip().lstrip("v")
        if not normalized:
            raise ValueError("Для режима 'exact_version' требуется указать номер версии в target_version.")
        for item in items:
            if item.version == normalized:
                return item
        available = ", ".join(c.version for c in items)
        raise ValueError(f"Версия v{normalized} не найдена среди совместимых версий MLflow ({available}).")

    if mode == RollbackMode.STEPS_BACK:
        if steps_back < 1:
            raise ValueError(f"steps_back должен быть >= 1, получено {steps_back}.")
        champ_idx = next((idx for idx, c in enumerate(items) if "@champion" in c.aliases), -1)
        base_idx = champ_idx if champ_idx >= 0 else 0
        target_idx = base_idx + steps_back
        if target_idx >= len(items):
            available = ", ".join(c.version for c in items)
            raise ValueError(
                f"Недостаточно прошлых версий для отката на {steps_back} шаг(ов) назад (доступны: {available})."
            )
        return items[target_idx]

    raise ValueError(f"Режим {mode!r} требует предварительного интерактивного выбора версии.")


@task(name="switch_mlflow_champion_version")
def switch_mlflow_champion_task(
    target_version: str,
    reason: str = "manual_rollback",
    model_name: str = "bot_detector",
    tracking_uri: str | None = None,
) -> dict[str, str | None]:
    """Переключает `@champion` в MLflow на выбранную версию, ставит `@rolled_back` и сдвигает `@challenger`."""
    resolved_uri = tracking_uri or config.mlflow_tracking_uri
    client = MlflowClient(tracking_uri=resolved_uri)
    target_ver = str(target_version)
    mv_target = client.get_model_version(model_name, target_ver)
    ModelVersionInspector.validate_target_run_onnx(client, str(mv_target.run_id or ""))

    prev_ver = ModelVersionInspector.safe_alias_version(client, model_name, "champion")
    if prev_ver is not None and prev_ver != target_ver:
        client.set_registered_model_alias(model_name, "rolled_back", prev_ver)
        client.set_model_version_tag(model_name, prev_ver, "rolled_back_reason", reason)
    elif ModelVersionInspector.safe_alias_version(client, model_name, "rolled_back") == target_ver:
        try:
            client.delete_registered_model_alias(model_name, "rolled_back")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not clear @rolled_back from target v%s: %s", target_ver, exc)

    client.set_registered_model_alias(model_name, "champion", target_ver)
    client.set_model_version_tag(model_name, target_ver, "rollback_reason", reason)
    client.set_model_version_tag(model_name, target_ver, "rolled_back_at", dt.datetime.now(dt.UTC).isoformat())
    RollbackAliasManager.rebalance_after_rollback(client, model_name, target_ver, prev_ver, resolved_uri)
    logger.warning("Switched %s @champion in MLflow: v%s -> v%s (reason=%s)", model_name, prev_ver, target_ver, reason)
    return {"previous_version": prev_ver, "target_version": target_ver, "run_id": str(mv_target.run_id or "")}


@task(name="verify_triton_health")
def verify_triton_health_task() -> bool:
    """Проверка доступности и статуса health ансамбля моделей в Triton."""
    client = TritonEnsembleClient()
    try:
        return client.is_healthy()
    finally:
        client.close()


@task(name="invalidate_prediction_cache")
def invalidate_prediction_cache_task() -> int:
    """Очищает кэшированные вердикты диалогов (`dialog_score:*`) в Redis без удаления `async_task:*`."""
    removed = 0
    try:
        r_conn = redis.Redis.from_url(config.redis.url, socket_timeout=2.0)
        cursor: int = 0
        keys_to_delete: list[str] = []
        while True:
            cursor, keys = r_conn.scan(cursor=cursor, match="dialog_score:*", count=100)
            if keys:
                keys_to_delete.extend(k.decode() if isinstance(k, bytes) else str(k) for k in keys)
            if cursor == 0:
                break
        if keys_to_delete:
            removed = int(r_conn.delete(*keys_to_delete))
        r_conn.close()
        logger.info("Invalidated %d cached predictions with pattern 'dialog_score:*'", removed)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prediction cache invalidation failed (non-critical): %s", exc)
    return removed
