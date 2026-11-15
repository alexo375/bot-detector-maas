"""Движок принятия решений и проверок состояния для мониторинга дрифта."""

import json
import os
import time
import urllib.error
import urllib.request

import redis

from project.core.config import config
from project.core.logging import ProjectLogger
from project.pipeline.drift_monitoring.schemas import DriftMonitoringResult

logger = ProjectLogger.get_logger(__name__)


class DriftDecisionEngine:
    """Матрица адаптивных решений о необходимости переобучения и проверке блокировок."""

    @staticmethod
    def is_retrain_in_progress() -> bool:
        """Проверка активного переобучения по Redis-блокировке или активным прогонам Prefect API."""
        try:
            r_conn = redis.Redis.from_url(config.redis.url, socket_timeout=1.0)
            in_progress = bool(r_conn.get("bot_detector:retrain_in_progress"))
            r_conn.close()
            if in_progress:
                return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis retrain lock check skipped: %s", exc)

        prefect_api = os.getenv("PREFECT_API_URL")
        if not prefect_api:
            return False
        try:
            payload = json.dumps(
                {
                    "limit": 5,
                    "flow_runs": {
                        "state": {"type": {"any_": ["RUNNING", "PENDING", "SCHEDULED"]}},
                        "flow_name": {"any_": ["bot_detection_retrain_flow"]},
                    },
                }
            ).encode("utf-8")
            req = urllib.request.Request(  # noqa: S310
                f"{prefect_api.rstrip('/')}/flow_runs/filter",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2) as resp:  # noqa: S310
                runs = json.loads(resp.read())
                return bool(isinstance(runs, list) and len(runs) > 0)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.debug("Prefect active retrain check skipped: %s", exc)
            return False

    @staticmethod
    def is_retrain_in_cooldown() -> bool:
        """Проверка cooldown-паузы после завершения недавнего переобучения."""
        try:
            r_conn = redis.Redis.from_url(config.redis.url, socket_timeout=1.0)
            raw_until = r_conn.get("bot_detector:retrain_cooldown_until")
            r_conn.close()
            return raw_until is not None and float(raw_until) > time.time()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis cooldown check skipped: %s", exc)
            return False

    @staticmethod
    def build_zero_monitoring_result(
        *,
        dialogues_count: int,
        model_age_days: float,
        num_columns: int,
        retrain_status: int,
        reason: str,
        psi: float = 0.0,
        entropy: float = 0.0,
        disagreement_rate: float = 0.0,
        concept_drift_detected: bool = False,
    ) -> DriftMonitoringResult:
        """Сборка унифицированного ответа мониторинга без дрифта."""
        return {
            "drift_share": 0.0,
            "dataset_drift": False,
            "prediction_drift_psi": psi,
            "prediction_entropy": entropy,
            "ensemble_disagreement_rate": disagreement_rate,
            "concept_drift_detected": concept_drift_detected,
            "retrain_recommended": False,
            "retrain_status": retrain_status,
            "decision_reason": reason,
            "production_dialogues_count": dialogues_count,
            "model_age_days": model_age_days,
            "drifted_features": [],
            "number_of_drifted_columns": 0,
            "number_of_columns": num_columns,
        }

    @staticmethod
    def evaluate(
        *,
        any_drift: bool,
        is_stale: bool,
        has_enough_data: bool,
        should_auto_trigger: bool,
        is_in_cooldown: bool,
        production_dialogues_count: int,
        min_retrain_samples: int,
        drift_share: float,
        model_age_days: float,
    ) -> tuple[int, str, bool]:
        """Матрица решений: `0=HEALTHY, 1=AWAITING_DATA, 2=RETRAIN_NEEDED, 3=RETRAINING`."""
        if (any_drift or is_stale) and has_enough_data:
            if should_auto_trigger and is_in_cooldown:
                logger.info("ADAPTIVE DECISION: Retrain cooldown active. Skipping immediate re-trigger.")
                return 0, "retrain_cooldown_active", False

            reason = "drift_detected" if any_drift else "model_staleness_exceeded"
            mode_label = "Triggering retrain" if should_auto_trigger else "Retrain needed (manual mode,"
            logger.warning(
                "ADAPTIVE DECISION: %s reason=%s, drift=%.2f%%, dialogues=%d, model_age=%.1fd",
                mode_label,
                reason,
                drift_share * 100,
                production_dialogues_count,
                model_age_days,
            )
            return 2, reason, True

        if any_drift and not has_enough_data:
            logger.warning(
                "ADAPTIVE DECISION: Drift detected but awaiting more data (dialogues: %d < min: %d). Skipping retrain.",
                production_dialogues_count,
                min_retrain_samples,
            )
            return 1, "drift_detected_awaiting_data", False

        logger.info(
            "ADAPTIVE DECISION: System healthy (drift=%.2f%%, dialogues=%d, model_age=%.1fd). No retrain needed.",
            drift_share * 100,
            production_dialogues_count,
            model_age_days,
        )
        return 0, "system_healthy", False


__all__ = ["DriftDecisionEngine"]
