import logging

from prometheus_client import Gauge

from project.drift.data_drift.summary import DataDriftSummary
from project.features.metadata import FeatureMetadataRegistry

logger = logging.getLogger(__name__)


class DataDriftPrometheusExporter:
    """Экспортер метрик дрифта распределения признаков в Prometheus."""

    DATA_DRIFT_SHARE = Gauge(
        "bot_detector_data_drift_share",
        "Доля признаков с выявленным дрифтом распределения",
    )
    DATA_DRIFT_DETECTED = Gauge(
        "bot_detector_data_drift_detected",
        "Флаг детекции общего датасет-дрифта (0 или 1)",
    )
    FEATURE_DRIFT_P_VALUE = Gauge(
        "bot_detector_feature_drift_p_value",
        "P-value статистического теста распределения для признака",
        labelnames=["feature", "display_name"],
    )
    FEATURE_DRIFT_STATUS = Gauge(
        "bot_detector_feature_drift_status",
        "Флаг дрифта отдельного признака (0 или 1)",
        labelnames=["feature", "display_name"],
    )
    DATA_DRIFT_AGE_SECONDS = Gauge(
        "bot_detector_data_drift_age_seconds",
        "Время в секундах с момента последнего расчета дрифта данных",
    )

    def export(
        self,
        summary: DataDriftSummary,
        column_p_values: dict[str, float] | None = None,
    ) -> None:
        """Обновление метрик Prometheus на основе результатов детекции дрифта."""
        try:
            self.DATA_DRIFT_SHARE.set(summary.drift_share)
            self.DATA_DRIFT_DETECTED.set(1.0 if summary.dataset_drift else 0.0)

            if column_p_values is not None:
                self.FEATURE_DRIFT_P_VALUE.clear()
                self.FEATURE_DRIFT_STATUS.clear()
                for feature, p_val in column_p_values.items():
                    dname = FeatureMetadataRegistry.get_display_name(feature)
                    self.FEATURE_DRIFT_P_VALUE.labels(feature=feature, display_name=dname).set(p_val)
                    is_drift = 1.0 if feature in summary.drifted_features else 0.0
                    self.FEATURE_DRIFT_STATUS.labels(feature=feature, display_name=dname).set(is_drift)
            logger.info("Drift metrics successfully exported to Prometheus")
            # Сохранение снапшота метрик в постоянное хранилище Redis (TTL = 14 дней)
            # Защита: не загрязняем продуктовый Redis тестовыми синтетическими фичами во время запуска pytest
            import os

            if os.getenv("PYTEST_CURRENT_TEST"):
                return

            try:
                import json
                import time

                import redis

                from project.core.config import config

                r_client = redis.Redis.from_url(config.redis.url, socket_timeout=1.0)
                payload = {
                    "drift_share": summary.drift_share,
                    "dataset_drift": summary.dataset_drift,
                    "number_of_drifted_columns": summary.number_of_drifted_columns,
                    "number_of_columns": summary.number_of_columns,
                    "drifted_features": summary.drifted_features,
                    "column_p_values": column_p_values or {},
                    "computed_at_ts": time.time(),
                }
                r_client.set("bot_detector:drift_metrics", json.dumps(payload), ex=1209600)
                r_client.close()
                logger.info("Drift summary successfully persisted to Redis (TTL=14d)")
            except Exception as r_exc:  # noqa: BLE001  # Resilient fallback
                logger.debug("Redis persistence for drift metrics skipped or failed: %s", r_exc)
        except Exception as exc:  # noqa: BLE001  # Resilient fallback
            logger.warning("Failed to export drift metrics to Prometheus: %s", exc)
