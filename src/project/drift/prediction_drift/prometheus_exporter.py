"""Экспортер метрик Prediction Drift в Prometheus и постоянное хранилище Redis."""

import json
import logging
import os

from prometheus_client import Gauge

from project.drift.prediction_drift.schemas import PredictionDriftReport

logger = logging.getLogger(__name__)


class PredictionDriftPrometheusExporter:
    """Экспортер метрик Prediction Drift и энтропии моделей в Prometheus."""

    PREDICTION_DRIFT_PSI = Gauge(
        "bot_detector_prediction_drift_psi",
        "Population Stability Index (PSI) распределения вероятностей моделей",
    )
    PREDICTION_ENTROPY = Gauge(
        "bot_detector_prediction_entropy",
        "Средняя энтропия Шеннона предсказаний моделей (мера неопределенности)",
    )
    PREDICTION_DRIFT_DETECTED = Gauge(
        "bot_detector_prediction_drift_detected",
        "Флаг критического дрифта вероятностей предсказаний (0 или 1)",
    )
    VERDICT_SHARE = Gauge(
        "bot_detector_verdict_share",
        "Доля выданных вердиктов по классам",
        labelnames=["verdict"],
    )

    def export(self, report: PredictionDriftReport) -> None:
        """Обновление метрик Prometheus и сохранение снапшота в Redis."""
        try:
            self.PREDICTION_DRIFT_PSI.set(report.psi)
            self.PREDICTION_ENTROPY.set(report.mean_entropy)
            self.PREDICTION_DRIFT_DETECTED.set(1.0 if report.is_drifted else 0.0)

            for verdict, share in report.verdict_shares.items():
                self.VERDICT_SHARE.labels(verdict=verdict).set(share)

            logger.info("Prediction drift metrics successfully exported to Prometheus")

            # В тестах не модифицируем продуктовый Redis
            if os.getenv("PYTEST_CURRENT_TEST"):
                return

            try:
                import redis

                from project.core.config import config

                r_client = redis.Redis.from_url(config.redis.url, socket_timeout=1.0)
                r_client.set("bot_detector:prediction_drift_metrics", json.dumps(report.to_dict()), ex=1209600)
                r_client.close()
                logger.info("Prediction drift report persisted to Redis (TTL=14d)")
            except Exception as r_exc:  # noqa: BLE001
                logger.debug("Redis persistence for prediction drift failed: %s", r_exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to export prediction drift metrics: %s", exc)
