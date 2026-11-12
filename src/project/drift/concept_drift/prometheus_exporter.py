"""Экспортер метрик Concept Drift в Prometheus и постоянное хранилище Redis."""

import json
import logging
import os

from prometheus_client import Gauge

from project.drift.concept_drift.schemas import ConceptDriftReport

logger = logging.getLogger(__name__)


class ConceptDriftPrometheusExporter:
    """Экспортер метрик сдвига концепта и рассогласования моделей в Prometheus."""

    ENSEMBLE_DISAGREEMENT = Gauge(
        "bot_detector_ensemble_disagreement_rate",
        "Доля диалогов с рассогласованием предсказаний в ансамбле моделей",
    )
    FEATURE_RANK_CORRELATION = Gauge(
        "bot_detector_feature_rank_correlation",
        "Ранговая корреляция Спирмена важности признаков (Champion vs Challenger)",
    )
    CONCEPT_DRIFT_DETECTED = Gauge(
        "bot_detector_concept_drift_detected",
        "Флаг обнаружения сдвига концепта (0 или 1)",
    )

    def export(self, report: ConceptDriftReport) -> None:
        """Обновление метрик Prometheus и сохранение снапшота в Redis."""
        try:
            self.ENSEMBLE_DISAGREEMENT.set(report.disagreement_rate)
            self.FEATURE_RANK_CORRELATION.set(report.feature_rank_correlation)
            self.CONCEPT_DRIFT_DETECTED.set(1.0 if report.concept_drift_detected else 0.0)

            logger.info("Concept drift metrics successfully exported to Prometheus")

            # В тестах не модифицируем продуктовый Redis
            if os.getenv("PYTEST_CURRENT_TEST"):
                return

            try:
                import redis

                from project.core.config import config

                r_client = redis.Redis.from_url(config.redis.url, socket_timeout=1.0)
                r_client.set("bot_detector:concept_drift_metrics", json.dumps(report.to_dict()), ex=1209600)
                r_client.close()
                logger.info("Concept drift report persisted to Redis (TTL=14d)")
            except Exception as r_exc:  # noqa: BLE001
                logger.debug("Redis persistence for concept drift failed: %s", r_exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to export concept drift metrics: %s", exc)
