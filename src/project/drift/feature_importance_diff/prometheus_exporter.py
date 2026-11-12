import json
import logging
import os

from prometheus_client import Gauge

from project.drift.feature_importance_diff.schemas.report import FeatureImportanceDiffReport
from project.features.metadata import FeatureMetadataRegistry

logger = logging.getLogger(__name__)


class FeatureImportanceDiffPrometheusExporter:
    """Экспортер метрик дрифта важности признаков в Prometheus."""

    FEATURE_IMPORTANCE_CURRENT = Gauge(
        "bot_detector_feature_importance_current",
        "Текущая важность признака в модели (Challenger)",
        labelnames=["feature", "display_name"],
    )
    FEATURE_IMPORTANCE_REFERENCE = Gauge(
        "bot_detector_feature_importance_reference",
        "Эталонная важность признака в модели (Champion)",
        labelnames=["feature", "display_name"],
    )
    FEATURE_IMPORTANCE_DELTA = Gauge(
        "bot_detector_feature_importance_delta",
        "Изменение важности признака (Challenger - Champion)",
        labelnames=["feature", "display_name"],
    )
    FEATURE_RANK_SHIFT = Gauge(
        "bot_detector_feature_rank_shift",
        "Смещение ранга важности признака в топ-рейтинге",
        labelnames=["feature", "display_name"],
    )

    def export(self, report: FeatureImportanceDiffReport) -> None:
        """Обновление метрик Prometheus и сохранение снапшота в Redis."""
        try:
            for rec in report.features:
                dname = FeatureMetadataRegistry.get_display_name(rec.feature)
                self.FEATURE_IMPORTANCE_CURRENT.labels(feature=rec.feature, display_name=dname).set(
                    rec.current_importance
                )
                self.FEATURE_IMPORTANCE_REFERENCE.labels(feature=rec.feature, display_name=dname).set(
                    rec.previous_importance
                )
                self.FEATURE_IMPORTANCE_DELTA.labels(feature=rec.feature, display_name=dname).set(rec.importance_delta)
                self.FEATURE_RANK_SHIFT.labels(feature=rec.feature, display_name=dname).set(rec.rank_shift)
            logger.info("Feature importance diff metrics successfully exported to Prometheus")

            if os.getenv("PYTEST_CURRENT_TEST"):
                return

            try:
                import redis

                from project.core.config import config

                r_client = redis.Redis.from_url(config.redis.url, socket_timeout=1.0)
                r_client.set("bot_detector:feature_importance_metrics", json.dumps(report.to_dict()), ex=1209600)
                r_client.close()
                logger.info("Feature importance diff report persisted to Redis (TTL=14d)")
            except Exception as r_exc:  # noqa: BLE001
                logger.debug("Redis persistence for feature importance diff failed: %s", r_exc)
        except Exception as exc:  # noqa: BLE001  # Resilient fallback
            logger.warning("Failed to export feature importance diff to Prometheus: %s", exc)
