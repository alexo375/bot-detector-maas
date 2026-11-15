"""Утилита синхронизации Prometheus-метрик API Gateway и baseline-состояния дрифта в Redis."""

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence

import numpy as np
import redis

from project.core.config import config
from project.core.logging import ProjectLogger
from project.drift.concept_drift import ConceptDriftDetector, ConceptDriftReport
from project.drift.data_drift.prometheus_exporter import DataDriftPrometheusExporter
from project.drift.data_drift.summary import DataDriftSummary
from project.drift.prediction_drift import PredictionDriftDetector, PredictionDriftReport

logger = ProjectLogger.get_logger(__name__)


class DriftMetricsSyncService:
    """Сервис синхронизации метрик дрифта в Redis и сброса baseline на API Gateway."""

    @staticmethod
    def trigger_api_gateway_metrics() -> None:
        """Вызов эндпоинта `/metrics` для обновления Gauges из Redis."""
        env_url = os.getenv("API_GATEWAY_URL")
        candidates = [f"{env_url.rstrip('/')}/metrics"] if env_url else []
        candidates.extend(["http://api-gateway:8080/metrics", "http://localhost:8080/metrics"])
        for url in candidates:
            try:
                req = urllib.request.Request(url, method="GET")  # noqa: S310
                with urllib.request.urlopen(req, timeout=3) as resp:  # noqa: S310
                    resp.read()
                    return
            except (urllib.error.URLError, OSError) as exc:
                logger.debug("Could not reach API gateway metrics endpoint %s: %s", url, exc)

    @staticmethod
    def persist_adaptive_decision(
        dialogues_count: int,
        model_age_days: float,
        status: int,
        reason: str,
    ) -> None:
        """Сохранение снапшота адаптивного решения в Redis без вызова `/metrics`."""
        try:
            r_conn = redis.Redis.from_url(config.redis.url, socket_timeout=1.0)
            r_conn.set(
                "bot_detector:adaptive_decision",
                json.dumps(
                    {
                        "dialogues_count": dialogues_count,
                        "model_age_days": model_age_days,
                        "retrain_status": status,
                        "reason": reason,
                    }
                ),
            )
            r_conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to store adaptive decision metrics: %s", exc)

    @classmethod
    def reset_baseline(
        cls,
        feature_names: Sequence[str],
        *,
        drift_threshold: float = 0.20,
        dialogues_count: int = 0,
        model_age_days: float = 0.0,
        retrain_status: int = 0,
        reason: str = "system_healthy",
    ) -> tuple[PredictionDriftReport, ConceptDriftReport]:
        """Сброс data/prediction/concept drift и adaptive_decision в здоровый baseline."""
        feat_list = list(feature_names)
        zero_summary = DataDriftSummary(
            dataset_drift=False,
            drift_share=0.0,
            number_of_drifted_columns=0,
            number_of_columns=len(feat_list),
            drift_threshold=drift_threshold,
            retrain_recommended=False,
            drifted_features=[],
        )
        DataDriftPrometheusExporter().export(zero_summary, column_p_values=dict.fromkeys(feat_list, 1.0))

        empty_arr = np.array([], dtype=np.float64)
        pred_report = PredictionDriftDetector().detect_drift(empty_arr, empty_arr, [])
        concept_report = ConceptDriftDetector().detect_drift(model_probs={})
        cls.persist_adaptive_decision(dialogues_count, model_age_days, retrain_status, reason)
        return pred_report, concept_report


__all__ = ["DriftMetricsSyncService"]
