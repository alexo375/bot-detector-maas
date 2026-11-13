import asyncio
import json
import threading
import time
from typing import Any, ClassVar

import redis
from fastapi import FastAPI, Request
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from project.core.config import config
from project.core.logging import ProjectLogger
from project.core.metrics import (
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    INFERENCE_LATENCY,
    PREDICTIONS_TOTAL,
)
from project.drift.concept_drift.prometheus_exporter import ConceptDriftPrometheusExporter
from project.drift.data_drift.prometheus_exporter import DataDriftPrometheusExporter
from project.drift.feature_importance_diff.prometheus_exporter import FeatureImportanceDiffPrometheusExporter
from project.drift.prediction_drift.prometheus_exporter import PredictionDriftPrometheusExporter
from project.features.metadata import FeatureMetadataRegistry

logger = ProjectLogger.get_logger(__name__)

# 4. Метрики адаптивного ретрейна и объема данных
PRODUCTION_DIALOGUES_COUNT = Gauge(
    "bot_detector_production_dialogues_count",
    "Количество валидных диалогов в ClickHouse DWH за окно мониторинга",
)
MODEL_AGE_DAYS = Gauge(
    "bot_detector_model_age_days",
    "Возраст активной модели Champion в днях",
)
ADAPTIVE_RETRAIN_STATUS = Gauge(
    "bot_detector_adaptive_retrain_status",
    "Статус адаптивного ретрейна: 0=HEALTHY, 1=AWAITING_DATA, 2=RETRAIN_NEEDED, 3=RETRAINING",
)

# 6. Метрики Data Drift и качества распределений (ссылки на экспортер)
DATA_DRIFT_SHARE = DataDriftPrometheusExporter.DATA_DRIFT_SHARE
DATA_DRIFT_DETECTED = DataDriftPrometheusExporter.DATA_DRIFT_DETECTED
FEATURE_DRIFT_P_VALUE = DataDriftPrometheusExporter.FEATURE_DRIFT_P_VALUE
FEATURE_DRIFT_STATUS = DataDriftPrometheusExporter.FEATURE_DRIFT_STATUS
DATA_DRIFT_AGE_SECONDS = DataDriftPrometheusExporter.DATA_DRIFT_AGE_SECONDS


class PrometheusMetricsManager:
    """Синхронизатор метрик дрифта из Redis и инструментатор FastAPI `/metrics`."""

    _REDIS_KEYS: ClassVar[list[str]] = [
        "bot_detector:drift_metrics",
        "bot_detector:prediction_drift_metrics",
        "bot_detector:concept_drift_metrics",
        "bot_detector:feature_importance_metrics",
        "bot_detector:adaptive_decision",
    ]

    def __init__(self) -> None:
        self._redis_client: redis.Redis | None = None
        self._lock = threading.Lock()

    def _get_redis_client(self) -> redis.Redis:
        if self._redis_client is None:
            self._redis_client = redis.Redis.from_url(config.redis.url, socket_timeout=1.0)
        return self._redis_client

    @staticmethod
    def _sync_adaptive(raw_adaptive: bytes | str | None) -> None:
        if not raw_adaptive:
            return
        ad_payload = json.loads(raw_adaptive)
        PRODUCTION_DIALOGUES_COUNT.set(float(ad_payload.get("dialogues_count", 0.0)))
        MODEL_AGE_DAYS.set(float(ad_payload.get("model_age_days", 0.0)))
        ADAPTIVE_RETRAIN_STATUS.set(float(ad_payload.get("retrain_status", 0.0)))

    @staticmethod
    def _sync_data_drift(raw_data: bytes | str | None) -> None:
        if not raw_data:
            return
        payload = json.loads(raw_data)
        DATA_DRIFT_SHARE.set(float(payload.get("drift_share", 0.0)))
        DATA_DRIFT_DETECTED.set(1.0 if payload.get("dataset_drift", False) else 0.0)
        p_values = payload.get("column_p_values", {})
        drifted_features = set(payload.get("drifted_features", []))
        FEATURE_DRIFT_P_VALUE.clear()
        FEATURE_DRIFT_STATUS.clear()
        for feature, p_val in p_values.items():
            dname = FeatureMetadataRegistry.get_display_name(feature)
            FEATURE_DRIFT_P_VALUE.labels(feature=feature, display_name=dname).set(float(p_val))
            FEATURE_DRIFT_STATUS.labels(feature=feature, display_name=dname).set(
                1.0 if feature in drifted_features else 0.0
            )
        computed_at = float(payload.get("computed_at_ts", time.time()))
        DATA_DRIFT_AGE_SECONDS.set(max(0.0, time.time() - computed_at))

    @staticmethod
    def _sync_prediction_drift(raw_pred: bytes | str | None) -> None:
        if not raw_pred:
            return
        pred_payload = json.loads(raw_pred)
        PredictionDriftPrometheusExporter.PREDICTION_DRIFT_PSI.set(float(pred_payload.get("psi", 0.0)))
        PredictionDriftPrometheusExporter.PREDICTION_ENTROPY.set(float(pred_payload.get("mean_entropy", 0.0)))
        PredictionDriftPrometheusExporter.PREDICTION_DRIFT_DETECTED.set(
            1.0 if pred_payload.get("is_drifted", False) else 0.0
        )
        PredictionDriftPrometheusExporter.VERDICT_SHARE.clear()
        for verdict, share in pred_payload.get("verdict_shares", {}).items():
            PredictionDriftPrometheusExporter.VERDICT_SHARE.labels(verdict=verdict).set(float(share))

    @staticmethod
    def _sync_concept_drift(raw_concept: bytes | str | None) -> None:
        if not raw_concept:
            return
        concept_payload = json.loads(raw_concept)
        ConceptDriftPrometheusExporter.ENSEMBLE_DISAGREEMENT.set(float(concept_payload.get("disagreement_rate", 0.0)))
        ConceptDriftPrometheusExporter.FEATURE_RANK_CORRELATION.set(
            float(concept_payload.get("feature_rank_correlation", 1.0))
        )
        ConceptDriftPrometheusExporter.CONCEPT_DRIFT_DETECTED.set(
            1.0 if concept_payload.get("concept_drift_detected", False) else 0.0
        )

    @staticmethod
    def _sync_feature_importance(raw_feat_imp: bytes | str | None) -> None:
        if not raw_feat_imp:
            return
        feat_payload = json.loads(raw_feat_imp)
        FeatureImportanceDiffPrometheusExporter.FEATURE_IMPORTANCE_CURRENT.clear()
        FeatureImportanceDiffPrometheusExporter.FEATURE_IMPORTANCE_REFERENCE.clear()
        FeatureImportanceDiffPrometheusExporter.FEATURE_IMPORTANCE_DELTA.clear()
        FeatureImportanceDiffPrometheusExporter.FEATURE_RANK_SHIFT.clear()
        for rec in feat_payload.get("features", []):
            fname = rec.get("feature")
            if not fname:
                continue
            dname = FeatureMetadataRegistry.get_display_name(fname)
            FeatureImportanceDiffPrometheusExporter.FEATURE_IMPORTANCE_CURRENT.labels(
                feature=fname, display_name=dname
            ).set(float(rec.get("current_importance", 0.0)))
            FeatureImportanceDiffPrometheusExporter.FEATURE_IMPORTANCE_REFERENCE.labels(
                feature=fname, display_name=dname
            ).set(float(rec.get("previous_importance", 0.0)))
            FeatureImportanceDiffPrometheusExporter.FEATURE_IMPORTANCE_DELTA.labels(
                feature=fname, display_name=dname
            ).set(float(rec.get("importance_delta", 0.0)))
            FeatureImportanceDiffPrometheusExporter.FEATURE_RANK_SHIFT.labels(feature=fname, display_name=dname).set(
                float(rec.get("rank_shift", 0.0))
            )

    def sync_from_redis(self) -> None:
        """Синхронизация метрик дрифта из постоянного хранилища Redis перед скрейпингом Prometheus."""
        with self._lock:
            try:
                r_client = self._get_redis_client()
                raw_data, raw_pred, raw_concept, raw_feat_imp, raw_adaptive = r_client.mget(self._REDIS_KEYS)
                self._sync_adaptive(raw_adaptive)
                self._sync_data_drift(raw_data)
                self._sync_prediction_drift(raw_pred)
                self._sync_concept_drift(raw_concept)
                self._sync_feature_importance(raw_feat_imp)
            except (redis.RedisError, json.JSONDecodeError, ValueError) as exc:
                logger.debug("Failed to sync drift metrics from Redis: %s", exc)
                self._redis_client = None

    def setup(self, app: FastAPI) -> Instrumentator:
        """Инструментация FastAPI приложения с эндпоинтом `/metrics`."""

        @app.middleware("http")
        async def sync_drift_metrics_middleware(request: Request, call_next: Any) -> Any:
            if request.url.path == "/metrics":
                await asyncio.to_thread(self.sync_from_redis)
            return await call_next(request)

        instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=False,
            excluded_handlers=["/metrics", "/health", "/alerts/webhook"],
            env_var_name="ENABLE_METRICS",
        )
        instrumentator.instrument(app)
        instrumentator.expose(app, endpoint="/metrics", include_in_schema=True, tags=["Monitoring"])
        return instrumentator


metrics_manager = PrometheusMetricsManager()

__all__ = [
    "ADAPTIVE_RETRAIN_STATUS",
    "CACHE_HITS_TOTAL",
    "CACHE_MISSES_TOTAL",
    "DATA_DRIFT_AGE_SECONDS",
    "DATA_DRIFT_DETECTED",
    "DATA_DRIFT_SHARE",
    "FEATURE_DRIFT_P_VALUE",
    "FEATURE_DRIFT_STATUS",
    "INFERENCE_LATENCY",
    "MODEL_AGE_DAYS",
    "PREDICTIONS_TOTAL",
    "PRODUCTION_DIALOGUES_COUNT",
    "PrometheusMetricsManager",
    "metrics_manager",
]
