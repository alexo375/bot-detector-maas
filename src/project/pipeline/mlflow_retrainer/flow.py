import argparse
import json
import time
from typing import Any

import redis
from prefect import flow

from project.core.config import config
from project.core.logging import LoggingConfigurator, ProjectLogger
from project.core.utils import Reproducibility
from project.drift.data_drift.fetcher import DWHFeatureFetcher
from project.drift.model_age import ChampionMetadataReader
from project.pipeline.drift_monitoring.tasks import (
    detect_feature_drift_task,
    fetch_dwh_data_task,
)
from project.pipeline.metrics_sync import DriftMetricsSyncService
from project.pipeline.mlflow_retrainer.schemas import (
    FlowExecutionResult,
    FlowStatus,
    GatekeeperMetrics,
)
from project.pipeline.mlflow_retrainer.tasks import retrain_models_task
from project.pipeline.serving.tasks import (
    export_onnx_task,
    hot_reload_triton_task,
)

logger = ProjectLogger.get_logger(__name__)


class RetrainStateCoordinator:
    """Управление Redis-блокировками, cooldown и синхронизацией baseline после ретрейна."""

    @staticmethod
    def update(
        in_progress: bool,
        status: int | None = None,
        reason: str | None = None,
        reset_drift_baseline: bool = False,
    ) -> None:
        """Обновляет флаг активного переобучения и статус метрик дрифта в Redis."""
        try:
            r_conn = redis.Redis.from_url(config.redis.url, socket_timeout=1.0)
            if in_progress:
                r_conn.set("bot_detector:retrain_in_progress", "1", ex=7200)
            else:
                r_conn.delete("bot_detector:retrain_in_progress")
                if reset_drift_baseline:
                    r_conn.delete("bot_detector:retrain_cooldown_until", "bot_detector:champion_effective_since_ts")
                elif config.schedule.retrain_cooldown_sec > 0 and reason != "promoted":
                    r_conn.set(
                        "bot_detector:retrain_cooldown_until",
                        str(time.time() + config.schedule.retrain_cooldown_sec),
                        ex=config.schedule.retrain_cooldown_sec,
                    )
            if not reset_drift_baseline and status is not None:
                raw = r_conn.get("bot_detector:adaptive_decision")
                payload = json.loads(raw) if raw else {}
                payload["retrain_status"] = int(status)
                if reason is not None:
                    payload["reason"] = reason
                r_conn.set("bot_detector:adaptive_decision", json.dumps(payload))
            r_conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis retrain state update skipped: %s", exc)

        if reset_drift_baseline:
            DriftMetricsSyncService.reset_baseline(
                DWHFeatureFetcher().get_feature_names(),
                dialogues_count=0,
                model_age_days=0.0,
                retrain_status=0 if status is None else int(status),
                reason=reason or "promoted",
            )
        DriftMetricsSyncService.trigger_api_gateway_metrics()


class RetrainFlowDeployer:
    """Регистратор deployment переобучения моделей в Prefect Server."""

    @staticmethod
    def deploy() -> str:
        """Регистрация event-driven и on-demand деплоймента переобучения в Prefect Server."""
        from prefect.client.schemas.objects import ConcurrencyLimitConfig, ConcurrencyLimitStrategy
        from prefect.deployments.runner import EntrypointType

        deployment: Any = bot_detection_retrain_flow.to_deployment(
            name="bot-detection-retraining",
            work_pool_name="default-agent-pool",
            schedules=[],
            concurrency_limit=ConcurrencyLimitConfig(
                limit=1,
                collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW,
            ),
            tags=["retraining", "mlops", "event-driven", "on-demand"],
            description="Адаптивный пайплайн переобучения 3 GBDT + Stacking, валидации Gatekeeper и деплоя в Triton",
            entrypoint_type=EntrypointType.MODULE_PATH,
            parameters={
                "force_retrain": False,
                "drift_threshold": config.schedule.drift_threshold,
                "fast_mode": False,
                "n_estimators": config.retrain.n_estimators,
            },
        )
        deployment.entrypoint = f"project.pipeline.mlflow_retrainer.flow:{bot_detection_retrain_flow.name}"
        dep_id = deployment.apply()
        logger.info("Successfully registered deployment 'bot-detection-retraining' in Prefect (ID: %s)", dep_id)
        return str(dep_id)

    @staticmethod
    def check_drift_summary(force_retrain: bool, effective_threshold: float) -> Any:
        """Выполняет проверку дрейфа с безопасным пропуском при `force_retrain=True`."""
        try:
            if (
                force_retrain
                and DWHFeatureFetcher().count_recent_dialogues(days=config.schedule.drift_window_days) < 10
            ):
                logger.warning(
                    "Drift check skipped under force_retrain=True: insufficient production dialogues (< 10)."
                )
                return {
                    "retrain_recommended": False,
                    "dataset_drift": False,
                    "drift_share": 0.0,
                    "reason": "skipped_on_force_retrain",
                }
            data = fetch_dwh_data_task()
            return detect_feature_drift_task(data=data, drift_threshold=effective_threshold)
        except Exception as exc:
            if not force_retrain:
                raise
            logger.warning("Drift check skipped under force_retrain=True (insufficient chat_logs): %s", exc)
            return {
                "retrain_recommended": False,
                "dataset_drift": False,
                "drift_share": 0.0,
                "reason": "skipped_on_force_retrain",
            }


@flow(name="bot_detection_retrain_flow")
def bot_detection_retrain_flow(
    force_retrain: bool = False,
    drift_threshold: float | None = None,
    fast_mode: bool = False,
    n_estimators: int | None = None,
) -> FlowExecutionResult:
    """Оркестрирующий Flow адаптивного переобучения моделей ML-системы."""
    Reproducibility.seed(42)
    effective_threshold = config.schedule.drift_threshold if drift_threshold is None else float(drift_threshold)
    logger.info("Starting Bot Detection Retrain Flow (force=%s, fast_mode=%s)", force_retrain, fast_mode)

    if not force_retrain:
        fetcher = DWHFeatureFetcher()
        new_dialogues_count = fetcher.count_recent_dialogues(days=config.schedule.drift_window_days)
        if new_dialogues_count < config.schedule.min_retrain_samples:
            logger.info(
                "Retraining skipped: insufficient production samples (%d < %d)",
                new_dialogues_count,
                config.schedule.min_retrain_samples,
            )
            return FlowExecutionResult(
                status=FlowStatus.SKIPPED,
                reason="insufficient_data",
                drift_summary={
                    "new_dialogues_count": new_dialogues_count,
                    "min_required": config.schedule.min_retrain_samples,
                },
            )

    drift_summary = RetrainFlowDeployer.check_drift_summary(force_retrain, effective_threshold)
    champion_info = ChampionMetadataReader.read("bot_detector")
    model_age_days = float(champion_info.get("age_days", 0.0))
    is_stale = model_age_days >= config.schedule.max_model_staleness_days
    retrain_recommended = (
        drift_summary.get("retrain_recommended", False)
        if isinstance(drift_summary, dict)
        else drift_summary.retrain_recommended
    )

    if not (force_retrain or retrain_recommended or is_stale):
        logger.info(
            "No significant drift detected and model is fresh (age=%.1fd < %dd). Skipping retraining.",
            model_age_days,
            config.schedule.max_model_staleness_days,
        )
        return FlowExecutionResult(
            status=FlowStatus.SKIPPED,
            reason="no_drift_detected",
            drift_summary=drift_summary,
        )

    RetrainStateCoordinator.update(in_progress=True, status=3, reason="retraining_in_progress")
    promoted = False
    sync_promoted_baseline = False
    try:
        retrain_result = retrain_models_task(fast_mode=fast_mode, n_estimators=n_estimators)
        raw_metrics = retrain_result["metrics"] if isinstance(retrain_result, dict) else retrain_result.metrics
        metrics = GatekeeperMetrics.from_dict(raw_metrics) if isinstance(raw_metrics, dict) else raw_metrics
        promoted = metrics.promoted

        if promoted:
            export_onnx_task()
            reload_status: dict[str, Any] = hot_reload_triton_task()
            sync_promoted_baseline = True
        else:
            logger.warning("Challenger model was rejected by Gatekeeper. Triton reload skipped.")
            reload_status = {"skipped": True, "reason": metrics.reason or "gatekeeper_rejected"}
    finally:
        RetrainStateCoordinator.update(
            in_progress=False,
            status=0,
            reason="promoted" if promoted else "retrain_completed_cooldown",
            reset_drift_baseline=sync_promoted_baseline,
        )

    diff_data = retrain_result.get("diff_data") if isinstance(retrain_result, dict) else retrain_result.diff_data
    return FlowExecutionResult(
        status=FlowStatus.COMPLETED,
        retrained=True,
        promoted=promoted,
        decision=metrics.decision,
        metrics=metrics,
        feature_importance_diff=diff_data,
        drift_summary=drift_summary,
        triton_reload=reload_status,
    )


def main() -> None:
    """CLI точка входа для запуска Flow."""
    parser = argparse.ArgumentParser(description="Prefect Bot Detection Retraining Pipeline")
    parser.add_argument("--deploy", action="store_true", help="Register retraining schedule in Prefect UI")
    parser.add_argument("--force", action="store_true", help="Force retraining regardless of drift status")
    parser.add_argument("--threshold", type=float, default=None, help="Drift threshold")
    parser.add_argument("--fast", action="store_true", help="Fast/Lite mode for testing (few trees, 3 Optuna trials)")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=config.retrain.n_estimators,
        help="Number of trees for GBDTs",
    )
    args = parser.parse_args()

    LoggingConfigurator.configure()
    if args.deploy:
        RetrainFlowDeployer.deploy()
        return

    result = bot_detection_retrain_flow(
        force_retrain=args.force,
        drift_threshold=args.threshold,
        fast_mode=args.fast,
        n_estimators=args.n_estimators,
    )
    logger.info("Bot Detection Retrain Flow Finished: status=%s, promoted=%s", result.status, result.promoted)


if __name__ == "__main__":
    main()
