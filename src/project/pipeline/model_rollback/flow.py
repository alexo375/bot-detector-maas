"""Prefect Flow и точка входа для ручного отката ансамбля моделей через MLflow -> Triton."""

import argparse
import json
from typing import Any

from prefect import flow

from project.core.logging import LoggingConfigurator, ProjectLogger
from project.pipeline.model_rollback.inspector import ModelVersionInspector
from project.pipeline.model_rollback.interactive import RollbackPromptUI
from project.pipeline.model_rollback.schemas import (
    ModelVersionCandidate,
    RollbackMode,
    RollbackResult,
)
from project.pipeline.model_rollback.tasks import (
    check_no_active_retrain_task,
    invalidate_prediction_cache_task,
    reset_window_after_rollback_task,
    resolve_rollback_target_task,
    switch_mlflow_champion_task,
    verify_triton_health_task,
)
from project.pipeline.serving.tasks import export_onnx_task, hot_reload_triton_task

logger = ProjectLogger.get_logger(__name__)


class RollbackFlowDeployer:
    """Регистратор on-demand Prefect deployment для отката моделей."""

    @staticmethod
    def deploy() -> str:
        """Регистрация on-demand deployment отката моделей в Prefect Server."""
        from prefect.client.schemas.objects import ConcurrencyLimitConfig, ConcurrencyLimitStrategy
        from prefect.deployments.runner import EntrypointType

        deployment: Any = bot_detection_rollback_flow.to_deployment(
            name="bot-detection-rollback",
            work_pool_name="default-agent-pool",
            schedules=[],
            concurrency_limit=ConcurrencyLimitConfig(
                limit=1,
                collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW,
            ),
            tags=["rollback", "mlops", "on-demand"],
            description="Ручной откат ансамбля моделей (MLflow @champion -> Triton ONNX + словарь + кэш)",
            entrypoint_type=EntrypointType.MODULE_PATH,
        )
        deployment.entrypoint = f"project.pipeline.model_rollback.flow:{bot_detection_rollback_flow.name}"
        dep_id = deployment.apply()
        logger.info("Successfully registered deployment 'bot-detection-rollback' in Prefect (ID: %s)", dep_id)
        return str(dep_id)


@flow(name="bot_detection_rollback_flow")
def bot_detection_rollback_flow(
    mode: RollbackMode = RollbackMode.INTERACTIVE_PICK,
    steps_back: int = 1,
    target_version: str = "",
    invalidate_cache: bool = True,
    reset_drift_baseline: bool = True,
    reason: str = "manual_rollback",
) -> RollbackResult:
    """Ручной откат ансамбля моделей через MLflow Registry и Hot Reload в Triton."""
    check_no_active_retrain_task()
    candidates = ModelVersionInspector.list_candidates()
    if not candidates:
        raise ValueError("В MLflow Model Registry отсутствуют совместимые версии 'bot_detector' для отката.")

    chosen: ModelVersionCandidate
    if mode == RollbackMode.INTERACTIVE_PICK:
        chosen_version = RollbackPromptUI.prompt_target_version(candidates)
        chosen = resolve_rollback_target_task(
            mode=RollbackMode.EXACT_VERSION,
            target_version=chosen_version,
            candidates=candidates,
        )
    else:
        chosen = resolve_rollback_target_task(
            mode=mode,
            steps_back=steps_back,
            target_version=target_version,
            candidates=candidates,
        )

    switch_info = switch_mlflow_champion_task(target_version=chosen.version, reason=reason)
    export_onnx_task()
    reload_status = hot_reload_triton_task()
    triton_healthy = verify_triton_health_task()
    cleared_cache = invalidate_prediction_cache_task() if invalidate_cache else 0
    reset_window_after_rollback_task(target_version=chosen.version, reset_drift_baseline=reset_drift_baseline)

    return RollbackResult(
        status="completed",
        previous_version=switch_info["previous_version"],
        rolled_back_to_version=chosen.version,
        target_run_id=chosen.run_id,
        triton_reload=reload_status,
        triton_healthy=triton_healthy,
        cache_invalidated_keys=cleared_cache,
        drift_window_reset=reset_drift_baseline,
        reason=reason,
    )


def main() -> None:
    """CLI точка входа для регистрации деплоймента и ручного отката из терминала."""
    parser = argparse.ArgumentParser(description="Prefect Bot Detection Model Rollback Flow")
    parser.add_argument("--deploy", action="store_true", help="Register rollback deployment in Prefect UI")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.value for m in RollbackMode],
        default=RollbackMode.STEPS_BACK.value,
        help="Rollback selection mode",
    )
    parser.add_argument("--steps-back", type=int, default=1, help="Number of versions to roll back")
    parser.add_argument("--target-version", type=str, default="", help="Exact MLflow model version")
    parser.add_argument("--keep-cache", action="store_true", help="Do not clear Redis dialog_score:* cache")
    parser.add_argument("--keep-window", action="store_true", help="Do not reset drift window/metrics")
    parser.add_argument("--reason", type=str, default="manual_cli_rollback", help="Rollback reason tag")
    args = parser.parse_args()

    LoggingConfigurator.configure()
    if args.deploy:
        RollbackFlowDeployer.deploy()
        return

    res = bot_detection_rollback_flow(
        mode=RollbackMode(args.mode),
        steps_back=args.steps_back,
        target_version=args.target_version,
        invalidate_cache=not args.keep_cache,
        reset_drift_baseline=not args.keep_window,
        reason=args.reason,
    )
    logger.info("Rollback result: %s", json.dumps(res.model_dump(), ensure_ascii=False))


if __name__ == "__main__":
    main()
