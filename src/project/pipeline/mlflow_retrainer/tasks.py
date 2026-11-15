"""Prefect-задача запуска переобучения и оценки ансамбля через `RetrainingGatekeeper`."""

from typing import Any

from prefect import task

from project.core.logging import ProjectLogger
from project.pipeline.mlflow_retrainer.components import RetrainingGatekeeper
from project.pipeline.mlflow_retrainer.schemas import RetrainEvaluationResult

logger = ProjectLogger.get_logger(__name__)


@task(name="retrain_models")
def retrain_models_task(
    fast_mode: bool = False,
    n_estimators: int | None = None,
) -> RetrainEvaluationResult | dict[str, Any]:
    """Переобучение 3 GBDT + Meta-Stacker, валидация Champion vs Challenger и запись в MLflow."""
    gatekeeper = RetrainingGatekeeper(
        fast_mode=fast_mode,
        n_estimators=n_estimators,
    )
    return gatekeeper.retrain_and_evaluate()
