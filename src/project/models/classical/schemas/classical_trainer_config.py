from dataclasses import dataclass
from typing import Any

from project.tracking.experiment_tracker.schemas import ExperimentEnvironment


@dataclass(frozen=True)
class ClassicalTrainerConfig:
    """Конфигурация тренера классических градиентных бустингов."""

    experiment_name: str = "bot-detection"
    run_name: str = "model_run"
    description: str = ""
    n_trials: int = 300
    patience: int = 50
    n_splits: int = 5
    random_state: int = 42
    early_stopping_rounds: int = 40
    pairwise_eps: float = 0.005
    register_as: str | None = None
    dataset_name: str = "train_dataset"
    environment: ExperimentEnvironment = ExperimentEnvironment.OFFLINE
    initial_params: dict[str, Any] | None = None
