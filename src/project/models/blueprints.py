from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelBlueprint:
    """Standardized blueprint for ML models across training pipelines and MLflow tracking."""

    run_name: str
    description: str
    tags: dict[str, str] = field(default_factory=dict)
    default_params: dict[str, Any] = field(default_factory=dict)
    n_splits: int = 5
    early_stopping_rounds: int = 40
    registered_model_name: str | None = None


CATBOOST_BLUEPRINT = ModelBlueprint(
    run_name="catboost_detector",
    description=(
        "Level-1 CatBoost gradient boosted decision tree classifier trained on tabular dialogue features "
        "with GroupKFold cross-validation and pairwise calibration."
    ),
    tags={
        "model.type": "gradient_boosting",
        "model.framework": "catboost",
        "model.stage": "level1",
        "input.features": "tabular",
        "task": "bot_classification",
    },
    default_params={
        "iterations": 750,
        "learning_rate": 0.03,
        "depth": 5,
        "eval_metric": "Logloss",
        "random_seed": 42,
        "verbose": 0,
        "allow_writing_files": False,
    },
    n_splits=5,
    early_stopping_rounds=40,
)

LIGHTGBM_BLUEPRINT = ModelBlueprint(
    run_name="lightgbm_detector",
    description=(
        "Level-1 LightGBM histogram-based gradient boosting classifier trained on selected tabular dialogue features "
        "with early stopping and GroupKFold cross-validation."
    ),
    tags={
        "model.type": "gradient_boosting",
        "model.framework": "lightgbm",
        "model.stage": "level1",
        "input.features": "tabular",
        "task": "bot_classification",
    },
    default_params={
        "n_estimators": 750,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "random_state": 42,
        "verbose": -1,
    },
    n_splits=5,
    early_stopping_rounds=40,
)

XGBOOST_BLUEPRINT = ModelBlueprint(
    run_name="xgboost_detector",
    description=(
        "Level-1 XGBoost gradient boosted classifier trained on tabular dialogue features "
        "with second-order loss optimization and GroupKFold cross-validation."
    ),
    tags={
        "model.type": "gradient_boosting",
        "model.framework": "xgboost",
        "model.stage": "level1",
        "input.features": "tabular",
        "task": "bot_classification",
    },
    default_params={
        "n_estimators": 750,
        "learning_rate": 0.03,
        "max_depth": 5,
        "random_state": 42,
        "verbosity": 0,
    },
    n_splits=5,
    early_stopping_rounds=40,
)

RUBERT_BLUEPRINT = ModelBlueprint(
    run_name="rubert_tiny2_finetune",
    description=(
        "Pretrained Russian Transformer (cointegrated/rubert-tiny2) with continual fine-tuning on dialogue texts, "
        "adaptive learning rate, PyTorch Sequence Classification head, and checkpointing in MLflow Model Registry."
    ),
    tags={
        "model.type": "transformer",
        "model.framework": "pytorch_transformers",
        "model.stage": "backbone",
        "architecture": "cointegrated/rubert-tiny2",
        "task": "bot_classification",
    },
    default_params={
        "max_epochs": 10,
        "batch_size": 64,
        "lr": 3e-5,
        "finetune_lr": 1e-5,
        "weight_decay": 0.01,
        "max_length": 128,
        "seed": 42,
    },
    registered_model_name="rubert_detector",
)
