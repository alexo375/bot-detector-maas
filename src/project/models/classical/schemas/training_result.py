from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass
class TrainingResult:
    """Типизированный результат обучения ансамбля моделей по фолдам."""

    oof_raw: np.ndarray
    oof_calibrated: np.ndarray
    test_raw: np.ndarray
    test_calibrated: np.ndarray
    metrics: dict[str, float]
    best_params: dict[str, Any]
    models: list[Any]
    selected_features: list[str]
    submission_df: pl.DataFrame
    run_id: str
    fold_indices: np.ndarray | None = None
    full_model: Any = None
    best_iteration: int | None = None
    feature_importances: dict[str, float] | None = None
    selected_feature_indices: list[int] | None = None
