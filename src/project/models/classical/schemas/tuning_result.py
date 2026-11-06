from dataclasses import dataclass
from typing import Any

import optuna


@dataclass
class TuningResult:
    """Типизированный результат процедуры байесовской оптимизации гиперпараметров."""

    best_params: dict[str, Any]
    best_score: float
    optimal_features_count: int | None = None
    study: optuna.Study | None = None
