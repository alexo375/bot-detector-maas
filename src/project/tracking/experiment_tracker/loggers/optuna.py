from collections.abc import Callable
from typing import Any

import mlflow


class OptunaMetricsLogger:
    """Логер метрик оптимизации гиперпараметров Optuna в активный MLflow ран."""

    @staticmethod
    def create_callback(metric_name: str = "val_logloss") -> Callable[[Any, Any], None]:
        """Возвращает callback для Optuna, логирующий метрики каждого триала со `step=trial.number`."""

        def callback(study: Any, trial: Any) -> None:
            if trial.value is not None:
                mlflow.log_metric(f"optuna_trial_{metric_name}", float(trial.value), step=trial.number)
                if hasattr(study, "best_value") and study.best_value is not None:
                    mlflow.log_metric(f"optuna_best_{metric_name}", float(study.best_value), step=trial.number)

        return callback


__all__ = ["OptunaMetricsLogger"]
