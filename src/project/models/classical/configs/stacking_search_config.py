from typing import Any

import optuna

from project.models.classical.configs.base_search_config import BaseSearchConfig


class StackingSearchConfig(BaseSearchConfig):
    """Конфигурация пространства гиперпараметров Logistic Meta-Stacker."""

    def __init__(
        self,
        c_min: float = 0.001,
        c_max: float = 10.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
    ) -> None:
        super().__init__(tune_feature_count=False)
        self.c_min = c_min
        self.c_max = c_max
        self.solver = solver
        self.max_iter = max_iter

    def get_fixed_parameters(self) -> dict[str, Any]:
        """Возвращает фиксированные параметры LogisticRegression."""
        return {
            "solver": self.solver,
            "max_iter": self.max_iter,
        }

    def sample_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """Сэмплирует регуляризацию `C` и `fit_intercept` мета-стекера для Optuna."""
        return {
            "C": trial.suggest_float("C", self.c_min, self.c_max, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
