from typing import Any

import optuna

from project.models.classical.configs.base_search_config import BaseSearchConfig


class CatBoostSearchConfig(BaseSearchConfig):
    """Декларативное пространство гиперпараметров для CatBoost."""

    def __init__(
        self,
        iterations: int = 750,
        depth_range: tuple[int, int] = (4, 7),
        learning_rate_range: tuple[float, float] = (0.015, 0.12),
        l2_leaf_reg_range: tuple[float, float] = (1.0, 10.0),
        random_strength_range: tuple[float, float] = (1e-3, 10.0),
        tune_feature_count: bool = False,
        min_features: int = 5,
        max_features: int | None = None,
    ) -> None:
        super().__init__(
            tune_feature_count=tune_feature_count,
            min_features=min_features,
            max_features=max_features,
        )
        self.iterations = iterations
        self.depth_range = depth_range
        self.learning_rate_range = learning_rate_range
        self.l2_leaf_reg_range = l2_leaf_reg_range
        self.random_strength_range = random_strength_range

    def get_fixed_parameters(self) -> dict[str, Any]:
        """Возвращает фиксированные параметры CatBoost."""
        return {"iterations": self.iterations}

    def sample_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """Сэмплирует пространство гиперпараметров CatBoost для Optuna."""
        params: dict[str, Any] = {
            "iterations": self.iterations,
            "learning_rate": trial.suggest_float("learning_rate", *self.learning_rate_range, log=True),
            "depth": trial.suggest_int("depth", *self.depth_range),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *self.l2_leaf_reg_range),
            "random_strength": trial.suggest_float("random_strength", *self.random_strength_range, log=True),
        }

        if self.tune_feature_count and self.max_features is not None:
            params["top_n_features"] = trial.suggest_int("top_n_features", self.min_features, self.max_features)

        return params
