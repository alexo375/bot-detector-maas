from typing import Any

import optuna

from project.models.classical.configs.base_search_config import BaseSearchConfig


class LightGBMSearchConfig(BaseSearchConfig):
    """Декларативное пространство гиперпараметров для LightGBM."""

    def __init__(
        self,
        n_estimators: int = 750,
        learning_rate_range: tuple[float, float] = (0.015, 0.12),
        num_leaves_range: tuple[int, int] = (15, 63),
        max_depth_range: tuple[int, int] = (3, 8),
        min_child_samples_range: tuple[int, int] = (10, 50),
        min_child_samples_step: int = 5,
        subsample_range: tuple[float, float] = (0.6, 1.0),
        colsample_bytree_range: tuple[float, float] = (0.4, 1.0),
        reg_alpha_range: tuple[float, float] = (1e-3, 5.0),
        reg_lambda_range: tuple[float, float] = (1e-3, 5.0),
        tune_feature_count: bool = False,
        min_features: int = 5,
        max_features: int | None = None,
    ) -> None:
        super().__init__(
            tune_feature_count=tune_feature_count,
            min_features=min_features,
            max_features=max_features,
        )
        self.n_estimators = n_estimators
        self.learning_rate_range = learning_rate_range
        self.num_leaves_range = num_leaves_range
        self.max_depth_range = max_depth_range
        self.min_child_samples_range = min_child_samples_range
        self.min_child_samples_step = min_child_samples_step
        self.subsample_range = subsample_range
        self.colsample_bytree_range = colsample_bytree_range
        self.reg_alpha_range = reg_alpha_range
        self.reg_lambda_range = reg_lambda_range

    def get_fixed_parameters(self) -> dict[str, Any]:
        """Возвращает фиксированные параметры LightGBM."""
        return {"n_estimators": self.n_estimators}

    def sample_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """Сэмплирует пространство гиперпараметров LightGBM для Optuna."""
        params: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "learning_rate": trial.suggest_float("learning_rate", *self.learning_rate_range, log=True),
            "num_leaves": trial.suggest_int("num_leaves", *self.num_leaves_range),
            "max_depth": trial.suggest_int("max_depth", *self.max_depth_range),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                *self.min_child_samples_range,
                step=self.min_child_samples_step,
            ),
            "subsample": trial.suggest_float("subsample", *self.subsample_range),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *self.colsample_bytree_range),
            "reg_alpha": trial.suggest_float("reg_alpha", *self.reg_alpha_range, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *self.reg_lambda_range, log=True),
        }

        if self.tune_feature_count and self.max_features is not None:
            params["top_n_features"] = trial.suggest_int("top_n_features", self.min_features, self.max_features)

        return params
