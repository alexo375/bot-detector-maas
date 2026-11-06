from typing import Any

import numpy as np
import xgboost as xgb

from project.models.classical.configs.xgboost_search_config import XGBoostSearchConfig
from project.models.classical.trainers.base_classical_trainer import BaseClassicalTrainer


class XGBoostTrainer(BaseClassicalTrainer):
    """Специализированный тренер для моделей XGBoost."""

    @property
    def model_family(self) -> str:
        """Идентификатор семейства модели."""
        return "XGBClassifier"

    def default_search_config(self) -> XGBoostSearchConfig:
        """Конфигурация поиска гиперпараметров по умолчанию."""
        return XGBoostSearchConfig()

    def build_model(self, params: dict[str, Any]) -> xgb.XGBClassifier:
        """Инстанцирование классификатора XGBoost с заданными параметрами."""
        clean_params = params.copy()
        clean_params.setdefault("eval_metric", "logloss")
        clean_params.setdefault("tree_method", "hist")
        clean_params.setdefault("random_state", self.config.random_state)
        clean_params.setdefault("n_jobs", 4)
        clean_params.setdefault("early_stopping_rounds", self.config.early_stopping_rounds)
        return xgb.XGBClassifier(**clean_params)

    def fit_fold(
        self,
        model: xgb.XGBClassifier,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
    ) -> None:
        """Обучение модели на одном фолде с ранней остановкой."""
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
