from typing import Any

import lightgbm as lgb
import numpy as np

from project.models.classical.configs.lightgbm_search_config import LightGBMSearchConfig
from project.models.classical.trainers.base_classical_trainer import BaseClassicalTrainer


class LightGBMTrainer(BaseClassicalTrainer):
    """Специализированный тренер для моделей LightGBM."""

    @property
    def model_family(self) -> str:
        """Идентификатор семейства модели."""
        return "LGBMClassifier"

    def default_search_config(self) -> LightGBMSearchConfig:
        """Конфигурация поиска гиперпараметров по умолчанию."""
        return LightGBMSearchConfig()

    def build_model(self, params: dict[str, Any]) -> lgb.LGBMClassifier:
        """Инстанцирование классификатора LightGBM с заданными параметрами."""
        clean_params = params.copy()
        clean_params.setdefault("random_state", self.config.random_state)
        clean_params.setdefault("verbose", -1)
        clean_params.setdefault("n_jobs", 4)
        return lgb.LGBMClassifier(**clean_params)

    def fit_fold(
        self,
        model: lgb.LGBMClassifier,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
    ) -> None:
        """Обучение модели на одном фолде с ранней остановкой."""
        model.fit(
            X_tr,
            y_tr,
            eval_X=X_va,
            eval_y=y_va,
            callbacks=[lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds, verbose=False)],
        )
