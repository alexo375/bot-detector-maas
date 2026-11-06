from typing import Any

import catboost as cb
import numpy as np

from project.models.classical.configs.catboost_search_config import CatBoostSearchConfig
from project.models.classical.trainers.base_classical_trainer import BaseClassicalTrainer


class CatBoostTrainer(BaseClassicalTrainer):
    """Специализированный тренер для моделей CatBoost."""

    @property
    def model_family(self) -> str:
        """Идентификатор семейства модели."""
        return "CatBoostClassifier"

    def default_search_config(self) -> CatBoostSearchConfig:
        """Конфигурация поиска гиперпараметров по умолчанию."""
        return CatBoostSearchConfig()

    def build_model(self, params: dict[str, Any]) -> cb.CatBoostClassifier:
        """Инстанцирование классификатора CatBoost с заданными параметрами."""
        clean_params = params.copy()
        clean_params.setdefault("random_seed", self.config.random_state)
        clean_params.setdefault("verbose", False)
        clean_params.setdefault("allow_writing_files", False)
        clean_params.setdefault("thread_count", 4)
        return cb.CatBoostClassifier(**clean_params)

    def fit_fold(
        self,
        model: cb.CatBoostClassifier,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
    ) -> None:
        """Обучение модели на одном фолде с ранней остановкой."""
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=False,
        )
