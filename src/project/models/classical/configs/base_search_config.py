from abc import ABC, abstractmethod
from typing import Any

import optuna


class BaseSearchConfig(ABC):
    """Базовый контракт конфигурации пространства гиперпараметров."""

    def __init__(
        self,
        tune_feature_count: bool = False,
        min_features: int = 5,
        max_features: int | None = None,
    ) -> None:
        self.tune_feature_count = tune_feature_count
        self.min_features = min_features
        self.max_features = max_features

    def get_fixed_parameters(self) -> dict[str, Any]:
        """Возвращает фиксированные (не настраиваемые Optuna) параметры модели."""
        return {}

    @abstractmethod
    def sample_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """Сэмплирует гиперпараметры модели из Optuna Trial."""
