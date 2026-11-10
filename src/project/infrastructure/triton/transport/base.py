"""Базовый абстрактный контракт транспорта к Triton Inference Server."""

from abc import ABC, abstractmethod

import numpy as np


class BaseTritonTransport(ABC):
    """Абстрактный интерфейс сетевого транспорта к нативному ансамблю Triton."""

    @abstractmethod
    def infer(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Отправка батчевого запроса инференса в модель Triton.

        Args:
            features: Матрица вещественных признаков [batch_size, n_features] типа float32.

        Returns:
            Кортеж (p_meta, p_cb, p_lgb, p_xgb).

        """

    @abstractmethod
    def is_healthy(self) -> bool:
        """Проверка доступности сервера инференса."""

    @abstractmethod
    def close(self) -> None:
        """Освобождение сетевых ресурсов и закрытие сессий."""


__all__ = ["BaseTritonTransport"]
