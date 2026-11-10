"""Высокоуровневый клиент к нативному ансамблю Triton Inference Server."""

import httpx
import numpy as np

from project.core.config import config
from project.core.exceptions import TritonServerError
from project.core.logging import ProjectLogger
from project.infrastructure.triton.circuit_breaker import CircuitBreaker
from project.infrastructure.triton.transport.base import BaseTritonTransport
from project.infrastructure.triton.transport.factory import TritonTransportFactory

logger = ProjectLogger.get_logger(__name__)


class TritonEnsembleClient:
    """Высокоуровневый клиент к нативному ансамблю Triton (`bot_ensemble`).

    Вся оркестрация между базовыми моделями (CatBoost, LightGBM, XGBoost)
    и мета-стекером выполняется внутри ядра Triton Inference Server.

    Архитектура:
    - Делегирует сетевой транспорт классу BaseTritonTransport (HTTP REST или бинарный gRPC).
    - Прозрачно разбивает крупные батчи по лимиту `max_batch_size`.
    - Защищен паттерном Circuit Breaker для предотвращения каскадных сбоев при деградации бэкенда.
    """

    def __init__(
        self,
        triton_url: str | None = None,
        protocol: str | None = None,
        client: httpx.Client | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        transport: BaseTritonTransport | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        """Инициализация клиента Triton с транспортом и защитой Circuit Breaker."""
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.transport = transport or TritonTransportFactory.create(
            protocol=protocol,
            url=triton_url,
            client=client,
        )
        self.max_batch_size = max(1, max_batch_size or config.triton.max_batch_size)

    def infer(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Отправка батчевого запроса в ансамбль Triton с авто-чанкингом и защитой Circuit Breaker.

        Args:
            features: Матрица вещественных признаков [batch_size, n_features] типа float32.

        Returns:
            Кортеж (p_meta, p_cb, p_lgb, p_xgb).

        """
        n_rows = len(features)
        if n_rows <= self.max_batch_size:
            return self.circuit_breaker.execute(
                self.transport.infer,
                features=features,
            )

        chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for start in range(0, n_rows, self.max_batch_size):
            end = start + self.max_batch_size
            chunk_res = self.circuit_breaker.execute(
                self.transport.infer,
                features=features[start:end],
            )
            chunks.append(chunk_res)

        return (
            np.concatenate([c[0] for c in chunks]),
            np.concatenate([c[1] for c in chunks]),
            np.concatenate([c[2] for c in chunks]),
            np.concatenate([c[3] for c in chunks]),
        )

    def _probe_health(self) -> bool:
        """Внутренняя проверка здоровья, преобразующая False в ошибку для Circuit Breaker."""
        healthy = self.transport.is_healthy()
        if not healthy:
            raise TritonServerError("Triton health check returned False")
        return True

    def is_healthy(self) -> bool:
        """Проверка доступности сервера через активный транспорт."""
        try:
            return self.circuit_breaker.execute(self._probe_health)
        except (TritonServerError, OSError) as exc:
            logger.warning("Ошибка проверки здоровья Triton сервера: %s", exc)
            return False

    def close(self) -> None:
        """Закрытие сетевых сессий активного транспорта."""
        self.transport.close()


__all__ = ["TritonEnsembleClient"]
