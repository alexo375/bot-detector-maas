"""Централизованный контейнер внедрения зависимостей (IoC / Service Container).

Управляет жизненным циклом и пулами соединений инфраструктурных сервисов:
Redis, RabbitMQ и Triton.
"""

import threading
from typing import Optional

from project.core.config import AppConfig, config
from project.core.logging import ProjectLogger
from project.infrastructure.rabbitmq import InferenceTaskPublisher
from project.infrastructure.redis import DialogueCache
from project.infrastructure.triton.client import TritonEnsembleClient

logger = ProjectLogger.get_logger(__name__)


class ServiceContainer:
    """Централизованный контейнер синглтонов инфраструктурных сервисов с контролем жизненного цикла."""

    _instance: Optional["ServiceContainer"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        app_config: AppConfig | None = None,
        triton_client: TritonEnsembleClient | None = None,
        cache: DialogueCache | None = None,
        publisher: InferenceTaskPublisher | None = None,
    ) -> None:
        self.config = app_config or config
        self._triton_client = triton_client
        self._cache = cache
        self._publisher = publisher

    @classmethod
    def get_instance(cls) -> "ServiceContainer":
        """Получение единого синглтон-экземпляра контейнера."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def triton_client(self) -> TritonEnsembleClient:
        """Клиент Triton Inference Server (транспорт выбирается из config.triton.protocol)."""
        if self._triton_client is None:
            triton_cfg = self.config.triton
            proto = triton_cfg.protocol.lower()
            target_url = triton_cfg.grpc_url if proto == "grpc" else triton_cfg.url
            self._triton_client = TritonEnsembleClient(
                triton_url=target_url,
                protocol=proto,
                max_batch_size=triton_cfg.max_batch_size,
            )
        return self._triton_client

    @triton_client.setter
    def triton_client(self, value: TritonEnsembleClient) -> None:
        self._triton_client = value

    @property
    def cache(self) -> DialogueCache:
        """Распределенный кэш Redis."""
        if self._cache is None:
            self._cache = DialogueCache(self.config.redis.url)
        return self._cache

    @cache.setter
    def cache(self, value: DialogueCache) -> None:
        self._cache = value

    @property
    def publisher(self) -> InferenceTaskPublisher:
        """Издатель задач инференса в RabbitMQ."""
        if self._publisher is None:
            self._publisher = InferenceTaskPublisher(self.config.rabbitmq.url)
        return self._publisher

    @publisher.setter
    def publisher(self, value: InferenceTaskPublisher) -> None:
        self._publisher = value

    async def initialize(self) -> None:
        """Инициализация соединений при старте приложения (FastAPI lifespan)."""
        await self.publisher.connect()

    async def close(self) -> None:
        """Корректное завершение работы сервисов при остановке приложения."""
        if self._publisher is not None:
            try:
                await self._publisher.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing RabbitMQ publisher: %s", exc)
            self._publisher = None

        if self._cache is not None:
            try:
                self._cache.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing Redis cache: %s", exc)
            self._cache = None

        if self._triton_client is not None:
            try:
                self._triton_client.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing Triton client: %s", exc)
            self._triton_client = None


__all__ = ["ServiceContainer"]
