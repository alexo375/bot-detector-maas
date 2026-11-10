import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

import aio_pika

from project.core.config import config

logger = logging.getLogger(__name__)


class BaseRabbitMQPublisher(ABC):
    """Базовый абстрактный асинхронный клиент RabbitMQ."""

    def __init__(self, rabbitmq_url: str | None = None) -> None:
        """Инициализация базового паблишера."""
        self.rabbitmq_url = rabbitmq_url or config.rabbitmq.url
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._is_connected = False
        self._last_connect_attempt = -100.0
        self._reconnect_backoff_sec = 15.0
        self._connect_lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Установка неблокирующего подключения к брокеру RabbitMQ."""
        async with self._connect_lock:
            if self.is_connected:
                return True

            now = time.monotonic()
            if now - self._last_connect_attempt < self._reconnect_backoff_sec:
                return False
            self._last_connect_attempt = now

            if self._connection and not self._connection.is_closed:
                try:
                    await self._connection.close()
                except Exception as close_err:  # noqa: BLE001
                    logger.debug("Error closing stale RabbitMQ connection before reconnect: %s", close_err)

            try:
                self._connection = await aio_pika.connect_robust(
                    self.rabbitmq_url,
                    timeout=0.5,
                )
                self._channel = await self._connection.channel()
                self._is_connected = True
                await self._declare_queues()
                logger.info("Connected to RabbitMQ at %s", self.rabbitmq_url)
                return True
            except Exception as e:  # noqa: BLE001  # Resilient fallback
                logger.warning("RabbitMQ is not available (%s). Message streaming in fallback mode.", e)
                self._is_connected = False
                self._channel = None
                return False

    @abstractmethod
    async def _declare_queues(self) -> None:
        """Декларация необходимых очередей (обязательно реализуется в подклассах)."""
        raise NotImplementedError

    async def close(self) -> None:
        """Корректное закрытие соединения с брокером при завершении работы сервиса."""
        if self._connection and not self._connection.is_closed:
            try:
                await self._connection.close()
            except Exception as close_err:  # noqa: BLE001  # Resilient fallback
                logger.debug("Error while closing RabbitMQ connection: %s", close_err)
            self._is_connected = False
            logger.info("RabbitMQ connection closed")

    @property
    def is_connected(self) -> bool:
        """Проверка активности сетевого соединения с брокером."""
        return self._is_connected and self._connection is not None and not self._connection.is_closed

    async def _publish_raw_message(self, body_dict: dict[str, Any], routing_key: str) -> bool:
        """Внутренняя отправка персистентного сообщения в указанную очередь."""
        if not self.is_connected:
            connected = await self.connect()
            if not connected or self._channel is None:
                return False

        try:
            if self._channel is None:
                return False
            body = json.dumps(body_dict, ensure_ascii=False).encode("utf-8")
            message = aio_pika.Message(
                body=body,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                timestamp=datetime.now(UTC),
            )
            await self._channel.default_exchange.publish(
                message,
                routing_key=routing_key,
            )
            return True
        except Exception as e:  # noqa: BLE001  # Resilient fallback
            logger.error("Failed to publish message to queue '%s': %s", routing_key, e)
            self._is_connected = False
            return False
