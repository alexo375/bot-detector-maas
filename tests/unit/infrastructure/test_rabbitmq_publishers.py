"""Модульные тесты для паблишеров RabbitMQ (InferenceTaskPublisher, AuditLogPublisher, BaseRabbitMQPublisher)."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project.infrastructure.rabbitmq.publishers.audit_log_publisher import AuditLogPublisher
from project.infrastructure.rabbitmq.publishers.base_publisher import BaseRabbitMQPublisher
from project.infrastructure.rabbitmq.publishers.inference_task_publisher import InferenceTaskPublisher
from project.infrastructure.rabbitmq.schemas.task import InferenceTask


class ConcretePublisher(BaseRabbitMQPublisher):
    """Тестовая реализация абстрактного паблишера."""

    async def _declare_queues(self) -> None:
        if self._channel is not None:
            await self._channel.declare_queue("test_queue", durable=True)


@pytest.mark.asyncio
async def test_base_publisher_connect_and_reconnect_backoff() -> None:
    """Проверка логики троттлинга повторных подключений при сбоях брокера."""
    pub = ConcretePublisher(rabbitmq_url="amqp://guest:guest@localhost:5672/")

    with patch("aio_pika.connect_robust", side_effect=ConnectionError("Failed")):
        # Первая попытка подключения -> неудачно
        res1 = await pub.connect()
        assert res1 is False
        assert pub.is_connected is False

        # Немедленная вторая попытка -> отсекается backoff-интервалом (15 сек)
        res2 = await pub.connect()
        assert res2 is False


@pytest.mark.asyncio
async def test_base_publisher_close_handling() -> None:
    """Проверка корректного закрытия активного соединения."""
    pub = ConcretePublisher(rabbitmq_url="amqp://guest:guest@localhost:5672/")
    mock_conn = MagicMock()
    mock_conn.is_closed = False
    mock_conn.close = AsyncMock()
    pub._connection = mock_conn

    await pub.close()
    mock_conn.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_inference_task_publisher_success() -> None:
    """Проверка успешной публикации задачи инференса в очередь."""
    pub = InferenceTaskPublisher(rabbitmq_url="amqp://guest:guest@localhost:5672/")
    mock_exchange = MagicMock()
    mock_exchange.publish = AsyncMock()

    mock_channel = MagicMock()
    mock_channel.default_exchange = mock_exchange
    pub._channel = mock_channel

    mock_conn = MagicMock()
    mock_conn.is_closed = False
    pub._connection = mock_conn
    pub._is_connected = True

    task = InferenceTask(
        task_id="task-xyz",
        dialogues=[{"dialog_id": "d1", "messages": [{"participant_index": 0, "text": "hello"}]}],
    )

    success = await pub.publish_task(task)
    assert success is True
    mock_exchange.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_inference_task_publisher_disconnected_fallback() -> None:
    """Проверка поведения паблишера задач при отсутствии подключения."""
    pub = InferenceTaskPublisher(rabbitmq_url="amqp://guest:guest@localhost:5672/")
    pub._is_connected = False
    pub._last_connect_attempt = time.monotonic()  # блокируем reconnect

    task = InferenceTask(task_id="task-xyz", dialogues=[])
    success = await pub.publish_task(task)
    assert success is False


@pytest.mark.asyncio
async def test_audit_log_publisher_lifecycle() -> None:
    """Проверка публикации записей аудита через AuditLogPublisher."""
    pub = AuditLogPublisher(rabbitmq_url="amqp://guest:guest@localhost:5672/")
    mock_exchange = MagicMock()
    mock_exchange.publish = AsyncMock()

    mock_channel = MagicMock()
    mock_channel.default_exchange = mock_exchange
    pub._channel = mock_channel

    mock_conn = MagicMock()
    mock_conn.is_closed = False
    pub._connection = mock_conn
    pub._is_connected = True

    res = await pub.publish_event({"event": "user_predict", "status": "ok"})
    assert res is True
    mock_exchange.publish.assert_awaited_once()
