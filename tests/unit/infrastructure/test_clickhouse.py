"""Модульные тесты воркера микробатчинга и сохранения событий в ClickHouse."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project.core.schemas import AuditMode
from project.infrastructure.clickhouse.audit import ChatLogRecord
from project.infrastructure.clickhouse.audit.consumer import ClickHouseLogConsumer


def test_clickhouse_consumer_parse_event() -> None:
    """Проверка корректной трансформации JSON события в строку таблицы ClickHouse."""
    consumer = ClickHouseLogConsumer()
    payload = {
        "dialog_id": "test-parse-001",
        "timestamp": "2026-11-03T18:00:00+00:00",
        "num_messages": 3,
        "messages_json": '[{"p":0,"t":"hello"}]',
        "features_p0": [1.0] * 45,
        "features_p1": [2.0] * 45,
        "prob_p0": 0.15,
        "prob_p1": 0.80,
        "base_model_probs": {"catboost": [0.1, 0.9]},
        "is_bot_p0": 0,
        "is_bot_p1": 1,
        "verdict": "p1_bot",
        "confidence": "high",
        "cached": 1,
        "latency_ms": 3.4,
    }

    row = consumer.parse_event_to_row(payload)
    assert len(row) == len(ChatLogRecord.column_names())
    assert row[0] == "test-parse-001"
    assert row[2] == 3
    assert row[6] == 0.15
    assert row[7] == 0.80
    assert row[8] == {"catboost": [0.1, 0.9]}
    assert row[11] == "p1_bot"
    assert row[13] == 1
    assert len(row[4]) == 45
    assert len(row[5]) == 45


@pytest.mark.asyncio
async def test_clickhouse_consumer_batch_flush_success() -> None:
    """Проверка батчевой записи в ClickHouse и отправки ack сообщениям."""
    consumer = ClickHouseLogConsumer(batch_size=2)
    mock_ch_client = MagicMock()
    consumer.ch_client = mock_ch_client

    msg1 = AsyncMock()
    msg2 = AsyncMock()

    row1 = ["id1", "ts", 2, "[]", [0.0] * 45, [0.0] * 45, 0.1, 0.2, {}, 0, 0, "both_humans", "high", 0, 1.0]
    row2 = ["id2", "ts", 4, "[]", [0.0] * 45, [0.0] * 45, 0.8, 0.1, {}, 1, 0, "p0_bot", "high", 0, 2.0]

    consumer._message_buffer.append((msg1, row1, AuditMode.PRODUCTION))
    consumer._message_buffer.append((msg2, row2, AuditMode.PRODUCTION))

    inserted_count = await consumer.flush_buffer()

    assert inserted_count == 2
    mock_ch_client.insert.assert_called_once_with(
        table="chat_logs",
        data=[row1, row2],
        column_names=ChatLogRecord.column_names(),
    )
    msg1.ack.assert_awaited_once()
    msg2.ack.assert_awaited_once()
    assert len(consumer._message_buffer) == 0


@pytest.mark.asyncio
async def test_clickhouse_consumer_batch_retry_on_failure() -> None:
    """Проверка requeue при сбое записи в ClickHouse (at-least-once)."""
    consumer = ClickHouseLogConsumer()
    mock_ch_client = MagicMock()
    mock_ch_client.insert.side_effect = RuntimeError("ClickHouse connection reset")
    consumer.ch_client = mock_ch_client

    msg1 = AsyncMock()
    row1 = ["id1", "ts", 2, "[]", [0.0] * 45, [0.0] * 45, 0.1, 0.2, {}, 0, 0, "both_humans", "high", 0, 1.0]

    consumer._message_buffer.append((msg1, row1, AuditMode.PRODUCTION))

    inserted_count = await consumer.flush_buffer()

    assert inserted_count == 0
    msg1.ack.assert_not_awaited()
    msg1.nack.assert_awaited_once_with(requeue=True)


def test_clickhouse_consumer_connect_failure() -> None:
    """Проверка обработки недоступности ClickHouse при инициализации соединения."""
    consumer = ClickHouseLogConsumer(ch_host="unreachable.clickhouse.host")
    with patch("clickhouse_connect.get_client", side_effect=Exception("Connection refused")):
        success = consumer.connect_clickhouse()
        assert success is False
        assert consumer.ch_client is None


@pytest.mark.asyncio
async def test_clickhouse_consumer_close() -> None:
    """Проверка корректного закрытия соединений потребителя логов."""
    consumer = ClickHouseLogConsumer()
    mock_conn = MagicMock()
    mock_conn.is_closed = False
    mock_conn.close = AsyncMock()
    consumer.connection = mock_conn

    await consumer.close()
    assert consumer._running is False
    mock_conn.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_clickhouse_consumer_run_once_no_rabbitmq() -> None:
    """Проверка завершения run(run_once=True) при недоступности RabbitMQ."""
    consumer = ClickHouseLogConsumer(rabbitmq_url="amqp://nonexistent:5672/")
    with patch.object(consumer, "connect_rabbitmq", return_value=False):
        processed = await consumer.run(run_once=True)
        assert processed == 0


@pytest.mark.asyncio
async def test_clickhouse_consumer_routes_test_audit_mode_to_test_database() -> None:
    """События с audit_mode='test' пишутся строго в изолированную БД bot_detection_test."""
    consumer = ClickHouseLogConsumer(ch_database="bot_detection")
    mock_ch_client = MagicMock()
    consumer.ch_client = mock_ch_client

    prod_msg = AsyncMock()
    test_msg = AsyncMock()
    row_prod = ["p1", "ts", 2, "[]", [0.0] * 45, [0.0] * 45, 0.1, 0.2, 0, 0, "both_humans", "high", 0, 1.0]
    row_test = ["t1", "ts", 2, "[]", [0.0] * 45, [0.0] * 45, 0.8, 0.1, 1, 0, "p0_bot", "high", 0, 2.0]

    consumer._message_buffer.append((prod_msg, row_prod, AuditMode.PRODUCTION))
    consumer._message_buffer.append((test_msg, row_test, AuditMode.TEST))

    inserted = await consumer.flush_buffer()

    assert inserted == 2
    assert mock_ch_client.insert.call_count == 2
    first_call, second_call = mock_ch_client.insert.call_args_list
    assert first_call.kwargs == {
        "table": "chat_logs",
        "data": [row_prod],
        "column_names": ChatLogRecord.column_names(),
    }
    assert second_call.kwargs == {
        "table": "chat_logs",
        "data": [row_test],
        "column_names": ChatLogRecord.column_names(),
        "database": "bot_detection_test",
    }
    prod_msg.ack.assert_awaited_once()
    test_msg.ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_clickhouse_consumer_acks_and_skips_audit_mode_off() -> None:
    """Сообщения с audit_mode='off' не пишутся в ClickHouse, но подтверждаются (ack)."""
    consumer = ClickHouseLogConsumer(ch_database="bot_detection")
    mock_ch_client = MagicMock()
    consumer.ch_client = mock_ch_client

    off_msg = AsyncMock()
    row_off = ["off1", "ts", 2, "[]", [0.0] * 45, [0.0] * 45, 0.2, 0.3, {}, 0, 0, "both_humans", "high", 0, 1.0]
    consumer._message_buffer.append((off_msg, row_off, AuditMode.OFF))

    inserted = await consumer.flush_buffer()
    assert inserted == 0
    mock_ch_client.insert.assert_not_called()
    off_msg.ack.assert_awaited_once()
