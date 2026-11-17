"""Интеграционные тесты аудиторского логирования: ClickHouseAuditWriter -> RabbitMQ -> Consumer -> ClickHouse."""

import uuid
from collections.abc import Generator
from unittest.mock import AsyncMock

import pytest

from project.core.schemas import AuditMode
from project.infrastructure.clickhouse import ClickHouseConnectionFactory, FeatureDWH
from project.infrastructure.clickhouse.audit import ChatLogRecord, ClickHouseAuditWriter
from project.infrastructure.clickhouse.audit.consumer import ClickHouseLogConsumer
from project.infrastructure.rabbitmq.schemas.dialogue_verdict import DialogueVerdict

pytestmark = [pytest.mark.live, pytest.mark.xdist_group("clickhouse_test_db")]

TEST_DB = "bot_detection_test"


@pytest.fixture(scope="module")
def ch_client() -> Generator[object]:
    """Изолированная тестовая БД ClickHouse для интеграционного тестирования аудита."""
    admin = ClickHouseConnectionFactory.create()
    admin.command(f"CREATE DATABASE IF NOT EXISTS {TEST_DB}")
    client = ClickHouseConnectionFactory.create(database=TEST_DB)
    FeatureDWH(client).init_schema()
    try:
        yield client
    finally:
        client.close()
        admin.command(f"DROP DATABASE IF EXISTS {TEST_DB}")
        admin.close()


def test_clickhouse_audit_writer_event_creation():
    """Проверка формирования типизированного DialogueEvent через ClickHouseAuditWriter."""
    mock_publisher = AsyncMock()
    writer = ClickHouseAuditWriter(publisher=mock_publisher)

    test_dialog_id = f"test_audit_event_{uuid.uuid4().hex[:8]}"
    messages = [
        {"participant_index": 0, "text": "Привет, бот!"},
        {"participant_index": 1, "text": "Я не бот, я человек."},
    ]
    verdicts = [
        {"participant_index": 0, "bot_probability": 0.12, "is_bot": False},
        {"participant_index": 1, "bot_probability": 0.88, "is_bot": True},
    ]
    features = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    event = writer.create_event(
        dialog_id=test_dialog_id,
        messages=messages,
        verdicts=verdicts,
        features=features,
        latency_ms=12.5,
        cached=False,
    )

    assert event.dialog_id == test_dialog_id
    assert event.num_messages == 2
    assert event.prob_p0 == 0.12
    assert event.prob_p1 == 0.88
    assert event.is_bot_p0 == 0
    assert event.is_bot_p1 == 1
    assert event.verdict == DialogueVerdict.P1_BOT
    assert event.cached == 0
    assert event.latency_ms == 12.5


def test_clickhouse_consumer_row_mapping_and_insertion(ch_client):
    """Интеграционный тест: запись события аудита через ChatLogRecord в реальную таблицу chat_logs."""
    unique_dialog_id = f"test_ch_audit_{uuid.uuid4().hex[:10]}"

    payload = {
        "dialog_id": unique_dialog_id,
        "timestamp": "2026-11-11T12:00:00+00:00",
        "num_messages": 3,
        "messages_json": '[{"participant_index": 0, "text": "hello"}]',
        "features_p0": [1.0, 2.0, 3.0],
        "features_p1": [4.0, 5.0, 6.0],
        "prob_p0": 0.15,
        "prob_p1": 0.85,
        "is_bot_p0": 0,
        "is_bot_p1": 1,
        "verdict": "p1_bot",
        "confidence": "high",
        "cached": 0,
        "latency_ms": 15.4,
    }

    # 1. Проверяем преобразование payload в ChatLogRecord
    record = ChatLogRecord.from_payload(payload)
    row = record.to_row()
    assert len(row) == len(ChatLogRecord.column_names())
    assert row[0] == unique_dialog_id

    # 2. Вставка строки через клиент ClickHouse
    ch_client.insert(
        table="chat_logs",
        data=[row],
        column_names=ChatLogRecord.column_names(),
    )

    # 3. Вычитка и проверка данных из ClickHouse
    query_res = ch_client.query(
        f"SELECT dialog_id, num_messages, prob_p0, prob_p1, is_bot_p0, is_bot_p1, verdict, latency_ms "
        f"FROM chat_logs WHERE dialog_id = '{unique_dialog_id}'"
    )
    rows = query_res.result_rows
    assert len(rows) == 1
    ch_row = rows[0]
    assert ch_row[0] == unique_dialog_id
    assert ch_row[1] == 3
    assert abs(ch_row[2] - 0.15) < 1e-4
    assert abs(ch_row[3] - 0.85) < 1e-4
    assert ch_row[4] == 0
    assert ch_row[5] == 1
    assert ch_row[6] == "p1_bot"
    assert abs(ch_row[7] - 15.4) < 1e-4


@pytest.mark.asyncio
async def test_consumer_flush_buffer_lifecycle(ch_client):
    """Интеграционный тест жизненного цикла буфера ClickHouseLogConsumer: накопление и сброс."""
    unique_dialog_id = f"test_consumer_flush_{uuid.uuid4().hex[:10]}"

    consumer = ClickHouseLogConsumer(batch_size=5, flush_interval_sec=0.1)
    consumer.ch_client = ch_client

    payload = {
        "dialog_id": unique_dialog_id,
        "timestamp": "2026-11-11T12:00:00+00:00",
        "num_messages": 2,
        "messages_json": "[]",
        "features_p0": [],
        "features_p1": [],
        "prob_p0": 0.05,
        "prob_p1": 0.95,
        "is_bot_p0": 0,
        "is_bot_p1": 1,
        "verdict": "p1_bot",
        "confidence": "high",
        "cached": 0,
        "latency_ms": 10.0,
    }

    mock_msg = AsyncMock()
    row = consumer.parse_event_to_row(payload)
    consumer._message_buffer.append((mock_msg, row, AuditMode.PRODUCTION))

    # Сбрасываем буфер
    inserted = await consumer.flush_buffer()
    assert inserted == 1
    assert len(consumer._message_buffer) == 0
    mock_msg.ack.assert_awaited_once()

    # Проверяем наличие в изолированной тестовой БД
    query_res = ch_client.query(f"SELECT count() FROM chat_logs WHERE dialog_id = '{unique_dialog_id}'")
    assert query_res.result_rows[0][0] == 1
