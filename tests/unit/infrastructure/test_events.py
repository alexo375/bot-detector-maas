"""Модульные тесты генерации событий диалогов и публикации в RabbitMQ."""

import pytest

from project.infrastructure.rabbitmq import AuditLogPublisher, DialogueEvent


def test_build_dialogue_event() -> None:
    """Проверка структуры и валидности формируемого события диалога."""
    messages = [
        {"participant_index": 0, "text": "Привет! Ты бот?"},
        {"participant_index": 1, "text": "Нет, человек."},
    ]
    eval_result = {
        "dialogue_length": 2,
        "confidence": "low_data_uncertainty",
        "verdicts": [
            {"participant_index": 0, "bot_probability": 0.25, "is_bot": False},
            {"participant_index": 1, "bot_probability": 0.85, "is_bot": True},
        ],
        "features": [[0.1] * 45, [0.9] * 45],
    }

    event = DialogueEvent.create(
        dialog_id="test-session-123",
        messages=messages,
        eval_result=eval_result,
        cached=False,
        latency_ms=12.5,
    )

    assert event.dialog_id == "test-session-123"
    assert event.num_messages == 2
    assert event.prob_p0 == 0.25
    assert event.prob_p1 == 0.85
    assert event.is_bot_p0 == 0
    assert event.is_bot_p1 == 1
    assert event.verdict == "p1_bot"
    assert event.confidence == "low_data_uncertainty"
    assert event.cached == 0
    assert event.latency_ms == 12.5
    assert len(event.features_p0) == 45
    assert len(event.features_p1) == 45
    assert event.messages_json


@pytest.mark.asyncio
async def test_audit_log_publisher_graceful_fallback() -> None:
    """Проверка бессбойной работы паблишера при отсутствии брокера RabbitMQ."""
    publisher = AuditLogPublisher(rabbitmq_url="amqp://non_existent_host:5672/")
    event_data = {"dialog_id": "fallback-test", "num_messages": 1}

    res = await publisher.publish_event(event_data)
    assert res is False
    assert not publisher.is_connected


@pytest.mark.asyncio
async def test_clickhouse_audit_writer_create_and_publish() -> None:
    """Проверка формирования и отправки аудиторских событий через ClickHouseAuditWriter."""
    from unittest.mock import AsyncMock

    from project.infrastructure.clickhouse import ClickHouseAuditWriter

    mock_publisher = AsyncMock()
    writer = ClickHouseAuditWriter(publisher=mock_publisher)

    event = writer.create_event(
        dialog_id="ch-test-1",
        messages=[{"participant_index": 0, "text": "test"}],
        verdicts=[
            {"participant_index": 0, "bot_probability": 0.1, "is_bot": False},
            {"participant_index": 1, "bot_probability": 0.9, "is_bot": True},
        ],
        features=[[0.5] * 10, [0.8] * 10],
        latency_ms=5.0,
        cached=False,
    )

    assert event.dialog_id == "ch-test-1"
    assert event.prob_p0 == 0.1
    assert event.prob_p1 == 0.9
    assert event.verdict == "p1_bot"
    assert len(event.features_p0) == 10
    assert len(event.features_p1) == 10

    await writer.publish_event(event)
    assert mock_publisher.publish_event.called


@pytest.mark.asyncio
async def test_specialized_publishers_queue_routing() -> None:
    """Проверка очередей назначения специализированных паблишеров."""
    from project.infrastructure.rabbitmq import (
        QUEUE_AUDIT_EVENTS,
        QUEUE_INFERENCE_TASKS,
        AuditLogPublisher,
        InferenceTaskPublisher,
    )

    task_pub = InferenceTaskPublisher(rabbitmq_url="amqp://non_existent:5672/")
    audit_pub = AuditLogPublisher(rabbitmq_url="amqp://non_existent:5672/")

    assert QUEUE_INFERENCE_TASKS == "dialogue_inference_tasks"
    assert QUEUE_AUDIT_EVENTS == "dialog_events"
    assert not task_pub.is_connected
    assert not audit_pub.is_connected


def test_base_rabbitmq_publisher_is_abstract() -> None:
    """Проверка, что BaseRabbitMQPublisher является абстрактным базовым классом (ABC)."""
    from project.infrastructure.rabbitmq import BaseRabbitMQPublisher

    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseRabbitMQPublisher"):
        BaseRabbitMQPublisher()  # type: ignore[abstract]


def test_audit_mode_serialization_and_off_skips_worker_events() -> None:
    """Проверка сериализации AuditMode и отключения публикации событий аудита при audit_mode='off'."""
    from unittest.mock import MagicMock

    import numpy as np

    from project.core.schemas import AuditMode
    from project.infrastructure.rabbitmq.schemas.task import InferenceTask
    from project.infrastructure.workers.service import InferencePipelineService
    from tests.fixtures.mock_fixtures import FakeDialogueCache

    task = InferenceTask(
        task_id="t-off",
        dialogues=[
            {
                "dialog_id": "dlg-off",
                "messages": [{"participant_index": 0, "text": "hi"}, {"participant_index": 1, "text": "yo"}],
            }
        ],
        audit_mode=AuditMode.OFF,
    )
    restored = InferenceTask.from_dict(task.to_dict())
    assert restored.audit_mode == AuditMode.OFF

    ev = DialogueEvent.create(
        dialog_id="d-test",
        messages=[{"participant_index": 0, "text": "a"}, {"participant_index": 1, "text": "b"}],
        eval_result={
            "verdicts": [
                {"participant_index": 0, "bot_probability": 0.2, "is_bot": False},
                {"participant_index": 1, "bot_probability": 0.8, "is_bot": True},
            ]
        },
        cached=False,
        latency_ms=1.0,
        audit_mode=AuditMode.TEST,
    )
    assert DialogueEvent.from_dict(ev.to_dict()).audit_mode == AuditMode.TEST

    mock_triton = MagicMock()
    mock_triton.infer.return_value = (
        np.array([0.1, 0.9], dtype=np.float32),
        np.array([0.1, 0.9], dtype=np.float32),
        np.array([0.1, 0.9], dtype=np.float32),
        np.array([0.1, 0.9], dtype=np.float32),
    )
    mock_fs = MagicMock()
    mock_fs.transform.return_value = np.zeros((2, 95), dtype=np.float32)

    svc = InferencePipelineService(
        triton_client=mock_triton,
        feature_store=mock_fs,
        cache=FakeDialogueCache(),  # type: ignore[arg-type]
    )
    _, events = svc.process_task(task)
    assert events == []
