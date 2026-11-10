"""Модульные тесты для InferenceTask и изолированного InferenceWorker (RabbitMQ -> Triton)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from project.infrastructure.rabbitmq.schemas.task import InferenceTask
from project.infrastructure.workers.inference_worker import InferenceWorker
from tests.fixtures.mock_fixtures import FakeDialogueCache as FakeCache


def test_inference_task_serialization():
    """Проверка сериализации и десериализации InferenceTask."""
    task = InferenceTask(
        task_id="task-123",
        dialogues=[
            {
                "dialog_id": "d1",
                "messages": [{"participant_index": 0, "text": "Привет"}],
            }
        ],
        webhook_url="https://example.com/webhook",
    )
    json_str = json.dumps(task.to_dict())
    reloaded = InferenceTask.from_dict(json.loads(json_str))

    assert reloaded.task_id == "task-123"
    assert len(reloaded.dialogues) == 1
    assert reloaded.dialogues[0]["dialog_id"] == "d1"
    assert reloaded.webhook_url == "https://example.com/webhook"


def test_inference_worker_process_task_sync():
    """Проверка логики process_task_sync изолированного воркера."""
    import numpy as np

    fake_cache = FakeCache()
    mock_triton = MagicMock()

    def mock_infer(features: np.ndarray) -> tuple[np.ndarray, ...]:
        n = features.shape[0]
        p = np.array([0.15 if i % 2 == 0 else 0.85 for i in range(n)], dtype=np.float32)
        return p, p, p, p

    mock_triton.infer.side_effect = mock_infer

    worker = InferenceWorker(
        triton_client=mock_triton,
        cache=fake_cache,  # type: ignore[arg-type]
    )

    task = InferenceTask(
        task_id="test-task-1",
        dialogues=[
            {
                "dialog_id": "test-dialog-1",
                "messages": [
                    {"participant_index": 0, "text": "Ты бот?"},
                    {"participant_index": 1, "text": "Да, я бот."},
                ],
            }
        ],
    )

    response, events = worker.process_task_sync(task)

    assert response.total == 1
    assert len(response.results) == 1
    assert response.results[0].dialog_id == "test-dialog-1"
    assert response.results[0].cached is False
    assert len(response.results[0].verdicts) == 2
    assert response.results[0].verdicts[0].bot_probability == 0.15
    assert response.results[0].verdicts[1].is_bot is True
    assert len(events) == 1
    assert set(events[0].base_model_probs.keys()) == {"catboost", "lightgbm", "xgboost"}
    assert mock_triton.infer.called
    called_features = mock_triton.infer.call_args[0][0]
    assert isinstance(called_features, np.ndarray)
    assert called_features.shape[0] == 2


@pytest.mark.asyncio
async def test_inference_worker_handle_message():
    """Проверка асинхронной обработки сообщения очереди в handle_message."""
    import numpy as np

    fake_cache = FakeCache()
    mock_triton = MagicMock()
    mock_publisher = AsyncMock()

    def mock_infer(features: np.ndarray) -> tuple[np.ndarray, ...]:
        n = features.shape[0]
        p = np.array([0.05 if i % 2 == 0 else 0.95 for i in range(n)], dtype=np.float32)
        return p, p, p, p

    mock_triton.infer.side_effect = mock_infer

    worker = InferenceWorker(
        triton_client=mock_triton,
        cache=fake_cache,  # type: ignore[arg-type]
        publisher=mock_publisher,
    )

    task = InferenceTask(
        task_id="async-task-999",
        dialogues=[
            {
                "dialog_id": "d-async",
                "messages": [
                    {"participant_index": 0, "text": "Проверка"},
                    {"participant_index": 1, "text": "Тест"},
                ],
            }
        ],
    )

    # Имитируем aio_pika IncomingMessage
    mock_msg = MagicMock()
    mock_msg.body = json.dumps(task.to_dict()).encode("utf-8")
    mock_msg.process.return_value.__aenter__ = AsyncMock()
    mock_msg.process.return_value.__aexit__ = AsyncMock()

    await worker.handle_message(mock_msg)

    # Проверяем, что результат сохранен в Redis кэш задач
    task_in_cache = fake_cache.get_task("async-task-999")
    assert task_in_cache["status"] == "completed"
    assert mock_publisher.publish_event.called
    assert mock_triton.infer.called


def test_inference_worker_pure_triton_infer():
    """Проверка чистого потока: FeatureStore воркера -> triton.infer(features) -> postprocessing."""
    import numpy as np

    fake_cache = FakeCache()
    mock_pure_triton = MagicMock()

    def mock_infer(features: np.ndarray) -> tuple[np.ndarray, ...]:
        n = features.shape[0]
        p = np.full(n, 0.15, dtype=np.float32)
        return p, p, p, p

    mock_pure_triton.infer.side_effect = mock_infer

    worker = InferenceWorker(
        triton_client=mock_pure_triton,
        cache=fake_cache,  # type: ignore[arg-type]
    )

    task = InferenceTask(
        task_id="pure-infer-1",
        dialogues=[
            {
                "dialog_id": "pure-dialog-1",
                "messages": [
                    {"participant_index": 0, "text": "Привет! Как дела?"},
                    {"participant_index": 1, "text": "Привет! Отлично, работаю."},
                ],
            }
        ],
    )

    response, events = worker.process_task_sync(task)
    assert response.total == 1
    assert mock_pure_triton.infer.called
    # Проверяем, что в infer передана numpy матрица признаков
    called_features = mock_pure_triton.infer.call_args[0][0]
    assert isinstance(called_features, np.ndarray)
    assert called_features.shape[0] == 2  # 2 участника диалога
    assert len(events) == 1


@pytest.mark.asyncio
async def test_inference_worker_stop() -> None:
    """Проверка корректной остановки воркера и закрытия соединения."""
    mock_conn = MagicMock()
    mock_conn.is_closed = False
    mock_conn.close = AsyncMock()

    worker = InferenceWorker()
    worker.connection = mock_conn
    worker._running = True

    await worker.stop()
    assert worker._running is False
    mock_conn.close.assert_awaited_once()


def test_inference_pipeline_partial_cache_merging() -> None:
    """Проверка оптимизации инференса: Triton вызывается ТОЛЬКО для промахов кэша."""
    import numpy as np

    from project.core.utils import DialogueHasher
    from project.infrastructure.workers.service import InferencePipelineService

    fake_cache = FakeCache()
    mock_triton = MagicMock()
    mock_triton.infer.return_value = (
        np.array([0.8, 0.2]),
        np.array([0.7, 0.3]),
        np.array([0.75, 0.25]),
        np.array([0.85, 0.15]),
    )

    d1_msgs = [{"participant_index": 0, "text": "Привет от бота"}]
    h1 = DialogueHasher.compute(d1_msgs)
    fake_cache.set(
        h1,
        {
            "dialogue_length": 1,
            "verdicts": [
                {"participant_index": 0, "bot_probability": 0.99, "is_bot": True},
                {"participant_index": 1, "bot_probability": 0.01, "is_bot": False},
            ],
        },
    )

    d2_msgs = [{"participant_index": 0, "text": "Новый диалог"}]
    service = InferencePipelineService(
        triton_client=mock_triton,
        cache=fake_cache,  # type: ignore[arg-type]
    )

    task = InferenceTask(
        task_id="partial-test-1",
        dialogues=[
            {"dialog_id": "dlg_cached", "messages": d1_msgs},
            {"dialog_id": "dlg_new", "messages": d2_msgs},
        ],
    )

    response, _events = service.process_task(task)
    assert response.total == 2
    assert response.results[0].dialog_id == "dlg_cached"
    assert response.results[0].cached is True
    assert response.results[0].verdicts[0].bot_probability == 0.99

    assert response.results[1].dialog_id == "dlg_new"
    assert response.results[1].cached is False

    called_features = mock_triton.infer.call_args[0][0]
    assert len(called_features) == 2
