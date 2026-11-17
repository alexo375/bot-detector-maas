"""Интеграционные тесты отказоустойчивости и обработки сбоев InferenceWorker."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from project.api.routers.prediction import get_prediction_result
from project.api.schemas import AsyncTaskStatus
from project.core.schemas import AuditMode
from project.infrastructure.container import ServiceContainer
from project.infrastructure.rabbitmq.schemas.task import InferenceTask
from project.infrastructure.workers.inference_worker import InferenceWorker
from tests.fixtures.mock_fixtures import FakeDialogueCache


@pytest.mark.asyncio
async def test_worker_triton_failure_marks_task_failed_and_api_returns_500():
    """Сбой инференса Triton переводит задачу в FAILED, а API при опросе возвращает HTTP 500."""
    cache = FakeDialogueCache()
    mock_triton = MagicMock()
    mock_triton.infer.side_effect = RuntimeError("Triton connection timeout: model meta_stacker unavailable")

    worker = InferenceWorker(
        triton_client=mock_triton,
        cache=cache,  # type: ignore[arg-type]
    )

    task_id = "failed-task-uuid-123"
    task = InferenceTask(
        task_id=task_id,
        dialogues=[
            {
                "dialog_id": "dlg_fail",
                "messages": [{"participant_index": 0, "text": "тест"}],
            }
        ],
        audit_mode=AuditMode.OFF,
    )

    mock_msg = MagicMock()
    mock_msg.body = json.dumps(task.to_dict()).encode("utf-8")
    mock_msg.process.return_value.__aenter__ = AsyncMock()
    mock_msg.process.return_value.__aexit__ = AsyncMock()

    # Worker перехватывает сбой инференса, не падает и помечает задачу как FAILED
    await worker.handle_message(mock_msg)

    cached_task = cache.get_task(task_id)
    assert cached_task is not None
    assert cached_task["status"] == AsyncTaskStatus.FAILED

    # Проверяем, что API при опросе этой задачи отдает 500
    container = ServiceContainer.get_instance()
    container.cache = cache  # type: ignore[assignment]

    with pytest.raises(HTTPException) as exc_info:
        get_prediction_result(task_id)

    assert exc_info.value.status_code == 500
    assert "Inference task execution failed" in exc_info.value.detail


@pytest.mark.asyncio
async def test_worker_webhook_failure_does_not_abort_task():
    """Сбой отправки Webhook не прерывает выполнение задачи и сохранение результата в кэш."""
    import numpy as np

    cache = FakeDialogueCache()
    mock_triton = MagicMock()

    def mock_infer(features: np.ndarray) -> tuple[np.ndarray, ...]:
        n = features.shape[0]
        p = np.full(n, 0.2, dtype=np.float32)
        return p, p, p, p

    mock_triton.infer.side_effect = mock_infer

    worker = InferenceWorker(
        triton_client=mock_triton,
        cache=cache,  # type: ignore[arg-type]
    )

    task_id = "task-webhook-fail-456"
    task = InferenceTask(
        task_id=task_id,
        dialogues=[
            {
                "dialog_id": "dlg_webhook",
                "messages": [
                    {"participant_index": 0, "text": "Привет"},
                    {"participant_index": 1, "text": "Привет"},
                ],
            }
        ],
        webhook_url="https://non-existent-webhook.invalid/endpoint",
        audit_mode=AuditMode.OFF,
    )

    mock_msg = MagicMock()
    mock_msg.body = json.dumps(task.to_dict()).encode("utf-8")
    mock_msg.process.return_value.__aenter__ = AsyncMock()
    mock_msg.process.return_value.__aexit__ = AsyncMock()

    # Имитируем сбой httpx при отправке вебхука
    with patch("project.infrastructure.workers.service.WebhookNotifier.send", new_callable=AsyncMock) as mock_send:
        mock_send.side_effect = Exception("Connection refused by webhook target")

        # Задача успешно завершается, несмотря на сбой отправки вебхука
        await worker.handle_message(mock_msg)

    task_in_cache = cache.get_task(task_id)
    assert task_in_cache is not None
    assert task_in_cache["status"] == AsyncTaskStatus.COMPLETED
    assert task_in_cache["result"] is not None


@pytest.mark.asyncio
async def test_worker_corrupted_message_payload():
    """Невалидный JSON в теле сообщения очереди обрабатывается без падения процесса."""
    cache = FakeDialogueCache()
    worker = InferenceWorker(cache=cache)  # type: ignore[arg-type]

    corrupted_msg = MagicMock()
    corrupted_msg.body = b"NOT_A_VALID_JSON_{{["
    corrupted_msg.process.return_value.__aenter__ = AsyncMock()
    corrupted_msg.process.return_value.__aexit__ = AsyncMock()

    # Воркер ловит ошибку парсинга и не падает
    await worker.handle_message(corrupted_msg)

    # В кэше не должно появиться мусорных записей
    assert len(cache.tasks) == 0
