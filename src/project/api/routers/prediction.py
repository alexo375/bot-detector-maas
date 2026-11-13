import asyncio
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Response, status

from project.api.metrics import (
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    INFERENCE_LATENCY,
)
from project.api.schemas import (
    AsyncTaskRecord,
    AsyncTaskResponse,
    AsyncTaskStatus,
    DialogueResultItem,
    PredictRequest,
    PredictResponse,
)
from project.core.config import config
from project.core.logging import ProjectLogger
from project.core.metrics import PredictionMetricsRecorder
from project.core.utils import DialogueHasher
from project.infrastructure.container import ServiceContainer
from project.infrastructure.rabbitmq import InferenceTask

logger = ProjectLogger.get_logger(__name__)

router = APIRouter(tags=["Predictions"])


@router.post(
    "/predict",
    response_model=None,
    summary="Классификация диалогов",
    description="""
Принимает список диалогов и определяет вероятность того, что участники являются ботами.

* Если диалоги есть в кэше Redis, возвращается 200 OK с готовыми результатами.
* При промахе кэша задача ставится в очередь RabbitMQ и возвращается 202 Accepted c task_id для получения результата через GET /predict/{task_id}.
""",
    responses={
        200: {"model": PredictResponse, "description": "Результаты получены из кэша Redis"},
        202: {"model": AsyncTaskResponse, "description": "Задача поставлена в очередь на расчет"},
        503: {"description": "Очередь обработки задач временно недоступна"},
    },
)
async def predict_dialogues(
    request: PredictRequest,
    response: Response,
) -> Any:
    """Классификация диалогов: возврат из кэша (200) или постановка в очередь (202)."""
    t0 = time.perf_counter()
    services = ServiceContainer.get_instance()

    # 1. Быстрая пакетная проверка кэша Redis (без блокировки event loop)
    raw_msgs_list = [[m.model_dump() for m in d.messages] for d in request.dialogues]
    hashes = [DialogueHasher.compute(raw_msgs) for raw_msgs in raw_msgs_list]
    cached_entries = await asyncio.to_thread(services.cache.mget, hashes)

    all_cached = bool(cached_entries and all(e is not None for e in cached_entries))

    # 2. Если ВСЕ диалоги найдены в кэше — немедленный синхронный ответ 200 OK (O(1))
    if all_cached:
        elapsed_sec = time.perf_counter() - t0
        cache_latency_ms = round(elapsed_sec * 1000.0, 3)
        INFERENCE_LATENCY.observe(elapsed_sec)
        cached_results: list[DialogueResultItem] = []
        for i, d in enumerate(request.dialogues):
            cached_val = cached_entries[i]
            if cached_val is not None:
                CACHE_HITS_TOTAL.inc()
                verdicts_list = cached_val["verdicts"]
                PredictionMetricsRecorder.record(verdicts_list)
                cached_results.append(
                    DialogueResultItem.from_inference(
                        dialog_id=d.dialog_id,
                        dialogue_length=cached_val["dialogue_length"],
                        verdicts=verdicts_list,
                        cached=True,
                        latency_ms=cache_latency_ms,
                    )
                )
        return PredictResponse(
            results=cached_results,
            total=len(cached_results),
            batch_latency_ms=cache_latency_ms,
        )

    # 3. При промахе кэша проверяем Redis и пробуем переподключить RabbitMQ при обрыве
    if not services.cache.is_connected:
        logger.error("Fail-fast: Redis cache unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference cache (Redis) is unavailable. Please retry shortly.",
        )

    if not services.publisher.is_connected:
        await services.publisher.connect()

    if not services.publisher.is_connected:
        logger.error("Fail-fast: RabbitMQ task broker unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference backend (RabbitMQ) is unavailable. Please retry shortly.",
        )

    task_id = str(uuid.uuid4())
    task_record = AsyncTaskRecord(status=AsyncTaskStatus.PROCESSING, result=None)
    queued_ok = await asyncio.to_thread(services.cache.set_task, task_id, task_record.to_dict())
    if queued_ok is False:
        logger.error("Failed to store initial state for inference task %s in Redis", task_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to persist task state in Redis cache. Please retry shortly.",
        )

    # Формируем задачу для изолированного воркера
    task = InferenceTask(
        task_id=task_id,
        dialogues=[
            {"dialog_id": d.dialog_id, "messages": [m.model_dump() for m in d.messages]} for d in request.dialogues
        ],
        webhook_url=str(request.webhook_url) if request.webhook_url else None,
        audit_mode=request.audit_mode,
    )

    published = await services.publisher.publish_task(task)
    if not published:
        await asyncio.to_thread(services.cache.delete_task, task_id)
        logger.error("Failed to enqueue inference task %s to RabbitMQ", task_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to enqueue inference task. Please retry shortly.",
        )

    for entry in cached_entries:
        if entry is not None:
            CACHE_HITS_TOTAL.inc()
        else:
            CACHE_MISSES_TOTAL.inc()

    response.status_code = status.HTTP_202_ACCEPTED
    first_d_id = request.dialogues[0].dialog_id if request.dialogues else "batch"
    return AsyncTaskResponse(
        task_id=task_id,
        dialog_id=first_d_id,
        status=AsyncTaskStatus.PROCESSING,
        poll_url=f"/predict/{task_id}",
        webhook_url=str(request.webhook_url) if request.webhook_url else None,
        message="Batch queued for asynchronous processing in worker",
    )


@router.get(
    "/predict/{task_id}",
    summary="Получение результатов расчета по ID задачи",
    description="Опрос статуса и получение готового вердикта по задаче инференса.",
    responses={
        200: {"model": PredictResponse, "description": "Расчет завершен, результат успешно получен"},
        202: {"description": "Задача выполняется воркером"},
        404: {"description": "Задача не найдена или истек срок хранения в кэше"},
        503: {"description": "Хранилище состояний задач (Redis) временно недоступно"},
    },
)
def get_prediction_result(task_id: str) -> PredictResponse:
    """Получение готового результата классификации диалогов по task_id."""
    services = ServiceContainer.get_instance()
    raw_task = services.cache.get_task(task_id)
    if raw_task is None:
        if not services.cache.is_connected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference cache (Redis) is unavailable. Please retry shortly.",
            )
        ttl_minutes = max(1, config.redis.task_default_ttl_sec // 60)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found or expired from cache (TTL {ttl_minutes}m). Please resubmit dialogue for prediction.",
        )

    try:
        task_record = AsyncTaskRecord.from_dict(raw_task)
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("Corrupted async task payload for %s: %s", task_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored inference task state is corrupted.",
        ) from exc

    if task_record.status == AsyncTaskStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference task execution failed during processing in worker.",
        )
    if task_record.status == AsyncTaskStatus.PROCESSING or task_record.result is None:
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail="Task is still processing. Please retry shortly.",
        )

    return task_record.result
