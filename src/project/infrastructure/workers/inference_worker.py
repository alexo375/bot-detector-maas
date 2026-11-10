"""Асинхронный воркер обработки очереди задач инференса диалогов из RabbitMQ."""

import argparse
import asyncio
import contextlib
import json
import signal

import aio_pika
from prometheus_client import start_http_server

from project.core.config import config
from project.core.logging import LoggingConfigurator, ProjectLogger
from project.core.metrics import INFERENCE_LATENCY, PredictionMetricsRecorder
from project.core.schemas import AsyncTaskRecord, AsyncTaskStatus, PredictResponse
from project.features.store import FeatureStore
from project.infrastructure.clickhouse import ClickHouseAuditWriter
from project.infrastructure.rabbitmq import QUEUE_INFERENCE_TASKS, AuditLogPublisher, DialogueEvent
from project.infrastructure.rabbitmq.schemas.task import InferenceTask
from project.infrastructure.redis import DialogueCache
from project.infrastructure.triton.client import TritonEnsembleClient
from project.infrastructure.workers.service import (
    InferencePipelineService,
    WebhookNotifier,
)
from project.models.postprocessors.pairwise import PairwisePostprocessor

logger = ProjectLogger.get_logger(__name__)


class InferenceWorker:
    """Изолированный воркер обработки очереди задач инференса диалогов.

    Архитектура:
    1. Забирает задачи из очереди RabbitMQ (`dialogue_inference_tasks`).
    2. Делегирует извлечение фичей, инференс и калибровку в `InferencePipelineService`.
    3. Сохраняет готовые результаты в Redis и шлет события аудита в RabbitMQ для ClickHouse DWH.
    """

    def __init__(
        self,
        rabbitmq_url: str | None = None,
        triton_client: TritonEnsembleClient | None = None,
        feature_store: FeatureStore | None = None,
        postprocessor: PairwisePostprocessor | None = None,
        cache: DialogueCache | None = None,
        publisher: AuditLogPublisher | None = None,
        audit_writer: ClickHouseAuditWriter | None = None,
        prefetch_count: int = 5,
        pipeline_service: InferencePipelineService | None = None,
    ) -> None:
        self.rabbitmq_url = rabbitmq_url or config.rabbitmq.url
        self.prefetch_count = prefetch_count
        self.cache = cache or DialogueCache()
        self.audit_writer = audit_writer or ClickHouseAuditWriter(publisher=publisher)

        self.pipeline_service = pipeline_service or InferencePipelineService(
            triton_client=triton_client,
            feature_store=feature_store,
            postprocessor=postprocessor,
            cache=self.cache,
            audit_writer=self.audit_writer,
        )

        self.connection: aio_pika.abc.AbstractRobustConnection | None = None
        self.channel: aio_pika.abc.AbstractChannel | None = None
        self.queue: aio_pika.abc.AbstractQueue | None = None
        self._running = True

    async def connect_rabbitmq(self) -> bool:
        """Установка надежного подключения к RabbitMQ."""
        try:
            self.connection = await aio_pika.connect_robust(self.rabbitmq_url, timeout=5.0)
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=self.prefetch_count)
            self.queue = await self.channel.declare_queue(QUEUE_INFERENCE_TASKS, durable=True)
            logger.info(
                "InferenceWorker connected to RabbitMQ (queue: '%s', prefetch: %d)",
                QUEUE_INFERENCE_TASKS,
                self.prefetch_count,
            )
            return True
        except Exception as e:  # noqa: BLE001  # Resilient fallback
            logger.error("InferenceWorker failed to connect to RabbitMQ: %s", e)
            return False

    def process_task_sync(self, task: InferenceTask) -> tuple[PredictResponse, list[DialogueEvent]]:
        """Прямой синхронный конвейер обработки задачи инференса."""
        return self.pipeline_service.process_task(task)

    async def handle_message(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        """Асинхронная обработка входящего сообщения с задачей инференса."""
        task_id: str | None = None
        async with message.process(requeue=False):
            try:
                task_raw = json.loads(message.body.decode("utf-8"))
                task = InferenceTask.from_dict(task_raw)
                task_id = task.task_id
                logger.info("Processing inference task %s (%d dialogues)", task.task_id, len(task.dialogues))

                loop = asyncio.get_running_loop()
                eval_response, events = await loop.run_in_executor(None, self.process_task_sync, task)

                INFERENCE_LATENCY.observe(eval_response.batch_latency_ms / 1000.0)
                for res_item in eval_response.results:
                    PredictionMetricsRecorder.record(res_item.verdicts)

                task_record = AsyncTaskRecord(status=AsyncTaskStatus.COMPLETED, result=eval_response)
                saved = await asyncio.to_thread(self.cache.set_task, task.task_id, task_record.to_dict())
                if saved is False:
                    logger.warning("Failed to persist COMPLETED status for task %s to Redis", task.task_id)

                if task.webhook_url:
                    try:
                        await WebhookNotifier.send(str(task.webhook_url), eval_response.model_dump())
                    except Exception as wh_err:  # noqa: BLE001
                        logger.warning("Failed to deliver webhook for task %s: %s", task.task_id, wh_err)

                for event in events:
                    try:
                        published = await self.audit_writer.publish_event(event)
                        if not published:
                            logger.warning(
                                "Audit event for dialogue %s in task %s was not published to RabbitMQ",
                                event.dialog_id,
                                task.task_id,
                            )
                    except Exception as pub_err:  # noqa: BLE001
                        logger.warning("Failed to publish audit event for task %s: %s", task.task_id, pub_err)

                logger.info(
                    "Successfully completed inference task %s in %.2f ms",
                    task.task_id,
                    eval_response.batch_latency_ms,
                )
            except Exception:
                logger.exception("Error processing inference task %s", task_id)
                if task_id:
                    try:
                        failed_record = AsyncTaskRecord(status=AsyncTaskStatus.FAILED, result=None)
                        await asyncio.to_thread(self.cache.set_task, task_id, failed_record.to_dict())
                    except Exception as cache_err:  # noqa: BLE001
                        logger.error("Failed to mark task %s as FAILED in cache: %s", task_id, cache_err)

    async def start(self) -> None:
        """Основной рабочий цикл консумера."""
        semaphore = asyncio.Semaphore(max(1, self.prefetch_count))
        active_tasks: set[asyncio.Task[None]] = set()

        async def _run_with_limit(msg: aio_pika.abc.AbstractIncomingMessage) -> None:
            try:
                await self.handle_message(msg)
            finally:
                semaphore.release()

        while self._running:
            if not self.connection or self.connection.is_closed:
                connected = await self.connect_rabbitmq()
                if not connected:
                    await asyncio.sleep(3.0)
                    continue

            try:
                if self.queue is None:
                    raise RuntimeError("RabbitMQ queue is not initialized")
                logger.info("InferenceWorker started consuming from '%s'...", QUEUE_INFERENCE_TASKS)
                async with self.queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        if not bool(getattr(self, "_running", True)):
                            break
                        await semaphore.acquire()
                        task = asyncio.create_task(_run_with_limit(message))
                        active_tasks.add(task)
                        task.add_done_callback(active_tasks.discard)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.warning("Connection lost in InferenceWorker (%s), reconnecting...", e)
                await asyncio.sleep(2.0)

        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Остановка воркера и освобождение сетевых соединений."""
        self._running = False
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
        if self.audit_writer.publisher is not None:
            with contextlib.suppress(Exception):
                await self.audit_writer.publisher.close()
        with contextlib.suppress(Exception):
            self.pipeline_service.triton_client.close()
        logger.info("InferenceWorker stopped.")


def main(argv: list[str] | None = None) -> None:
    """Точка входа изолированного воркера инференса диалогов."""
    parser = argparse.ArgumentParser(description="Async Dialogue Inference Worker")
    parser.add_argument(
        "--prefetch",
        type=int,
        default=config.rabbitmq.prefetch_count,
        help="QoS prefetch count для RabbitMQ",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=8003,
        help="Порт Prometheus HTTP-экспортера метрик воркера",
    )
    args = parser.parse_args(argv)

    LoggingConfigurator.configure()
    start_http_server(args.metrics_port)
    logger.info("Worker Prometheus metrics server started on :%d/metrics", args.metrics_port)
    worker = InferenceWorker(prefetch_count=args.prefetch)

    async def _run() -> None:
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()

        def _request_stop() -> None:
            worker._running = False
            if task is not None:
                task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _request_stop)

        try:
            await worker.start()
        finally:
            await worker.stop()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
