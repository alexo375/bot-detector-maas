import logging
from typing import Any

from project.infrastructure.rabbitmq.publishers.base_publisher import BaseRabbitMQPublisher
from project.infrastructure.rabbitmq.publishers.constants import QUEUE_INFERENCE_TASKS

logger = logging.getLogger(__name__)


class InferenceTaskPublisher(BaseRabbitMQPublisher):
    """Специализированный паблишер задач инференса в очередь `dialogue_inference_tasks`."""

    async def _declare_queues(self) -> None:
        if self._channel is not None:
            await self._channel.declare_queue(QUEUE_INFERENCE_TASKS, durable=True)

    async def publish_task(self, task: Any) -> bool:
        """Асинхронная публикация задачи инференса в очередь для обработки воркером."""
        from project.infrastructure.rabbitmq.schemas.task import InferenceTask

        task_dict = task.to_dict() if isinstance(task, InferenceTask) else dict(task)
        success = await self._publish_raw_message(task_dict, routing_key=QUEUE_INFERENCE_TASKS)
        if success:
            logger.info("Published inference task %s to queue '%s'", task_dict.get("task_id"), QUEUE_INFERENCE_TASKS)
        else:
            logger.warning("RabbitMQ is unavailable, cannot queue inference task: %s", task_dict.get("task_id"))
        return success
