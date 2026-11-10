from typing import Any

from project.infrastructure.rabbitmq.publishers.base_publisher import BaseRabbitMQPublisher
from project.infrastructure.rabbitmq.publishers.constants import QUEUE_AUDIT_EVENTS
from project.infrastructure.rabbitmq.schemas.event import DialogueEvent


class AuditLogPublisher(BaseRabbitMQPublisher):
    """Специализированный паблишер аналитических событий диалогов в очередь `dialog_events` для ClickHouse."""

    async def _declare_queues(self) -> None:
        if self._channel is not None:
            await self._channel.declare_queue(QUEUE_AUDIT_EVENTS, durable=True)

    async def publish_event(self, event: DialogueEvent | dict[str, Any]) -> bool:
        """Асинхронная публикация события диалога в очередь аудита ClickHouse DWH."""
        event_dict = event.to_dict() if isinstance(event, DialogueEvent) else event
        return await self._publish_raw_message(event_dict, routing_key=QUEUE_AUDIT_EVENTS)
