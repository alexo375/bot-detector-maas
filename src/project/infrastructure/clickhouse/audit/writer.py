import logging
from typing import Any

from project.core.schemas import AuditMode
from project.infrastructure.rabbitmq import AuditLogPublisher
from project.infrastructure.rabbitmq.schemas.event import DialogueEvent

logger = logging.getLogger(__name__)


class ClickHouseAuditWriter:
    """Сервис формирования и асинхронной публикации аудиторских логов диалогов в ClickHouse DWH через RabbitMQ."""

    def __init__(self, publisher: AuditLogPublisher | None = None) -> None:
        """Инициализация сервиса с клиентом очереди RabbitMQ."""
        self.publisher = publisher or AuditLogPublisher()

    def create_event(
        self,
        dialog_id: str,
        messages: list[dict[str, Any]],
        verdicts: list[Any],
        features: list[list[float]] | None = None,
        base_model_probs: dict[str, list[float]] | None = None,
        latency_ms: float = 0.0,
        cached: bool = False,
        audit_mode: AuditMode = AuditMode.PRODUCTION,
    ) -> DialogueEvent:
        """Формирование типизированного события DialogueEvent для последующей вставки в chat_logs ClickHouse."""
        verdicts_list = [v.model_dump() if hasattr(v, "model_dump") else dict(v) for v in verdicts]
        eval_result = {
            "verdicts": verdicts_list,
            "features": features if features is not None else [],
            "base_model_probs": base_model_probs if base_model_probs is not None else {},
        }
        return DialogueEvent.create(
            dialog_id=dialog_id,
            messages=messages,
            eval_result=eval_result,
            cached=cached,
            latency_ms=latency_ms,
            audit_mode=audit_mode,
        )

    async def publish_event(self, event: DialogueEvent | dict[str, Any]) -> bool:
        """Асинхронная публикация события в очередь аудита ClickHouse DWH."""
        return await self.publisher.publish_event(event)
