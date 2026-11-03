"""Общие Pydantic- и dataclass-схемы инференса и статуса асинхронных задач.

Используются совместно слоями API Gateway и фоновых воркеров без нарушения границ архитектуры.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class AuditMode(StrEnum):
    """Режим аудита событий инференса в ClickHouse DWH."""

    PRODUCTION = "production"
    TEST = "test"
    OFF = "off"


class ParticipantVerdict(BaseModel):
    """Схема классификационного вердикта по конкретному участнику диалога."""

    participant_index: int
    bot_probability: float
    is_bot: bool


class DialogueResultItem(BaseModel):
    """Результат оценки отдельной диалоговой сессии."""

    dialog_id: str
    dialogue_length: int
    verdicts: list[ParticipantVerdict]
    cached: bool
    latency_ms: float

    @classmethod
    def from_inference(
        cls,
        dialog_id: str,
        dialogue_length: int,
        verdicts: list[dict[str, Any]] | list[ParticipantVerdict],
        cached: bool,
        latency_ms: float,
    ) -> "DialogueResultItem":
        """Фабричный метод создания элемента ответа из результатов работы модели."""
        return cls(
            dialog_id=dialog_id,
            dialogue_length=dialogue_length,
            verdicts=[v if isinstance(v, ParticipantVerdict) else ParticipantVerdict(**v) for v in verdicts],
            cached=cached,
            latency_ms=round(latency_ms, 3),
        )


class PredictResponse(BaseModel):
    """Модель ответа пакетной классификации диалогов."""

    results: list[DialogueResultItem]
    total: int
    batch_latency_ms: float


class AsyncTaskStatus(StrEnum):
    """Статусы жизненного цикла асинхронной задачи."""

    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class AsyncTaskRecord:
    """Датакласс состояния фоновой задачи в хранилище кэша."""

    status: AsyncTaskStatus
    result: PredictResponse | None = None

    def to_dict(self) -> dict[str, Any]:
        """Преобразование записи задачи в словарь для сериализации."""
        return {
            "status": str(self.status),
            "result": self.result.model_dump() if self.result is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AsyncTaskRecord":
        """Создание записи задачи из словаря кэша с явной валидацией Enum."""
        raw_result = data.get("result")
        result = PredictResponse(**raw_result) if raw_result is not None else None
        return cls(status=AsyncTaskStatus(data.get("status", AsyncTaskStatus.UNKNOWN)), result=result)


__all__ = [
    "AsyncTaskRecord",
    "AsyncTaskStatus",
    "AuditMode",
    "DialogueResultItem",
    "ParticipantVerdict",
    "PredictResponse",
]
