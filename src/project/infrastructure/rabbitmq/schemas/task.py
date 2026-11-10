import time
from dataclasses import dataclass, field
from typing import Any

from project.core.schemas import AuditMode


@dataclass
class InferenceTask:
    """Структура задачи асинхронного инференса батча диалогов."""

    task_id: str
    dialogues: list[dict[str, Any]]
    webhook_url: str | None = None
    audit_mode: AuditMode = AuditMode.PRODUCTION
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Преобразование задачи в словарь для сериализации."""
        return {
            "task_id": self.task_id,
            "dialogues": self.dialogues,
            "webhook_url": self.webhook_url,
            "audit_mode": str(self.audit_mode),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceTask":
        """Создание задачи из словаря."""
        return cls(
            task_id=str(data["task_id"]),
            dialogues=list(data.get("dialogues", [])),
            webhook_url=data.get("webhook_url"),
            audit_mode=AuditMode(data.get("audit_mode", AuditMode.PRODUCTION)),
            created_at=float(data.get("created_at", time.time())),
        )
