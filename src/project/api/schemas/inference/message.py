from pydantic import BaseModel, Field


class MessageItem(BaseModel):
    """Схема отдельного сообщения диалога."""

    participant_index: int = Field(..., ge=0, le=1, description="Индекс участника диалога (0 или 1)")
    text: str = Field(..., description="Текст реплики участника")
    timestamp: str | None = Field(None, description="ISO-временная метка сообщения")
