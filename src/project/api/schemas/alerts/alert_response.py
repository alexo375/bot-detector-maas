from pydantic import BaseModel, Field


class AlertWebhookResponse(BaseModel):
    """Схема ответа endpoint обработки вебхука от Alertmanager."""

    status: str = Field("ok", description="Статус обработки вебхука (ok или error)")
    alert: str = Field(..., description="Сформированный заголовок алерта для ntfy")
    dispatched: bool = Field(True, description="Флаг успешной отправки уведомления в ntfy")
    error: str | None = Field(default=None, description="Текст ошибки при наличии")
