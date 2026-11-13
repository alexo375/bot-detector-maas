from pydantic import BaseModel, Field

from project.api.schemas.inference.dialogue_item import DialogueItem
from project.core.schemas import AuditMode


class PredictRequest(BaseModel):
    """Модель пакетного запроса на классификацию сессий диалогов."""

    dialogues: list[DialogueItem] = Field(
        ...,
        min_length=1,
        description="Список сессий диалогов для батчевой классификации",
    )
    webhook_url: str | None = Field(
        None,
        description="URL для Push-уведомления через Webhook о готовности результатов инференса",
    )
    audit_mode: AuditMode = Field(
        default=AuditMode.PRODUCTION,
        description="Режим аудита событий в ClickHouse: production, test (изолированная БД) или off",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dialogues": [
                        {
                            "dialog_id": "session-customer-support-001",
                            "messages": [
                                {
                                    "participant_index": 0,
                                    "text": "Здравствуйте, как оформить возврат товара?",
                                },
                                {
                                    "participant_index": 1,
                                    "text": "Добрый день! Для оформления возврата перейдите в личный кабинет, раздел Заказы.",
                                },
                                {
                                    "participant_index": 0,
                                    "text": "Спасибо, а электронный чек подойдет?",
                                },
                                {
                                    "participant_index": 1,
                                    "text": "Да, конечно, электронный чек имеет полную юридическую силу.",
                                },
                            ],
                        }
                    ],
                    "webhook_url": "https://callback.example.com/api/v1/webhook",
                }
            ]
        }
    }


__all__ = ["PredictRequest"]
