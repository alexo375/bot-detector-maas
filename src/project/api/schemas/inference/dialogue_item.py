from pydantic import BaseModel, Field

from project.api.schemas.inference.message import MessageItem


class DialogueItem(BaseModel):
    """Элемент диалоговой сессии с идентификатором и списком сообщений."""

    dialog_id: str = Field(..., description="Уникальный идентификатор сессии диалога")
    messages: list[MessageItem] = Field(..., min_length=1, description="Список сообщений сессии диалога")
