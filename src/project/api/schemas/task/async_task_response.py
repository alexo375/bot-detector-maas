from pydantic import BaseModel

from project.core.schemas import AsyncTaskStatus


class AsyncTaskResponse(BaseModel):
    """Схема ответа API при постановке задачи в асинхронную обработку."""

    task_id: str
    dialog_id: str
    status: AsyncTaskStatus
    poll_url: str
    webhook_url: str | None = None
    message: str = "Task accepted for processing"
