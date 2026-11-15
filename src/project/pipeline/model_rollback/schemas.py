"""Схемы данных и перечисления для подсистемы отката моделей."""

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, Field


class RollbackMode(StrEnum):
    """Режим выбора версии модели при откате."""

    INTERACTIVE_PICK = "interactive_pick"
    STEPS_BACK = "steps_back"
    EXACT_VERSION = "exact_version"


@dataclass(frozen=True)
class ModelVersionCandidate:
    """Кандидат версии модели из MLflow Model Registry."""

    version: str
    run_id: str
    created_ts: float
    val_logloss: float | None = None
    aliases: list[str] = field(default_factory=list)
    label: str = ""


class RollbackResult(BaseModel):
    """Строгая схема результата выполнения операции отката моделей."""

    status: str = Field(description="Статус завершения отката (completed / skipped / failed)")
    previous_version: str | None = Field(default=None, description="Прежняя версия champion в MLflow")
    rolled_back_to_version: str = Field(description="Версия модели, ставшая новым champion")
    target_run_id: str = Field(description="MLflow run_id целевой версии")
    triton_reload: dict[str, bool] = Field(default_factory=dict, description="Результаты Hot Reload в Triton")
    triton_healthy: bool = Field(description="Статус доступности ансамбля в Triton")
    cache_invalidated_keys: int = Field(default=0, description="Количество сброшенных ключей кэша предсказаний")
    drift_window_reset: bool = Field(description="Флаг сброса окна и baseline дрифта")
    reason: str = Field(description="Причина выполнения отката")
