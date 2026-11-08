from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from project.features.schema_base import BaseFeatureSchema

if TYPE_CHECKING:
    from project.features.store.context import DialogueContext


class BaseFeatureView(ABC):
    """Абстрактный интерфейс группы признаков для Feature Store."""

    schema: ClassVar[type | None] = None

    @property
    def name(self) -> str:
        """Имя группы признаков, вычисляемое из схемы."""
        if self.schema is not None:
            name = self.schema.__name__.lower()
            return name[:-8] if name.endswith("features") else name
        return self.__class__.__name__.lower()

    @property
    def features(self) -> list[str]:
        """Список признаков, вычисляемый динамически из полей схемы."""
        return self.get_feature_names()

    @classmethod
    def get_feature_names(cls) -> list[str]:
        """Получение списка имен признаков схемы без создания экземпляра класса."""
        if cls.schema is not None and hasattr(cls.schema, "__dataclass_fields__"):
            return list(cls.schema.__dataclass_fields__.keys())
        return []

    @abstractmethod
    def extract_participant_features(
        self,
        messages: list[dict[str, Any]],
        p: int,
        context: DialogueContext,
    ) -> dict[str, float]:
        """Вычисление признаков группы для участника диалога."""


__all__ = ["BaseFeatureSchema", "BaseFeatureView"]
