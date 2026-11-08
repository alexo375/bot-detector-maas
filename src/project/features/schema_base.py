from dataclasses import dataclass


@dataclass(frozen=True)
class BaseFeatureSchema:
    """Базовый датакласс схемы признаков с сериализацией в словарь."""

    def to_dict(self) -> dict[str, float]:
        """Преобразование полей датакласса в словарь значений float."""
        return {k: float(getattr(self, k)) for k in self.__dataclass_fields__}


__all__ = ["BaseFeatureSchema"]
