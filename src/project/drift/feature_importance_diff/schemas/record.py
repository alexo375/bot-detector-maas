from dataclasses import dataclass
from typing import Any

from project.drift.feature_importance_diff.schemas.shift_status import FeatureShiftStatus


@dataclass
class FeatureImportanceRecord:
    """Запись смещения отдельного признака в рейтинге важности."""

    feature: str
    current_rank: int
    previous_rank: int | None
    rank_shift: int
    current_importance: float
    previous_importance: float
    importance_delta: float
    status: FeatureShiftStatus

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureImportanceRecord":
        """Создание записи из словаря с явной валидацией Enum."""
        return cls(
            feature=str(data["feature"]),
            current_rank=int(data["current_rank"]),
            previous_rank=int(data["previous_rank"]) if data.get("previous_rank") is not None else None,
            rank_shift=int(data["rank_shift"]),
            current_importance=float(data["current_importance"]),
            previous_importance=float(data["previous_importance"]),
            importance_delta=float(data["importance_delta"]),
            status=FeatureShiftStatus(data["status"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Сериализация записи в словарь."""
        return {
            "feature": self.feature,
            "current_rank": self.current_rank,
            "previous_rank": self.previous_rank,
            "rank_shift": self.rank_shift,
            "current_importance": self.current_importance,
            "previous_importance": self.previous_importance,
            "importance_delta": self.importance_delta,
            "status": str(self.status),
        }
