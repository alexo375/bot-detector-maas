from dataclasses import dataclass
from typing import Any

from project.drift.feature_importance_diff.schemas.record import FeatureImportanceRecord
from project.drift.feature_importance_diff.schemas.shift_status import FeatureShiftStatus


@dataclass
class FeatureImportanceDiffReport:
    """Датакласс отчета смещения важности признаков без логики рендеринга."""

    top_n: int
    features: list[FeatureImportanceRecord]
    dropped_features: list[dict[str, Any]]
    summary: dict[str, int]

    @classmethod
    def create(
        cls,
        top_n: int,
        features: list[FeatureImportanceRecord],
        dropped_features: list[dict[str, Any]],
    ) -> "FeatureImportanceDiffReport":
        """Фабричный метод создания отчета с подсчетом агрегированной сводки."""
        summary = {
            "surging_count": sum(1 for r in features if r.status == FeatureShiftStatus.SURGING),
            "dropping_count": sum(1 for r in features if r.status == FeatureShiftStatus.DROPPING),
            "stable_count": sum(1 for r in features if r.status == FeatureShiftStatus.STABLE),
            "new_count": sum(1 for r in features if r.status == FeatureShiftStatus.NEW),
            "dropped_count": len(dropped_features),
        }
        return cls(
            top_n=top_n,
            features=features,
            dropped_features=dropped_features,
            summary=summary,
        )

    def to_dict(self) -> dict[str, Any]:
        """Преобразование отчета в сериализуемый словарь."""
        return {
            "top_n": self.top_n,
            "features": [r.to_dict() for r in self.features],
            "dropped_features": self.dropped_features,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureImportanceDiffReport":
        """Создание отчета из словаря."""
        features = [FeatureImportanceRecord.from_dict(f) for f in data.get("features", [])]
        return cls(
            top_n=int(data["top_n"]),
            features=features,
            dropped_features=list(data.get("dropped_features", [])),
            summary=dict(data.get("summary", {})),
        )
