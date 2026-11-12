"""Схемы данных для мониторинга сдвига концепта (Concept Drift)."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ConceptDriftReport:
    """Отчет о сдвиге концепта в ансамбле моделей."""

    disagreement_rate: float
    feature_rank_correlation: float
    concept_drift_detected: bool

    def to_dict(self) -> dict[str, Any]:
        """Сериализация отчета в словарь."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConceptDriftReport":
        """Создание отчета из словаря."""
        return cls(
            disagreement_rate=float(data.get("disagreement_rate", 0.0)),
            feature_rank_correlation=float(data.get("feature_rank_correlation", 1.0)),
            concept_drift_detected=bool(data.get("concept_drift_detected", False)),
        )
