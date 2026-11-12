"""Схемы данных для мониторинга Prediction Drift и неопределенности моделей."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PredictionDriftReport:
    """Отчет о сдвиге распределения вероятностей предсказаний (Prediction Drift)."""

    psi: float
    mean_entropy: float
    verdict_shares: dict[str, float]
    is_drifted: bool

    def to_dict(self) -> dict[str, Any]:
        """Сериализация отчета в словарь."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictionDriftReport":
        """Создание отчета из словаря."""
        return cls(
            psi=float(data.get("psi", 0.0)),
            mean_entropy=float(data.get("mean_entropy", 0.0)),
            verdict_shares=dict(data.get("verdict_shares", {})),
            is_drifted=bool(data.get("is_drifted", False)),
        )
