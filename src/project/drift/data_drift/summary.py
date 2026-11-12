from dataclasses import dataclass
from typing import Any


@dataclass
class DataDriftSummary:
    """Результаты статистической проверки дрифта признаков на базе критерия Колмогорова-Смирнова."""

    dataset_drift: bool
    drift_share: float
    number_of_drifted_columns: int
    number_of_columns: int
    drift_threshold: float
    retrain_recommended: bool
    drifted_features: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Сериализация результатов мониторинга в словарь."""
        return {
            "dataset_drift": self.dataset_drift,
            "drift_share": self.drift_share,
            "number_of_drifted_columns": self.number_of_drifted_columns,
            "number_of_columns": self.number_of_columns,
            "drift_threshold": self.drift_threshold,
            "retrain_recommended": self.retrain_recommended,
            "drifted_features": self.drifted_features,
        }
