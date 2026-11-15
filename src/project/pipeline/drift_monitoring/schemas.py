"""Типы и схемы данных подсистемы мониторинга дрифта."""

from dataclasses import dataclass
from typing import Any, TypedDict

import pandas as pd
import polars as pl


@dataclass
class FeatureDistributionData:
    """Контейнер датафреймов эталонного и текущего распределения признаков."""

    ref_df: pl.DataFrame | pd.DataFrame | None = None
    cur_df: pl.DataFrame | pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "ref_df": self.ref_df,
            "cur_df": self.cur_df,
        }


class DriftMonitoringResult(TypedDict):
    """Структура унифицированного отчёта мониторинга дрифта."""

    drift_share: float
    dataset_drift: bool
    prediction_drift_psi: float
    prediction_entropy: float
    ensemble_disagreement_rate: float
    concept_drift_detected: bool
    retrain_recommended: bool
    retrain_status: int
    decision_reason: str
    production_dialogues_count: int
    model_age_days: float
    drifted_features: list[str]
    number_of_drifted_columns: int
    number_of_columns: int
