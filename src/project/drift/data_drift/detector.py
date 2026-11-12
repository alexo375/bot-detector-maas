import math

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import ks_2samp

from project.core.logging import ProjectLogger
from project.drift.data_drift.prometheus_exporter import DataDriftPrometheusExporter
from project.drift.data_drift.summary import DataDriftSummary

logger = ProjectLogger.get_logger(__name__)


class DataDriftDetector:
    """Детектор дрифта признаков на базе двухвыборочного критерия Колмогорова-Смирнова."""

    def __init__(
        self,
        metrics_exporter: DataDriftPrometheusExporter | None = None,
    ) -> None:
        self.metrics_exporter = metrics_exporter or DataDriftPrometheusExporter()

    @staticmethod
    def _extract_series_array(df: pl.DataFrame | pd.DataFrame, col: str) -> np.ndarray:
        if isinstance(df, pl.DataFrame):
            arr = df[col].cast(pl.Float64, strict=False).drop_nulls().drop_nans().to_numpy()
            return np.asarray(arr[np.isfinite(arr)], dtype=np.float64)
        arr_pd = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        return np.asarray(arr_pd[np.isfinite(arr_pd)], dtype=np.float64)

    def detect_drift(
        self,
        reference_df: pl.DataFrame | pd.DataFrame,
        current_df: pl.DataFrame | pd.DataFrame,
        drift_share_threshold: float = 0.25,
        p_value_threshold: float = 0.05,
    ) -> DataDriftSummary:
        """Статистический анализ дрифта признаков через двухвыборочный KS-тест с экспортом в Prometheus."""
        logger.info(
            "Running statistical data drift check (Ref: %d rows, Cur: %d rows)",
            len(reference_df),
            len(current_df),
        )

        candidate_cols = [c for c in reference_df.columns if c in current_df.columns and c != "dialog_id"]
        if not candidate_cols:
            logger.warning("No overlapping feature columns found between reference and current datasets.")
            empty_summary = DataDriftSummary(
                dataset_drift=False,
                drift_share=0.0,
                number_of_drifted_columns=0,
                number_of_columns=0,
                drift_threshold=drift_share_threshold,
                retrain_recommended=False,
                drifted_features=[],
            )
            self.metrics_exporter.export(empty_summary, column_p_values={})
            return empty_summary

        drifted_features: list[str] = []
        column_p_values: dict[str, float] = {}

        for col in candidate_cols:
            ref_arr = self._extract_series_array(reference_df, col)
            cur_arr = self._extract_series_array(current_df, col)
            if len(ref_arr) < 2 or len(cur_arr) < 2:
                continue

            _stat, p_val = ks_2samp(ref_arr, cur_arr)
            p_val_float = float(p_val)
            if not math.isfinite(p_val_float):
                continue

            if p_val_float < p_value_threshold:
                drifted_features.append(col)
            column_p_values[col] = p_val_float

        number_of_columns = len(column_p_values)
        number_of_drifted_columns = len(drifted_features)
        drift_share = float(number_of_drifted_columns / number_of_columns) if number_of_columns > 0 else 0.0
        dataset_drift = bool(number_of_columns > 0 and drift_share >= drift_share_threshold)

        summary = DataDriftSummary(
            dataset_drift=dataset_drift,
            drift_share=drift_share,
            number_of_drifted_columns=number_of_drifted_columns,
            number_of_columns=number_of_columns,
            drift_threshold=drift_share_threshold,
            retrain_recommended=dataset_drift,
            drifted_features=drifted_features,
        )
        self.metrics_exporter.export(summary, column_p_values=column_p_values)
        return summary
