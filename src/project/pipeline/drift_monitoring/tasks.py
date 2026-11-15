"""Prefect-задачи для сбора данных из DWH и анализа дрифта признаков."""

from typing import Any

import pandas as pd
import polars as pl
from prefect import task

from project.core.logging import ProjectLogger
from project.drift.data_drift import (
    DataDriftDetector,
    DataDriftSummary,
    DWHFeatureFetcher,
)
from project.pipeline.drift_monitoring.schemas import FeatureDistributionData

logger = ProjectLogger.get_logger(__name__)


@task(name="fetch_data_from_dwh", retries=2, retry_delay_seconds=3)
def fetch_dwh_data_task(days: int = 7, since_ts: float | None = None) -> FeatureDistributionData:
    """Выгрузка текущих данных из ClickHouse DWH за окно и загрузка эталонного датасета."""
    logger.info("Fetching reference and production feature distributions...")
    fetcher = DWHFeatureFetcher()
    ref_df = fetcher.load_or_create_reference_dataset()
    cur_df = fetcher.fetch_current_features(days=days, since_ts=since_ts)
    return FeatureDistributionData(ref_df=ref_df, cur_df=cur_df)


@task(name="detect_feature_drift")
def detect_feature_drift_task(
    data: FeatureDistributionData | dict[str, Any],
    drift_threshold: float = 0.20,
) -> DataDriftSummary:
    """Статистический анализ дрифта признаков на базе критерия Колмогорова-Смирнова."""
    detector = DataDriftDetector()
    if isinstance(data, FeatureDistributionData):
        ref_df = data.ref_df
        cur_df = data.cur_df
    else:
        ref_df = data["ref_df"]
        cur_df = data["cur_df"]
    if not isinstance(ref_df, (pl.DataFrame, pd.DataFrame)) or not isinstance(cur_df, (pl.DataFrame, pd.DataFrame)):
        raise TypeError("Both reference_df and current_df must be Polars or Pandas DataFrames.")
    summary = detector.detect_drift(
        reference_df=ref_df,
        current_df=cur_df,
        drift_share_threshold=drift_threshold,
    )
    logger.info(
        "Drift check finished: dataset_drift=%s, drift_share=%.2f%%",
        summary.dataset_drift,
        summary.drift_share * 100,
    )
    return summary
