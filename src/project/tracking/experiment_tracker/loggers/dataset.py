import hashlib

import mlflow
import numpy as np
import pandas as pd
import polars as pl
from mlflow.data.pandas_dataset import from_pandas


class DatasetLogger:
    """Логер датасетов в MLflow с фиксацией схемы признаков и SHA-256 дайджеста."""

    @staticmethod
    def log(
        data: pl.DataFrame | pd.DataFrame | np.ndarray,
        name: str,
        targets: np.ndarray | pl.Series | pd.Series | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """Логирование датасета со строгой схемой колонок и отображением в MLflow UI."""
        if isinstance(data, np.ndarray):
            df_pd = pd.DataFrame(data, columns=feature_names) if feature_names is not None else pd.DataFrame(data)
        elif isinstance(data, pl.DataFrame):
            df_pd = data.to_pandas()
        else:
            df_pd = data.copy()

        if feature_names is not None and not isinstance(data, np.ndarray):
            cols = [c for c in df_pd.columns if c != "target"]
            if len(cols) == len(feature_names):
                df_pd = df_pd.rename(columns=dict(zip(cols, feature_names, strict=False)))

        target_arr = targets.to_numpy() if (targets is not None and hasattr(targets, "to_numpy")) else targets
        if target_arr is not None:
            df_pd["target"] = target_arr

        digest = hashlib.sha256(df_pd.to_json().encode()).hexdigest()[:10]
        mlflow.log_input(
            from_pandas(df_pd, name=name, targets="target" if target_arr is not None else None, digest=digest),
            context="training",
        )


__all__ = ["DatasetLogger"]
