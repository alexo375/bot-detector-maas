"""Строитель для стандартизированной сборки и валидации запуска эксперимента."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from project.core.config import config
from project.tracking.experiment_tracker.builder_validator import BuilderValidator
from project.tracking.experiment_tracker.schemas import ExperimentEnvironment
from project.tracking.experiment_tracker.tracker import ExperimentTracker


class ExperimentTrackerBuilder:
    """Строитель для стандартизированной сборки и валидации запуска эксперимента."""

    def __init__(
        self,
        run_name: str,
        experiment_name: str = "bot-detection",
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
        description: str | None = None,
        require_graphics: bool = True,
        require_feature_tree: bool = True,
        require_description: bool = True,
        require_submission: bool = True,
        require_metrics: bool = True,
        environment: ExperimentEnvironment = ExperimentEnvironment.OFFLINE,
    ):
        self._run_name = run_name
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri or config.mlflow_tracking_uri
        self._tags = tags or {}
        self._description = description
        self._require_graphics = require_graphics
        self._require_feature_tree = require_feature_tree
        self._require_description = require_description
        self._require_submission = require_submission
        self._require_metrics = require_metrics
        self._environment = (
            environment if isinstance(environment, ExperimentEnvironment) else ExperimentEnvironment(str(environment))
        )
        self._require_dataset: bool = True
        self._dataset: Any = None
        self._params: dict[str, Any] | None = None
        self._metrics: dict[str, Any] | None = None

    def set_environment(self, environment: ExperimentEnvironment | str) -> "ExperimentTrackerBuilder":
        """Устанавливает среду запуска (OFFLINE / ONLINE)."""
        self._environment = (
            environment if isinstance(environment, ExperimentEnvironment) else ExperimentEnvironment(str(environment))
        )
        return self

    def require_metrics(self, required: bool = True) -> "ExperimentTrackerBuilder":
        """Включает или отключает обязательную проверку метрик."""
        self._require_metrics = required
        return self

    def defer_metrics(self) -> "ExperimentTrackerBuilder":
        """Откладывает передачу метрик до вызова `tracker.log_metrics()` внутри контекста."""
        self._require_metrics = False
        return self

    def set_description(self, description: str) -> "ExperimentTrackerBuilder":
        """Устанавливает описание эксперимента."""
        self._description = description
        return self

    def require_description(self, required: bool = True) -> "ExperimentTrackerBuilder":
        """Включает или отключает требование описания эксперимента."""
        self._require_description = required
        return self

    def require_graphics(self, required: bool = True) -> "ExperimentTrackerBuilder":
        """Включает или отключает требование графиков."""
        self._require_graphics = required
        return self

    def require_feature_tree(self, required: bool = True) -> "ExperimentTrackerBuilder":
        """Включает или отключает требование `feature_tree.json`."""
        self._require_feature_tree = required
        return self

    def require_submission(self, required: bool = True) -> "ExperimentTrackerBuilder":
        """Включает или отключает требование `submission.csv`."""
        self._require_submission = required
        return self

    def defer_dataset(self) -> "ExperimentTrackerBuilder":
        """Откладывает логирование датасета до входа в контекст."""
        self._require_dataset = False
        return self

    def require_dataset(self, required: bool = True) -> "ExperimentTrackerBuilder":
        """Включает или отключает обязательное логирование датасета."""
        self._require_dataset = required
        return self

    def set_dataset(
        self,
        data: pl.DataFrame | pd.DataFrame | np.ndarray,
        name: str,
        targets: np.ndarray | pl.Series | pd.Series | None = None,
        feature_names: list[str] | None = None,
    ) -> "ExperimentTrackerBuilder":
        """Задает датасет для фиксации в MLflow input."""
        if isinstance(data, np.ndarray):
            warnings.warn(
                f"В set_dataset передан raw numpy.ndarray для '{name}'. Используйте DataFrame для сохранения схемы.",
                UserWarning,
                stacklevel=2,
            )
        self._dataset = (data, name, targets, feature_names)
        return self

    def set_params(
        self,
        model_family: str | None = None,
        params: dict[str, Any] | None = None,
        n_features: int | None = None,
        feature_names: list[str] | None = None,
        **extra_params: Any,
    ) -> "ExperimentTrackerBuilder":
        """Задает гиперпараметры и семейство модели."""
        self._params = {
            "model_family": model_family,
            "params": params,
            "n_features": n_features,
            "feature_names": feature_names,
            "extra_params": extra_params,
        }
        return self

    def set_metrics(
        self,
        val_logloss: float | None = None,
        val_roc_auc: float | None = None,
        val_accuracy: float | None = None,
        val_f1: float | None = None,
        val_logloss_raw: float | None = None,
        fold_scores: list[float] | None = None,
        extra_metrics: dict[str, float] | None = None,
        **custom_metrics: float,
    ) -> "ExperimentTrackerBuilder":
        """Задает канонический набор метрик качества классификатора."""
        self._metrics = {
            "val_logloss": val_logloss,
            "val_logloss_raw": val_logloss_raw,
            "val_roc_auc": val_roc_auc,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "fold_scores": fold_scores,
            "extra_metrics": extra_metrics,
            "custom_metrics": custom_metrics,
        }
        return self

    def build(self) -> ExperimentTracker:
        """Валидирует инварианты и собирает `ExperimentTracker`."""
        BuilderValidator.validate(
            run_name=self._run_name,
            require_description=self._require_description,
            description=self._description,
            require_dataset=self._require_dataset,
            dataset=self._dataset,
            params=self._params,
            require_metrics=self._require_metrics,
            metrics=self._metrics,
        )

        tracker = ExperimentTracker(
            run_name=self._run_name,
            experiment_name=self._experiment_name,
            tracking_uri=self._tracking_uri,
            tags=self._tags,
            description=self._description,
            require_graphics=self._require_graphics,
            require_feature_tree=self._require_feature_tree,
            require_description=self._require_description,
            require_submission=self._require_submission,
            require_dataset=self._require_dataset,
            environment=self._environment,
        )
        if self._dataset is not None:
            data, name, targets = self._dataset[0], self._dataset[1], self._dataset[2]
            fn = (
                self._dataset[3]
                if len(self._dataset) > 3
                else (self._params.get("feature_names") if self._params else None)
            )
            tracker.with_dataset(data=data, name=name, targets=targets, feature_names=fn)

        if self._params is not None:
            p = self._params
            tracker.with_params(
                model_family=p["model_family"],
                params=p["params"],
                n_features=p["n_features"],
                feature_names=p["feature_names"],
                **p["extra_params"],
            )

        if self._metrics is not None:
            m = self._metrics
            tracker.with_metrics(
                val_logloss=m["val_logloss"],
                val_logloss_raw=m["val_logloss_raw"],
                val_roc_auc=m["val_roc_auc"],
                val_accuracy=m["val_accuracy"],
                val_f1=m["val_f1"],
                fold_scores=m["fold_scores"],
                extra_metrics=m["extra_metrics"],
                **m["custom_metrics"],
            )
        return tracker


__all__ = ["ExperimentTrackerBuilder"]
