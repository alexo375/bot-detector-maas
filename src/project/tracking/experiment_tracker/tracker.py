"""Центральный фасад ExperimentTracker для отслеживания экспериментов в MLflow."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
import pandas as pd
import polars as pl

from project.core.config import config
from project.core.exceptions import TrackerError
from project.tracking.experiment_tracker.graphics_dispatcher import ExperimentGraphicsDispatcher
from project.tracking.experiment_tracker.lifecycle import RunLifecycleManager
from project.tracking.experiment_tracker.loggers.dataset import DatasetLogger
from project.tracking.experiment_tracker.loggers.optuna import OptunaMetricsLogger
from project.tracking.experiment_tracker.loggers.submissions import SubmissionLogger
from project.tracking.experiment_tracker.schemas import (
    ExperimentEnvironment,
    MetricOptimizationMode,
)

if TYPE_CHECKING:
    from project.tracking.experiment_tracker.builder import ExperimentTrackerBuilder


class ExperimentTracker:
    """Центральный фасад MLOps-трекера для воспроизводимости экспериментов."""

    def __init__(
        self,
        run_name: str,
        experiment_name: str = "bot-detection",
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
        description: str | None = None,
        nested: bool = False,
        require_graphics: bool = True,
        require_feature_tree: bool = True,
        require_description: bool = True,
        require_submission: bool = True,
        require_dataset: bool = True,
        artifact_location: str | None = None,
        environment: ExperimentEnvironment = ExperimentEnvironment.OFFLINE,
    ) -> None:
        self.run_name, self.experiment_name = run_name, experiment_name
        self.tracking_uri = tracking_uri or config.mlflow_tracking_uri
        self.tags, self.description, self.nested = tags or {}, description, nested
        self.require_graphics, self.require_feature_tree = require_graphics, require_feature_tree
        self.require_description, self.require_submission = require_description, require_submission
        self.require_dataset, self.artifact_location = require_dataset, artifact_location
        self.environment = (
            environment if isinstance(environment, ExperimentEnvironment) else ExperimentEnvironment(str(environment))
        )

        self.active_run: Any = None
        self._has_logged_dataset = self._has_logged_params = self._has_logged_metrics = False
        self._has_logged_graphics = self._has_logged_feature_tree = self._has_logged_submission = False
        self._logged_metrics: dict[str, float] = {}
        self._feature_names: list[str] | None = None
        self._pending_model_promotion: dict[str, Any] | None = None
        self._pending_dataset: Any = None
        self._pending_params: dict[str, Any] | None = None
        self._pending_metrics: dict[str, Any] | None = None
        self._start_time: float | None = None

    @classmethod
    def builder(cls, run_name: str, **kwargs: Any) -> "ExperimentTrackerBuilder":
        """Создает строитель `ExperimentTrackerBuilder` для пошаговой сборки рана."""
        from project.tracking.experiment_tracker.builder import ExperimentTrackerBuilder

        return ExperimentTrackerBuilder(run_name=run_name, **kwargs)

    @property
    def run_id(self) -> str:
        """Возвращает `run_id` активного MLflow рана."""
        if self.active_run:
            return str(self.active_run.info.run_id)
        raise TrackerError("Активный MLflow-ран не запущен. Обращайтесь к run_id внутри контекста 'with tracker:'.")

    def with_dataset(
        self,
        data: pl.DataFrame | pd.DataFrame | np.ndarray,
        name: str,
        targets: np.ndarray | pl.Series | pd.Series | None = None,
        feature_names: list[str] | None = None,
    ) -> "ExperimentTracker":
        """Прикрепляет обучающий или валидационный датасет к рану."""
        if self.active_run:
            self.log_dataset(data=data, name=name, targets=targets, feature_names=feature_names)
        else:
            self._pending_dataset = (data, name, targets, feature_names)
        return self

    def with_params(self, **kwargs: Any) -> "ExperimentTracker":
        """Прикрепляет параметры модели к рану."""
        if self.active_run:
            self.log_params(**kwargs)
        else:
            self._pending_params = kwargs
        return self

    def with_metrics(self, **kwargs: Any) -> "ExperimentTracker":
        """Прикрепляет метрики качества к рану."""
        if self.active_run:
            self.log_metrics(**kwargs)
        else:
            self._pending_metrics = kwargs
        return self

    def create_optuna_callback(self, metric_name: str = "val_logloss") -> Callable[[Any, Any], None]:
        """Создает Optuna callback для записи траектории оптимизации в MLflow."""
        return OptunaMetricsLogger.create_callback(metric_name=metric_name)

    def __enter__(self) -> "ExperimentTracker":
        return RunLifecycleManager.start_run(self)

    def __exit__(self, exc_type: Any, exc_val: Any, _exc_tb: Any) -> Any:
        return RunLifecycleManager.end_run(self, exc_type, exc_val)

    def log_tags(self, tags: dict[str, Any]) -> None:
        """Запись структурированных тегов в активный ран MLflow."""
        clean_tags = {str(k): str(v) for k, v in tags.items() if v is not None}
        if clean_tags:
            mlflow.set_tags(clean_tags)

    def log_dataset(
        self,
        data: pl.DataFrame | pd.DataFrame | np.ndarray,
        name: str,
        targets: np.ndarray | pl.Series | pd.Series | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """Логирование датасета со схемой колонок и хешем содержимого."""
        fn = (
            feature_names
            or self._feature_names
            or (self._pending_params.get("feature_names") if self._pending_params else None)
        )
        DatasetLogger.log(data=data, name=name, targets=targets, feature_names=fn)
        self._has_logged_dataset = True

    def log_params(
        self,
        model_family: str | None = None,
        params: dict[str, Any] | None = None,
        feature_names: list[str] | None = None,
        **extra_params: Any,
    ) -> None:
        """Логирование гиперпараметров модели без служебных шумовых ключей."""
        if feature_names is not None:
            self._feature_names = feature_names
        clean = RunLifecycleManager.sanitize_params(model_family, feature_names, params, extra_params)
        mlflow.log_params(clean)
        self._has_logged_params = True

    def log_metrics(
        self, *, fold_scores: list[float] | None = None, step: int | None = None, **metrics_kwargs: Any
    ) -> None:
        """Логирование метрик качества в MLflow."""
        metrics: dict[str, float] = {}
        for k, v in metrics_kwargs.items():
            if k == "extra_metrics" and isinstance(v, dict):
                metrics.update({ek: round(float(ev), 5) for ek, ev in v.items() if ev is not None})
            elif v is not None:
                metrics[k] = round(float(v), 5)
        if fold_scores:
            for idx, score in enumerate(fold_scores):
                metrics[f"fold_{idx}_logloss"] = round(float(score), 5)
        mlflow.log_metrics(metrics, step=step)
        self._logged_metrics.update(metrics)
        self._has_logged_metrics = True

    def log_confusion_matrix(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5, artifact_path: str = "graphics"
    ) -> None:
        """Построение и логирование матрицы ошибок."""
        ExperimentGraphicsDispatcher.log_confusion_matrix(y_true, y_pred_proba, threshold, artifact_path)
        self._has_logged_graphics = True

    def log_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, artifact_path: str = "graphics") -> None:
        """Построение и логирование ROC-кривой."""
        ExperimentGraphicsDispatcher.log_roc_curve(y_true, y_pred_proba, artifact_path)
        self._has_logged_graphics = True

    def log_calibration_curve(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10, artifact_path: str = "graphics"
    ) -> float:
        """Построение и логирование калибровочной кривой (без лишних метрик)."""
        brier = ExperimentGraphicsDispatcher.log_calibration_curve(y_true, y_pred_proba, n_bins, artifact_path)
        self._has_logged_graphics = True
        return brier

    def log_prediction_distribution(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, artifact_path: str = "graphics"
    ) -> None:
        """Построение и логирование гистограммы вероятностей по классам."""
        ExperimentGraphicsDispatcher.log_prediction_distribution(y_true, y_pred_proba, artifact_path)
        self._has_logged_graphics = True

    def log_feature_tree(self, tree: dict[str, Any], filename: str = "feature_tree.json") -> None:
        """Логирование полного иерархического дерева признаков (`feature_tree.json`)."""
        mlflow.log_dict(tree, filename)
        self._has_logged_feature_tree = True

    def log_feature_store_state(
        self, state_dict: dict[str, Any], filename: str = "features/feature_store_state.json"
    ) -> None:
        """Логирование обученного JSON-состояния `FeatureStore` для воспроизведения в продакшене."""
        mlflow.log_dict(state_dict, filename)

    def log_submission(
        self,
        submission: pl.DataFrame | pd.DataFrame,
        filename: str = "submission.csv",
        artifact_path: str | None = None,
    ) -> None:
        """Логирование CSV-файла сабмишена."""
        SubmissionLogger.log(submission=submission, filename=filename, artifact_path=artifact_path)
        self._has_logged_submission = True

    def log_model_instance(
        self,
        model: Any,
        artifact_path: str = "model",
        register_as: str | None = "bot_detector",
        metric_for_promotion: str = "val_logloss",
        mode: MetricOptimizationMode = MetricOptimizationMode.MIN,
    ) -> Any | None:
        """Регистрация модели в MLflow Model Registry и отложенный промоушн при выходе из контекста."""
        return RunLifecycleManager.log_model_and_schedule_promotion(
            tracker=self,
            model=model,
            artifact_path=artifact_path,
            register_as=register_as,
            metric_for_promotion=metric_for_promotion,
            mode=mode,
        )

    def log_all_graphics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        artifact_path: str = "graphics",
    ) -> None:
        """Логирование 4 канонических графиков качества модели."""
        ExperimentGraphicsDispatcher.log_all_graphics(
            tracker=self,
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            artifact_path=artifact_path,
        )


__all__ = ["ExperimentTracker"]
