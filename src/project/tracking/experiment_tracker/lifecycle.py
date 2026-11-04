"""Управление жизненным циклом ранов MLflow: подготовка, тегирование и MLOps-валидация."""

import platform
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.tracking import MlflowClient

from project.core.exceptions import DomainValidationError
from project.core.logging import ProjectLogger
from project.tracking.experiment_tracker.loggers.model_registry import ModelRegistryService
from project.tracking.experiment_tracker.schemas import (
    ExperimentEnvironment,
    MetricOptimizationMode,
    TrackerRunStatus,
)

if TYPE_CHECKING:
    from project.tracking.experiment_tracker.tracker import ExperimentTracker

logger = ProjectLogger.get_logger(__name__)


class RunLifecycleManager:
    """Управление жизненным циклом ранов MLflow: подготовка, тегирование и MLOps-валидация."""

    @staticmethod
    def ensure_experiment(
        tracking_uri: str,
        experiment_name: str,
        artifact_location: str | None = None,
    ) -> None:
        """Создание эксперимента в MLflow при отсутствии."""
        mlflow.set_tracking_uri(tracking_uri)
        try:
            client = MlflowClient(tracking_uri=tracking_uri)
            if client.get_experiment_by_name(experiment_name) is None:
                art_loc = artifact_location
                if not art_loc and tracking_uri.startswith("sqlite:///"):
                    art_dir = Path(tracking_uri.replace("sqlite:///", "")).parent / "artifacts"
                    art_dir.mkdir(parents=True, exist_ok=True)
                    art_loc = art_dir.resolve().as_uri()
                if art_loc:
                    client.create_experiment(experiment_name, artifact_location=art_loc)
                else:
                    client.create_experiment(experiment_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not pre-create MLflow experiment: %s", exc)

    @staticmethod
    def set_system_tags(environment: ExperimentEnvironment) -> None:
        """Запись системных метаданных (платформа, версия python, среда окружения)."""
        try:
            mlflow.set_tag("environment", environment.value)
            mlflow.set_tag("python_version", sys.version.split()[0])
            mlflow.set_tag("os_platform", platform.platform())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to set system tags: %s", exc)

    @staticmethod
    def start_run(tracker: "ExperimentTracker") -> "ExperimentTracker":
        """Запуск активного MLflow рана и применение отложенных параметров/датасетов."""
        RunLifecycleManager.ensure_experiment(
            tracking_uri=tracker.tracking_uri,
            experiment_name=tracker.experiment_name,
            artifact_location=tracker.artifact_location,
        )
        mlflow.set_experiment(tracker.experiment_name)
        tracker.active_run = mlflow.start_run(
            run_name=tracker.run_name,
            description=tracker.description,
            tags=tracker.tags,
            nested=tracker.nested,
        )
        try:
            tracker._start_time = time.perf_counter()
            RunLifecycleManager.set_system_tags(tracker.environment)

            if tracker._pending_params:
                tracker.log_params(**tracker._pending_params)
                tracker._pending_params = None

            if tracker._pending_dataset:
                d, n, t, fn = tracker._pending_dataset
                tracker.log_dataset(data=d, name=n, targets=t, feature_names=fn)
                tracker._pending_dataset = None

            if tracker._pending_metrics:
                tracker.log_metrics(**tracker._pending_metrics)
                tracker._pending_metrics = None
        except Exception:
            mlflow.end_run()
            tracker.active_run = None
            raise

        return tracker

    @staticmethod
    def end_run(tracker: "ExperimentTracker", exc_type: Any, exc_val: Any) -> bool:
        """Завершение активного MLflow рана с валидацией артефактов и промоушеном моделей."""
        if not tracker.active_run:
            return False

        if exc_type is not None:
            mlflow.set_tag("status", TrackerRunStatus.FAILED)
            mlflow.set_tag("error", str(exc_val))
            mlflow.end_run()
            tracker.active_run = None
            return False

        RunLifecycleManager.validate_run_completion(tracker)

        if tracker._pending_model_promotion:
            try:
                ModelRegistryService.evaluate_promotion(**tracker._pending_model_promotion)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to evaluate model promotion: %s", exc)
            tracker._pending_model_promotion = None

        RunLifecycleManager.finalize_run(tracker, tracker._start_time)
        return False

    @staticmethod
    def sanitize_params(
        model_family: str | None,
        feature_names: list[str] | None,
        params: dict[str, Any] | None,
        extra_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Очистка и сериализация параметров модели без мусорных полей."""
        _ = feature_names
        clean: dict[str, Any] = {}
        if model_family is not None:
            clean["model_family"] = model_family
        for src in (params or {}, extra_params):
            for k, v in src.items():
                clean[k] = str(v) if isinstance(v, (list, dict, tuple)) else v

        for junk in ("n_features", "num_rows", "dataset_num_rows", "dataset_name"):
            clean.pop(junk, None)
        clean.setdefault("cv_method", "GroupKFold(n_splits=5)")
        clean.setdefault("seed", 42)
        return clean

    @staticmethod
    def log_model_and_schedule_promotion(
        tracker: "ExperimentTracker",
        model: Any,
        artifact_path: str,
        register_as: str | None,
        metric_for_promotion: str,
        mode: MetricOptimizationMode,
    ) -> Any | None:
        """Логирование модели в реестр MLflow и планирование оценки промоушена."""
        mv = ModelRegistryService.log_model(
            model=model,
            artifact_path=artifact_path,
            register_as=register_as,
            active_run=tracker.active_run,
            tracking_uri=tracker.tracking_uri,
        )
        if register_as and mv:
            tracker._pending_model_promotion = {
                "mv": mv,
                "register_as": register_as,
                "metric_for_promotion": metric_for_promotion,
                "mode": mode,
                "active_run": tracker.active_run,
                "tracking_uri": tracker.tracking_uri,
                "logged_metrics": tracker._logged_metrics,
            }
        return mv

    @staticmethod
    def validate_run_completion(tracker: "ExperimentTracker") -> None:
        """MLOps-проверка обязательных артефактов и метрик перед закрытием рана."""
        checks = [
            (tracker.require_description, bool(tracker.description), "Отсутствует описание эксперимента (description)"),
            (
                tracker.require_dataset,
                tracker._has_logged_dataset,
                "Не залогирован датасет! Вызовите tracker.log_dataset()",
            ),
            (True, tracker._has_logged_params, "Не залогированы гиперпараметры модели! Вызовите tracker.log_params()"),
            (True, tracker._has_logged_metrics, "Не залогированы метрики качества! Вызовите tracker.log_metrics()"),
            (
                tracker.require_graphics,
                tracker._has_logged_graphics,
                "Обязательные графики не залогированы! Вызовите tracker.log_all_graphics() или отключите требование.",
            ),
            (
                tracker.require_feature_tree,
                tracker._has_logged_feature_tree,
                "Обязательное дерево признаков не залогировано! Вызовите tracker.log_feature_tree().",
            ),
            (
                tracker.require_submission,
                tracker._has_logged_submission,
                "Обязательный файл сабмишена не залогирован! Вызовите tracker.log_submission().",
            ),
            (
                "val_logloss" in tracker._logged_metrics,
                True,
                "Критическая метрика val_logloss не залогирована!",
            ),
        ]

        errors = [msg for req, ok, msg in checks if req and not ok]
        if errors:
            err_msg = (
                f"Ошибка MLOps-валидации: эксперимент '{tracker.run_name}' не удовлетворяет строгим критериям качества:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
            logger.error(err_msg)
            mlflow.set_tag("status", TrackerRunStatus.FAILED)
            mlflow.set_tag("validation_error", err_msg)
            mlflow.end_run()
            tracker.active_run = None
            raise DomainValidationError(err_msg)

    @staticmethod
    def finalize_run(tracker: "ExperimentTracker", start_time: float | None) -> None:
        """Успешное завершение рана и фиксация длительности (`execution_time_seconds`)."""
        if start_time is not None:
            duration_sec = time.perf_counter() - start_time
            mlflow.log_metric("execution_time_seconds", round(duration_sec, 2))
        mlflow.set_tag("status", TrackerRunStatus.SUCCESS)
        mlflow.end_run()
        tracker.active_run = None


__all__ = ["RunLifecycleManager"]
