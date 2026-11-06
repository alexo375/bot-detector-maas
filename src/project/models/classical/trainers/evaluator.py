"""Вспомогательные утилиты логирования артефактов и трекинга классических моделей."""

from typing import Any

import numpy as np
import polars as pl

from project.core.logging import ProjectLogger
from project.tracking import ExperimentTracker, ExperimentTrackerBuilder
from project.tracking.experiment_tracker.loggers.features import FeatureImportanceExtractor

logger = ProjectLogger.get_logger(__name__)


class TrainerArtifactManager:
    """Управление артефактами, деревьями признаков и инициализацией ExperimentTracker."""

    @staticmethod
    def initialize_tuning_tracker(
        config: Any,
        model_family: str,
        feature_names: list[str] | None,
    ) -> ExperimentTracker:
        """Создает и активирует `ExperimentTracker` для процесса тюнинга гиперпараметров."""
        tracker = (
            ExperimentTrackerBuilder(
                run_name=config.run_name,
                experiment_name=config.experiment_name,
            )
            .set_environment(config.environment)
            .set_description(config.description or f"Обучение и оптимизация модели {model_family}")
            .defer_dataset()
            .set_params(
                model_family=model_family,
                optimizer="Optuna",
                n_trials=config.n_trials,
                patience=config.patience,
                feature_names=feature_names,
            )
            .defer_metrics()
            .build()
        )
        tracker.__enter__()
        return tracker

    @staticmethod
    def prepare_feature_subsets(
        x_tr_mat: np.ndarray,
        x_te_mat: np.ndarray,
        feature_names: list[str],
        feature_ranking_indices: np.ndarray | None,
        optimal_features_count: int | None,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[int] | None]:
        """Выделяет оптимальный срез признаков согласно ранжированию и возвращает индексы колонок."""
        if feature_ranking_indices is not None and optimal_features_count is not None:
            sel_indices = [int(i) for i in feature_ranking_indices[:optimal_features_count]]
            return (
                x_tr_mat[:, sel_indices],
                x_te_mat[:, sel_indices],
                [feature_names[i] for i in sel_indices],
                sel_indices,
            )
        return x_tr_mat, x_te_mat, feature_names, None

    @staticmethod
    def initialize_tracker(
        config: Any,
        model_family: str,
        full_fit_df: pl.DataFrame,
        y_tr_arr: np.ndarray,
        final_feature_names: list[str],
    ) -> ExperimentTracker:
        """Создает и активирует `ExperimentTracker` для процесса обучения."""
        tracker = (
            ExperimentTrackerBuilder(
                run_name=config.run_name,
                experiment_name=config.experiment_name,
            )
            .set_environment(config.environment)
            .set_description(config.description or f"Обучение модели {model_family}")
            .set_dataset(
                data=full_fit_df,
                name=config.dataset_name,
                targets=y_tr_arr,
                feature_names=final_feature_names,
            )
            .set_params(
                model_family=model_family,
                optimizer="Manual",
                n_trials=config.n_trials,
                patience=config.patience,
            )
            .defer_metrics()
            .build()
        )
        tracker.__enter__()
        return tracker

    @staticmethod
    def _average_fold_importances(trained_models: list[Any], n_features: int) -> np.ndarray | None:
        vectors = [
            np.asarray(imp, dtype=np.float64).ravel()
            for model in trained_models
            if (imp := FeatureImportanceExtractor.extract(model)) is not None
        ]
        valid = [v for v in vectors if len(v) == n_features]
        if not valid:
            return None
        return np.asarray(np.mean(np.vstack(valid), axis=0), dtype=np.float64)

    @classmethod
    def log_artifacts_and_graphics(
        cls,
        tracker: ExperimentTracker,
        model_family: str,
        trained_models: list[Any],
        final_feature_names: list[str],
        y_tr_arr: np.ndarray,
        oof_calibrated: np.ndarray,
        x_tr_final: np.ndarray,
    ) -> dict[str, float]:
        """Логирование `feature_tree.json` (усредненного по фолдам) и 4 канонических графиков."""
        del x_tr_final
        importances = cls._average_fold_importances(trained_models, len(final_feature_names))
        importance_map: dict[str, float] = {}
        if importances is not None:
            sorted_idx = np.argsort(importances)[::-1]
            tree_features = [
                {"name": str(final_feature_names[i]), "importance": round(float(importances[i]), 5)} for i in sorted_idx
            ]
            importance_map = {str(final_feature_names[i]): round(float(importances[i]), 5) for i in sorted_idx}
        else:
            tree_features = [{"name": str(fn), "importance": None} for fn in final_feature_names]

        feature_tree = {
            "architecture": "SingleModel",
            "model_family": model_family,
            "n_features": len(final_feature_names),
            "features": tree_features,
        }
        tracker.log_feature_tree(feature_tree)
        tracker.log_all_graphics(y_true=y_tr_arr, y_pred_proba=oof_calibrated)
        return importance_map

    @staticmethod
    def register_model_if_configured(
        tracker: ExperimentTracker,
        config: Any,
        full_model: Any,
        model_family: str,
    ) -> None:
        """Регистрирует уже обученную финальную модель в MLflow Model Registry при наличии `register_as`."""
        if not config.register_as or full_model is None:
            return

        logger.info("Регистрация модели %s в MLflow Registry как '%s'", model_family, config.register_as)
        tracker.log_model_instance(
            full_model,
            artifact_path="model",
            register_as=config.register_as,
            metric_for_promotion="val_logloss",
        )


__all__ = ["TrainerArtifactManager"]
