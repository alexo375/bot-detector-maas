"""Диспетчер построения и логирования графиков качества в MLflow."""

from typing import Any

import numpy as np

from project.tracking.experiment_tracker.loggers.graphics import GraphicsLogger


class ExperimentGraphicsDispatcher:
    """Управление построением и логированием 4 канонических графиков оценки классификатора."""

    @staticmethod
    def log_confusion_matrix(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        artifact_path: str = "graphics",
    ) -> None:
        """Построение и логирование матрицы ошибок классификации."""
        GraphicsLogger.log_confusion_matrix(
            y_true=y_true, y_pred_proba=y_pred_proba, threshold=threshold, artifact_path=artifact_path
        )

    @staticmethod
    def log_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        artifact_path: str = "graphics",
    ) -> None:
        """Построение и логирование ROC-кривой."""
        GraphicsLogger.log_roc_curve(y_true=y_true, y_pred_proba=y_pred_proba, artifact_path=artifact_path)

    @staticmethod
    def log_calibration_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
        artifact_path: str = "graphics",
    ) -> float:
        """Построение кривой калибровки вероятностей."""
        return GraphicsLogger.log_calibration_curve(
            y_true=y_true, y_pred_proba=y_pred_proba, n_bins=n_bins, artifact_path=artifact_path
        )

    @staticmethod
    def log_prediction_distribution(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        artifact_path: str = "graphics",
    ) -> None:
        """Построение и логирование гистограммы распределения предсказанных вероятностей."""
        GraphicsLogger.log_prediction_distribution(
            y_true=y_true, y_pred_proba=y_pred_proba, artifact_path=artifact_path
        )

    @staticmethod
    def log_all_graphics(
        tracker: Any,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        artifact_path: str = "graphics",
    ) -> None:
        """Комплексное логирование 4 канонических графиков."""
        tracker.log_confusion_matrix(y_true, y_pred_proba, artifact_path=artifact_path)
        tracker.log_roc_curve(y_true, y_pred_proba, artifact_path=artifact_path)
        tracker.log_calibration_curve(y_true, y_pred_proba, artifact_path=artifact_path)
        tracker.log_prediction_distribution(y_true, y_pred_proba, artifact_path=artifact_path)


__all__ = ["ExperimentGraphicsDispatcher"]
