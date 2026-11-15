from dataclasses import dataclass
from typing import Any

from project.pipeline.mlflow_retrainer.schemas.metrics import GatekeeperMetrics


@dataclass
class RetrainEvaluationResult:
    """Результаты переобучения моделей и отбора признаков."""

    metrics: GatekeeperMetrics | dict[str, Any]
    diff_data: dict[str, Any]
    selected_features: list[str] | None = None
    selected_features_by_model: dict[str, list[str]] | None = None
    selected_feature_indices_by_model: dict[str, list[int]] | None = None
    total_features: int | None = None
    y_true: list[int] | None = None
    y_pred_proba: list[float] | None = None
    models: dict[str, Any] | None = None
    run_id: str = ""
    level1_runs: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Сериализация в словарь."""
        metrics_dict = self.metrics.to_dict() if hasattr(self.metrics, "to_dict") else self.metrics
        return {
            "metrics": metrics_dict,
            "diff_data": self.diff_data,
            "selected_features": self.selected_features,
            "selected_features_by_model": self.selected_features_by_model,
            "selected_feature_indices_by_model": self.selected_feature_indices_by_model,
            "total_features": self.total_features,
            "y_true": self.y_true,
            "y_pred_proba": self.y_pred_proba,
            "models": self.models,
            "run_id": self.run_id,
            "level1_runs": self.level1_runs,
        }
