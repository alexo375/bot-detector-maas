from dataclasses import dataclass
from typing import Any

from project.drift.data_drift import DataDriftSummary
from project.pipeline.mlflow_retrainer.schemas.decision import GatekeeperDecision
from project.pipeline.mlflow_retrainer.schemas.metrics import GatekeeperMetrics
from project.pipeline.mlflow_retrainer.schemas.status import FlowStatus


@dataclass
class FlowExecutionResult:
    """Результаты запуска регулярного пайплайна автоматического мониторинга и дообучения."""

    status: FlowStatus
    retrained: bool = False
    promoted: bool = False
    decision: GatekeeperDecision = GatekeeperDecision.UNKNOWN
    reason: str | None = None
    metrics: GatekeeperMetrics | dict[str, Any] | None = None
    feature_importance_diff: dict[str, Any] | None = None
    drift_summary: DataDriftSummary | dict[str, Any] | None = None
    triton_reload: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowExecutionResult":
        """Создание результата выполнения Flow из словаря с явной валидацией Enum."""
        raw_metrics = data.get("metrics")
        metrics: GatekeeperMetrics | None = None
        if raw_metrics is not None:
            if isinstance(raw_metrics, GatekeeperMetrics):
                metrics = raw_metrics
            elif isinstance(raw_metrics, dict):
                metrics = GatekeeperMetrics.from_dict(raw_metrics)

        return cls(
            status=FlowStatus(data["status"]),
            retrained=bool(data.get("retrained", False)),
            promoted=bool(data.get("promoted", False)),
            decision=GatekeeperDecision(data.get("decision", GatekeeperDecision.UNKNOWN)),
            reason=data.get("reason"),
            metrics=metrics,
            feature_importance_diff=data.get("feature_importance_diff"),
            drift_summary=data.get("drift_summary"),
            triton_reload=data.get("triton_reload"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Сериализация результата Flow в словарь."""
        data: dict[str, Any] = {
            "status": self.status,
            "retrained": self.retrained,
            "promoted": self.promoted,
            "decision": self.decision,
            "reason": self.reason,
            "feature_importance_diff": self.feature_importance_diff,
            "triton_reload": self.triton_reload,
        }
        if self.metrics is not None:
            data["metrics"] = self.metrics.to_dict() if hasattr(self.metrics, "to_dict") else self.metrics
        else:
            data["metrics"] = None

        if self.drift_summary is not None:
            data["drift_summary"] = (
                self.drift_summary.to_dict() if hasattr(self.drift_summary, "to_dict") else self.drift_summary
            )
        else:
            data["drift_summary"] = None

        return data
