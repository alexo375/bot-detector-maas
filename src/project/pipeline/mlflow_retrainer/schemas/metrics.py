from dataclasses import asdict, dataclass
from typing import Any

from project.pipeline.mlflow_retrainer.schemas.decision import GatekeeperDecision


@dataclass
class GatekeeperMetrics:
    """Метрики оценки дообученных моделей и вердикт гейткипера."""

    champion_loss: float | None
    challenger_loss: float
    catboost_val_loss: float
    lightgbm_val_loss: float
    xgboost_val_loss: float
    meta_stacker_val_loss: float
    promoted: bool
    decision: GatekeeperDecision
    reason: str
    val_logloss: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GatekeeperMetrics":
        """Создание объекта метрик из словаря с явной валидацией Enum."""
        raw_champ = data.get("champion_loss")
        return cls(
            champion_loss=float(raw_champ) if raw_champ is not None else None,
            challenger_loss=float(data.get("challenger_loss", 0.0)),
            catboost_val_loss=float(data.get("catboost_val_loss", 0.0)),
            lightgbm_val_loss=float(data.get("lightgbm_val_loss", 0.0)),
            xgboost_val_loss=float(data.get("xgboost_val_loss", 0.0)),
            meta_stacker_val_loss=float(data.get("meta_stacker_val_loss", data.get("challenger_loss", 0.0))),
            promoted=bool(data.get("promoted", False)),
            decision=GatekeeperDecision(data.get("decision", GatekeeperDecision.UNKNOWN)),
            reason=str(data.get("reason", "")),
            val_logloss=float(data["val_logloss"]) if data.get("val_logloss") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Сериализация метрик в словарь."""
        return asdict(self)
