import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from project.core.schemas import AuditMode
from project.infrastructure.rabbitmq.schemas.dialogue_verdict import DialogueVerdict


@dataclass
class DialogueEvent:
    """Структурированное событие завершенного диалога для мониторинга и аналитики в ClickHouse."""

    dialog_id: str
    timestamp: str
    num_messages: int
    messages_json: str
    features_p0: list[float]
    features_p1: list[float]
    prob_p0: float
    prob_p1: float
    is_bot_p0: int
    is_bot_p1: int
    verdict: DialogueVerdict
    confidence: str
    cached: int
    latency_ms: float
    audit_mode: AuditMode = AuditMode.PRODUCTION
    base_model_probs: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        dialog_id: str,
        messages: list[dict[str, Any]] | list[Any],
        eval_result: Any,
        cached: bool,
        latency_ms: float,
        audit_mode: AuditMode = AuditMode.PRODUCTION,
    ) -> "DialogueEvent":
        """Фабричный метод создания события из результатов инференса."""
        v0 = eval_result["verdicts"][0]
        v1 = eval_result["verdicts"][1]

        is_bot_0 = bool(v0.get("is_bot") if isinstance(v0, dict) else v0.is_bot)
        is_bot_1 = bool(v1.get("is_bot") if isinstance(v1, dict) else v1.is_bot)
        prob_0 = float(
            v0.get("bot_probability", 0.0) if isinstance(v0, dict) else (getattr(v0, "bot_probability", 0.0) or 0.0)
        )
        prob_1 = float(
            v1.get("bot_probability", 0.0) if isinstance(v1, dict) else (getattr(v1, "bot_probability", 0.0) or 0.0)
        )

        if is_bot_0 and is_bot_1:
            verdict = DialogueVerdict.BOTH_BOTS
        elif is_bot_0:
            verdict = DialogueVerdict.P0_BOT
        elif is_bot_1:
            verdict = DialogueVerdict.P1_BOT
        else:
            verdict = DialogueVerdict.BOTH_HUMANS

        raw_features = (
            eval_result.get("features") if isinstance(eval_result, dict) else getattr(eval_result, "features", None)
        )
        features = raw_features if raw_features is not None else []
        features_p0 = features[0] if len(features) > 0 else []
        features_p1 = features[1] if len(features) > 1 else []

        raw_msgs = [m.model_dump() if hasattr(m, "model_dump") else m for m in messages]

        explicit_conf = (
            eval_result.get("confidence", "")
            if isinstance(eval_result, dict)
            else getattr(eval_result, "confidence", "")
        )
        if explicit_conf:
            confidence = str(explicit_conf)
        else:
            margin = max(abs(prob_0 - 0.5), abs(prob_1 - 0.5))
            confidence = "high" if margin >= 0.3 else ("medium" if margin >= 0.15 else "low")

        raw_base = (
            eval_result.get("base_model_probs")
            if isinstance(eval_result, dict)
            else getattr(eval_result, "base_model_probs", None)
        )
        base_model_probs = (
            {
                str(model): [float(p) for p in probs]
                for model, probs in raw_base.items()
                if isinstance(probs, (list, tuple))
            }
            if isinstance(raw_base, dict)
            else {}
        )

        return cls(
            dialog_id=dialog_id,
            timestamp=datetime.now(UTC).isoformat(),
            num_messages=len(messages),
            messages_json=json.dumps(raw_msgs, ensure_ascii=False),
            features_p0=[float(x) for x in features_p0],
            features_p1=[float(x) for x in features_p1],
            prob_p0=prob_0,
            prob_p1=prob_1,
            is_bot_p0=int(is_bot_0),
            is_bot_p1=int(is_bot_1),
            verdict=verdict,
            confidence=confidence,
            cached=int(cached),
            latency_ms=float(latency_ms),
            audit_mode=audit_mode,
            base_model_probs=base_model_probs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DialogueEvent":
        """Создание события из словаря."""
        raw_base = data.get("base_model_probs") or {}
        base_model_probs = (
            {
                str(model): [float(p) for p in probs]
                for model, probs in raw_base.items()
                if isinstance(probs, (list, tuple))
            }
            if isinstance(raw_base, dict)
            else {}
        )
        return cls(
            dialog_id=str(data["dialog_id"]),
            timestamp=str(data["timestamp"]),
            num_messages=int(data["num_messages"]),
            messages_json=str(data["messages_json"]),
            features_p0=[float(x) for x in data.get("features_p0", [])],
            features_p1=[float(x) for x in data.get("features_p1", [])],
            prob_p0=float(data["prob_p0"]),
            prob_p1=float(data["prob_p1"]),
            is_bot_p0=int(data["is_bot_p0"]),
            is_bot_p1=int(data["is_bot_p1"]),
            verdict=DialogueVerdict(data["verdict"]),
            confidence=str(data.get("confidence", "")),
            cached=int(data["cached"]),
            latency_ms=float(data["latency_ms"]),
            audit_mode=AuditMode(data.get("audit_mode", AuditMode.PRODUCTION)),
            base_model_probs=base_model_probs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Сериализация события в словарь."""
        return {
            "dialog_id": self.dialog_id,
            "timestamp": self.timestamp,
            "num_messages": self.num_messages,
            "messages_json": self.messages_json,
            "features_p0": self.features_p0,
            "features_p1": self.features_p1,
            "prob_p0": self.prob_p0,
            "prob_p1": self.prob_p1,
            "is_bot_p0": self.is_bot_p0,
            "is_bot_p1": self.is_bot_p1,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "cached": self.cached,
            "latency_ms": self.latency_ms,
            "audit_mode": str(self.audit_mode),
            "base_model_probs": self.base_model_probs,
        }
