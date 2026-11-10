from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class ChatLogRecord:
    """Типизированная запись аудита диалога и результатов инференса для таблицы chat_logs."""

    dialog_id: str
    timestamp: datetime
    num_messages: int
    messages_json: str
    features_p0: list[float]
    features_p1: list[float]
    prob_p0: float
    prob_p1: float
    is_bot_p0: int
    is_bot_p1: int
    verdict: str
    confidence: str
    cached: int
    latency_ms: float
    base_model_probs: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def column_names(cls) -> list[str]:
        """Список имен колонок таблицы chat_logs в ClickHouse."""
        return [
            "dialog_id",
            "timestamp",
            "num_messages",
            "messages_json",
            "features_p0",
            "features_p1",
            "prob_p0",
            "prob_p1",
            "base_model_probs",
            "is_bot_p0",
            "is_bot_p1",
            "verdict",
            "confidence",
            "cached",
            "latency_ms",
        ]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ChatLogRecord":
        """Фабрика создания записи из входящего JSON-события брокера сообщений."""
        ts_str = payload.get("timestamp")
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str)
            except Exception:  # noqa: BLE001  # Resilient fallback
                dt = datetime.now(UTC)
        else:
            dt = datetime.now(UTC)

        raw_base = payload.get("base_model_probs") or {}
        base_probs = {
            str(model): [float(p) for p in probs]
            for model, probs in raw_base.items()
            if isinstance(probs, (list, tuple))
        }

        return cls(
            dialog_id=str(payload.get("dialog_id", "")),
            timestamp=dt,
            num_messages=int(payload.get("num_messages", 0)),
            messages_json=str(payload.get("messages_json", "[]")),
            features_p0=[float(x) for x in payload.get("features_p0", [])],
            features_p1=[float(x) for x in payload.get("features_p1", [])],
            prob_p0=float(payload.get("prob_p0", 0.0)),
            prob_p1=float(payload.get("prob_p1", 0.0)),
            is_bot_p0=int(payload.get("is_bot_p0", 0)),
            is_bot_p1=int(payload.get("is_bot_p1", 0)),
            verdict=str(payload.get("verdict", "unknown")),
            confidence=str(payload.get("confidence", "unknown")),
            cached=int(payload.get("cached", 0)),
            latency_ms=float(payload.get("latency_ms", 0.0)),
            base_model_probs=base_probs,
        )

    def to_row(self) -> list[Any]:
        """Преобразование записи в кортеж значений для вставки через ClickHouse-клиент."""
        return [
            self.dialog_id,
            self.timestamp,
            self.num_messages,
            self.messages_json,
            self.features_p0,
            self.features_p1,
            self.prob_p0,
            self.prob_p1,
            self.base_model_probs,
            self.is_bot_p0,
            self.is_bot_p1,
            self.verdict,
            self.confidence,
            self.cached,
            self.latency_ms,
        ]
