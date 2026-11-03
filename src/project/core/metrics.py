"""Общие Prometheus-метрики инференса, разделяемые между API Gateway и InferenceWorker."""

from typing import Any

from prometheus_client import Counter, Histogram

INFERENCE_LATENCY = Histogram(
    "bot_detector_inference_latency_seconds",
    "Задержка инференса моделей и экстракции фичей (секунды)",
    buckets=[0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500, 1.0],
)

CACHE_HITS_TOTAL = Counter(
    "bot_detector_cache_hits_total",
    "Количество успешных попаданий в кэш диалогов Redis",
)

CACHE_MISSES_TOTAL = Counter(
    "bot_detector_cache_misses_total",
    "Количество промахов кэша диалогов (полный инференс)",
)

PREDICTIONS_TOTAL = Counter(
    "bot_detector_predictions_total",
    "Количество выданных предсказаний по классам вердиктов",
    labelnames=["verdict"],
)


class PredictionMetricsRecorder:
    """Классификатор и рекордер вердиктов предсказаний для метрик Prometheus."""

    @staticmethod
    def classify(verdicts: list[Any]) -> str:
        """Определение строкового класса вердикта пары участников для метрики Prometheus."""
        v0_bot = bool(verdicts[0]["is_bot"] if isinstance(verdicts[0], dict) else verdicts[0].is_bot)
        v1_bot = bool(verdicts[1]["is_bot"] if isinstance(verdicts[1], dict) else verdicts[1].is_bot)
        if v0_bot and v1_bot:
            return "both_bots"
        if v0_bot:
            return "p0_bot"
        if v1_bot:
            return "p1_bot"
        return "both_humans"

    @classmethod
    def record(cls, verdicts: list[Any]) -> None:
        """Инкремент счетчика PREDICTIONS_TOTAL по паре вердиктов диалога."""
        if len(verdicts) >= 2:
            PREDICTIONS_TOTAL.labels(verdict=cls.classify(verdicts)).inc()


__all__ = [
    "CACHE_HITS_TOTAL",
    "CACHE_MISSES_TOTAL",
    "INFERENCE_LATENCY",
    "PREDICTIONS_TOTAL",
    "PredictionMetricsRecorder",
]
