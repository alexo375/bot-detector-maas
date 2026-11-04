from enum import StrEnum


class MetricOptimizationMode(StrEnum):
    """Направление оптимизации метрики (минимизация или максимизация)."""

    MIN = "min"
    MAX = "max"
