from enum import StrEnum


class HealthStatus(StrEnum):
    """Статусы работоспособности микросервиса."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
