from pydantic import BaseModel

from project.api.schemas.health.cache_backend import CacheBackend
from project.api.schemas.health.health_status import HealthStatus


class HealthResponse(BaseModel):
    """Схема ответа endpoint проверки работоспособности сервиса."""

    status: HealthStatus = HealthStatus.HEALTHY
    models_loaded: bool
    cache_backend: CacheBackend
    rabbitmq_connected: bool = False
    version: str = "0.1.0"
