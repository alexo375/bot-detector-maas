from fastapi import APIRouter, Response, status

from project import __version__
from project.api.schemas.health.cache_backend import CacheBackend
from project.api.schemas.health.health import HealthResponse
from project.api.schemas.health.health_status import HealthStatus
from project.infrastructure.container import ServiceContainer

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Комплексная проверка доступности сервиса и инфраструктуры",
    description="""
Проверяет доступность критических зависимостей шлюза:
* Статус подключения к кэшу Redis.
* Связность с брокером сообщений RabbitMQ.
* Доступность клиента моделей Triton Inference Server.
* Текущую версию сервиса.
""",
    responses={
        200: {"model": HealthResponse, "description": "Все критические зависимости доступны"},
        503: {"model": HealthResponse, "description": "Одна или несколько критических зависимостей недоступны"},
    },
)
def health_check(response: Response) -> HealthResponse:
    """Проверка доступности шлюза и загруженных моделей."""
    services = ServiceContainer.get_instance()
    cache_ok = bool(services.cache and getattr(services.cache, "is_connected", False))
    mq_ok = bool(services.publisher and services.publisher.is_connected)
    triton_ok = bool(services.triton_client and services.triton_client.is_healthy())

    cache_backend = CacheBackend.REDIS if cache_ok else CacheBackend.DISABLED
    overall_status = HealthStatus.HEALTHY if (cache_ok and mq_ok and triton_ok) else HealthStatus.UNHEALTHY
    if overall_status == HealthStatus.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status=overall_status,
        models_loaded=triton_ok,
        cache_backend=cache_backend,
        rabbitmq_connected=mq_ok,
        version=__version__,
    )


@router.get(
    "/health/live",
    summary="Kubernetes Liveness Probe",
    description="Легковесная проверка жизнеспособности HTTP процесса Uvicorn/FastAPI.",
    responses={
        200: {
            "content": {"application/json": {"example": {"status": "alive"}}},
            "description": "Процесс активен и обрабатывает HTTP запросы",
        }
    },
)
def liveness_check() -> dict[str, str]:
    """Liveness probe: проверка доступности HTTP процесса."""
    return {"status": "alive"}


@router.get(
    "/health/ready",
    summary="Kubernetes Readiness Probe",
    description="Проверка готовности сервиса к приему и обработке входящего клиентского трафика.",
    responses={
        200: {"model": HealthResponse, "description": "Сервис готов к обработке запросов"},
        503: {"model": HealthResponse, "description": "Сервис не готов к обработке запросов"},
    },
)
def readiness_check(response: Response) -> HealthResponse:
    """Readiness probe: проверка готовности сервиса к обработке запросов."""
    return health_check(response)
