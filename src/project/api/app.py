import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from project import Reproducibility, __version__
from project.api.metrics import metrics_manager
from project.api.routers.alerts import router as alerts_router
from project.api.routers.health import router as health_router
from project.api.routers.prediction import router as prediction_router
from project.core.exceptions import (
    DomainValidationError,
    InfrastructureError,
    ModelArtifactError,
    ProjectError,
)
from project.core.logging import LoggingConfigurator, ProjectLogger
from project.infrastructure.container import ServiceContainer

logger = ProjectLogger.get_logger(__name__)

API_DESCRIPTION = """
API сервиса бинарной классификации собеседников (бот / человек) в текстовых диалогах.
"""

TAGS_METADATA = [
    {
        "name": "Predictions",
        "description": "Классификация диалогов и получение результатов.",
    },
    {
        "name": "Health",
        "description": "Проверка состояния сервиса и компонентов инфраструктуры.",
    },
]


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Управление жизненным циклом приложения FastAPI."""
    if not os.getenv("PYTEST_CURRENT_TEST"):
        LoggingConfigurator.configure()
    Reproducibility.seed(42)
    services = ServiceContainer.get_instance()
    await services.initialize()
    yield
    await services.close()


def create_app() -> FastAPI:
    """Фабрика создания FastAPI приложения."""
    application = FastAPI(
        title="Bot Detection Service API",
        description=API_DESCRIPTION.strip(),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
    )

    metrics_manager.setup(application)
    application.include_router(health_router)
    application.include_router(prediction_router)
    application.include_router(alerts_router)

    @application.exception_handler(DomainValidationError)
    async def domain_validation_handler(_request: Request, exc: DomainValidationError) -> JSONResponse:
        logger.warning("Domain validation error: %s", exc)
        return JSONResponse(
            status_code=422,
            content={"error": "domain_validation_error", "message": exc.message, "details": exc.details},
        )

    @application.exception_handler(ModelArtifactError)
    async def model_artifact_handler(_request: Request, exc: ModelArtifactError) -> JSONResponse:
        logger.error("Model artifact error: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"error": "model_unavailable", "message": exc.message},
        )

    @application.exception_handler(InfrastructureError)
    async def infrastructure_handler(_request: Request, exc: InfrastructureError) -> JSONResponse:
        logger.error("Infrastructure error: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"error": "infrastructure_unavailable", "message": exc.message},
        )

    @application.exception_handler(ProjectError)
    async def project_error_handler(_request: Request, exc: ProjectError) -> JSONResponse:
        logger.error("Internal project error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": exc.message},
        )

    return application


app = create_app()
