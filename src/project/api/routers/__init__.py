from project.api.routers.alerts import router as alerts_router
from project.api.routers.health import router as health_router
from project.api.routers.prediction import router as prediction_router

__all__ = ["alerts_router", "health_router", "prediction_router"]
