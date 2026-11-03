"""Централизованный резолвер внешних и оперативных URL для мониторинга, алертов и UI.

Позволяет формировать ссылки на дашборды Grafana, деплойменты Prefect,
реестр моделей MLflow и документацию API.
"""

import os
from dataclasses import dataclass, field
from typing import Final

from dotenv import load_dotenv

from project.core.paths import paths

load_dotenv(paths.env_file, override=False)


@dataclass(frozen=True)
class ServiceUrls:
    """Менеджер ссылок на внутренние и внешние интерфейсы ML-платформы."""

    public_host: str = field(default_factory=lambda: os.getenv("PUBLIC_HOST", "localhost"))
    grafana_port: int = field(default_factory=lambda: int(os.getenv("GRAFANA_PORT", "3000")))
    prefect_port: int = field(default_factory=lambda: int(os.getenv("PREFECT_PORT", "4200")))
    mlflow_port: int = field(default_factory=lambda: int(os.getenv("MLFLOW_PORT", "5000")))

    @property
    def grafana_drift_dashboard(self) -> str:
        """Прямая ссылка на дашборд Data Drift в Grafana."""
        return f"http://{self.public_host}:{self.grafana_port}/d/data-drift-overview"

    @property
    def prefect_deployments(self) -> str:
        """Прямая ссылка на страницу деплойментов в Prefect UI (Prefect 3)."""
        return f"http://{self.public_host}:{self.prefect_port}/v2/deployments"

    @property
    def prefect_retrain_flow(self) -> str:
        """Ссылка на страницу деплойментов в Prefect UI."""
        return self.prefect_deployments

    @property
    def mlflow_models(self) -> str:
        """Прямая ссылка на реестр зарегистрированных моделей в MLflow."""
        return f"http://{self.public_host}:{self.mlflow_port}/#/models"


service_urls: Final = ServiceUrls()

__all__ = ["ServiceUrls", "service_urls"]
