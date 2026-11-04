"""Модульные фикстуры изолированного окружения MLflow Tracking и Registry."""

import shutil
from collections.abc import Generator
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from project.core.config import config


@pytest.fixture(scope="session")
def _mlflow_template_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Создает пред-мигрированную шаблонную SQLite БД MLflow один раз на сессию воркера."""
    tmpl_dir = tmp_path_factory.mktemp("mlflow_tmpl")
    db_path = tmpl_dir / "mlflow_template.db"
    artifacts_dir = tmpl_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{db_path}"
    client = MlflowClient(tracking_uri=tracking_uri)
    for exp_name in ("bot_detection_retraining", "bot-detection"):
        if client.get_experiment_by_name(exp_name) is None:
            client.create_experiment(
                exp_name,
                artifact_location=artifacts_dir.as_uri(),
            )
    return db_path


@pytest.fixture
def mlflow_sqlite_uri(tmp_path: Path) -> str:
    """Генератор локального URI SQLite базы данных MLflow для изолированного теста."""
    db_path = tmp_path / "mlflow.db"
    return f"sqlite:///{db_path}"


@pytest.fixture(autouse=True)
def auto_isolate_mlflow(tmp_path: Path, _mlflow_template_db: Path) -> Generator[str]:
    """Автоматическая глобальная изоляция MLflow Tracking для ВСЕХ тестов без исключения.

    Копирует готовую шаблонную БД SQLite вместо запуска Alembic-миграций на каждый тест,
    сохраняя 100% изоляцию между тестами и xdist-воркерами.
    """
    db_path = tmp_path / "mlflow_auto_isolated.db"
    shutil.copy2(_mlflow_template_db, db_path)
    tracking_uri = f"sqlite:///{db_path}"

    old_uri = config.mlflow_tracking_uri
    object.__setattr__(config, "mlflow_tracking_uri", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    try:
        yield tracking_uri
    finally:
        while mlflow.active_run() is not None:
            mlflow.end_run()
        object.__setattr__(config, "mlflow_tracking_uri", old_uri)
        mlflow.set_tracking_uri(old_uri)


@pytest.fixture
def isolated_mlflow_env(auto_isolate_mlflow: str) -> str:
    """Совместимая фикстура для тестов, явно запрашивающих isolated_mlflow_env."""
    return auto_isolate_mlflow
