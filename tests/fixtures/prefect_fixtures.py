"""Изолированная сессионная фикстура Prefect для автономного выполнения @flow в тестах."""

from collections.abc import Generator

import pytest
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(scope="session")
def prefect_isolated_server() -> Generator[None]:
    """Запуск единственного эфемерного тестового сервера Prefect на воркер xdist."""
    with prefect_test_harness(server_startup_timeout=30):
        yield
