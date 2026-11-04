"""Глобальная конфигурация pytest и регистрация модульных плагинов с фикстурами."""

import os

import matplotlib as mpl

# Изоляция тестов от внешнего Prefect сервера (переопределяем профиль ~/.prefect) и лимит потоков C++/ML
os.environ["PREFECT_API_URL"] = ""
os.environ.setdefault("PREFECT_SERVER_ALLOW_EPHEMERAL_MODE", "true")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

mpl.use("Agg")

pytest_plugins = [
    "tests.fixtures.mlflow_fixtures",
]
