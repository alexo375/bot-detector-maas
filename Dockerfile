# stage 1: сборка venv через uv
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Установка всех основных зависимостей проекта (включая mlflow и psycopg2-binary)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# stage 2: единый итоговый образ проекта
FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

# Копирование готового виртуального окружения
COPY --from=builder /app/.venv /app/.venv

# Копирование исходного кода и конфигураций деплоя
COPY src/ /app/src/
COPY deploy/ /app/deploy/

EXPOSE 8080 5000

HEALTHCHECK --interval=5s --timeout=3s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')" || exit 1

CMD ["uvicorn", "project.api.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
