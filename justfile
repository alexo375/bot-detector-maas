set shell := ["bash", "-euo", "pipefail", "-c"]

# Показать список доступных сценариев
default:
    @just --list

# === Комплексные сценарии валидации ===

# Быстрый цикл валидации для разработчика перед коммитом
[group('check')]
check-fast: check-static
    uv run poe test-unit

# Автоматическое исправление стиля кода и форматирования
[group('check')]
fix:
    uv run poe lint-fix
    uv run poe format

# Базовые статические проверки
[group('check')]
check-static:
    uv run poe lint
    uv run poe format-check
    uv run poe typecheck
    uv run poe check-lengths
    uv run poe oop-check

# === Эксплуатация и MLOps сценарии ===

# Комплексный аудит доступности всех компонентов рантайма (Triton Inference Server, API Gateway, Prometheus Metrics)
[group('ops')]
verify-system:
    curl -sf http://localhost:8000/v2/health/ready > /dev/null && echo "[Health] Triton Server: OK"
    curl -sf http://localhost:8080/health | jq .
    curl -sf http://localhost:8080/metrics > /dev/null && echo "[Health] Prometheus Metrics: OK"

# Имитация дрейфа данных в ClickHouse DWH для проверки мониторинга и алертинга
[group('ml')]
drift-inject *args:
    uv run python scripts/inject_drift.py {{args}}

# Сброс искусственного дрейфа данных к baseline-состоянию
[group('ml')]
drift-reset:
    uv run python scripts/inject_drift.py --reset

# Запуск ускоренного переобучения (3 Optuna-прогона, 25 деревьев) в контейнере Prefect Worker
[group('ml')]
retrain-fast:
    docker compose exec prefect-worker python -m project.pipeline.mlflow_retrainer.flow --force --fast

# Запуск полномасштабного переобучения всего ансамбля (ноутбучный бюджет: 300 Optuna-триалов, 750 деревьев) в Prefect Worker
[group('ml')]
retrain-full:
    docker compose exec prefect-worker python -m project.pipeline.mlflow_retrainer.flow --force --n-estimators 750

# Ручной откат ансамбля моделей на предыдущую валидную версию через Prefect Worker
[group('ml')]
rollback *args:
    docker compose exec prefect-worker python -m project.pipeline.model_rollback.flow {{args}}
