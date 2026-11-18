set shell := ["bash", "-euo", "pipefail", "-c"]

# Показать список доступных сценариев
default:
    @just --list

# === Комплексные сценарии валидации ===

# Полный прогон всех проверок проекта: статика, безопасность, тесты с порогом покрытия
[group('check')]
check: check-static check-security
    uv run poe test-cov

# Быстрый цикл валидации для разработчика перед коммитом (вся статика + модульные тесты)
[group('check')]
check-fast: check-static
    uv run poe test-unit

# Автоматическое исправление стиля кода, неиспользуемых импортов и форматирования
[group('check')]
fix:
    uv run poe lint-fix
    uv run poe format

# Все статические валидации: линтинг, форматирование, типы, мертвый код, длины файлов, архитектурные границы
[group('check')]
check-static:
    uv run poe lint
    uv run poe format-check
    uv run poe typecheck
    uv run poe dead-code
    uv run poe check-lengths
    uv run poe oop-check
    uv run poe arch-check

# Проверки безопасности: статический аудит AST (Bandit) и сканирование уязвимостей зависимостей (pip-audit)
[group('check')]
check-security:
    uv run poe security-audit
    uv run poe audit

# === CI/CD Сценарии ===

# Этап статического анализа и безопасности для GitLab CI
[group('ci')]
ci-lint: check-static check-security

# Этап прогона тестового набора для GitLab CI с генерацией JUnit XML отчета
[group('ci')]
ci-test:
    uv run pytest -m "not live" --junitxml=junit.xml

# Полная локальная имитация пайплайна GitLab CI (линтинг, безопасность, тесты, валидация compose-манифеста)
[group('ci')]
ci: ci-lint ci-test
    docker compose config -q

# === Нагрузочное тестирование (Locust) ===

# Быстрый 15-секундный smoke-тест под нагрузкой
[group('load')]
load-smoke:
    uv run locust -f tests/load/locustfile.py --headless -u 5 -r 1 --run-time 15s --host http://127.0.0.1:8080 --only-summary

# Стресс-тестирование сервиса (60 пользователей, 60 секунд)
[group('load')]
load-stress:
    uv run locust -f tests/load/locustfile.py --headless -u 60 -r 10 --run-time 60s --host http://127.0.0.1:8080 --only-summary

# Формирование подробного HTML-отчета и CSV-метрик нагрузочного тестирования (5 минут)
[group('load')]
load-report:
    mkdir -p reports
    uv run locust -f tests/load/locustfile.py --headless -u 100 -r 10 --run-time 5m --host http://127.0.0.1:8080 --html reports/locust_report.html --csv reports/locust_data --only-summary

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
