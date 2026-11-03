set shell := ["bash", "-euo", "pipefail", "-c"]

# Показать список доступных сценариев
default:
    @just --list

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
