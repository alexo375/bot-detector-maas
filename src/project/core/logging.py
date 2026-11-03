"""Централизованная система конфигурации и получения логгеров проекта.

Поддерживает текстовый и структурированный JSON формат.
"""

import datetime as dt
import json
import logging
import os
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

DEFAULT_LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class StructuredJsonFormatter(logging.Formatter):
    """Форматтер, сериализующий log records в компактный однострочный JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Форматирует запись лога в JSON объект."""
        log_data: dict[str, Any] = {
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        standard_record_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
        }

        extra_fields = {k: v for k, v in record.__dict__.items() if k not in standard_record_attrs}
        if extra_fields:
            log_data["context"] = extra_fields

        return json.dumps(log_data, ensure_ascii=False, default=str)


class LoggingConfigurator:
    """Глобальный конфигуратор формата и уровня логирования проекта."""

    @staticmethod
    def configure(
        level: str | int | None = None,
        log_format: str = DEFAULT_LOG_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
        use_json: bool | None = None,
    ) -> None:
        """Инициализирует глобальную конфигурацию логирования проекта."""
        if level is None:
            level_str = os.getenv("LOG_LEVEL", "INFO").upper()
            level = getattr(logging, level_str, logging.INFO)

        if use_json is None:
            use_json = os.getenv("LOG_FORMAT", "text").lower() == "json"

        handler = logging.StreamHandler(sys.stdout)
        formatter: logging.Formatter = (
            StructuredJsonFormatter() if use_json else logging.Formatter(fmt=log_format, datefmt=date_format)
        )
        handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.handlers.clear()
        root_logger.addHandler(handler)


class ProjectLogger(logging.LoggerAdapter[logging.Logger]):
    """Обертка над логгером с возможностью расширения контекста."""

    def __init__(self, logger: logging.Logger, extra: dict[str, Any] | None = None) -> None:
        super().__init__(logger, extra or {})

    @classmethod
    def get_logger(cls, name: str | None = None) -> "ProjectLogger":
        """Возвращает типизированный логгер с единой конфигурацией проекта."""
        return cls(logging.getLogger(name or "project"))

    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> tuple[Any, MutableMapping[str, Any]]:
        """Обрабатывает сообщение перед отправкой в логгер, добавляя базовый контекст extra."""
        extra: dict[str, Any] = dict(self.extra) if self.extra else {}
        if "extra" in kwargs and isinstance(kwargs["extra"], Mapping):
            extra.update(kwargs["extra"])
        if extra:
            kwargs["extra"] = extra
        return msg, kwargs


__all__ = [
    "DEFAULT_DATE_FORMAT",
    "DEFAULT_LOG_FORMAT",
    "LoggingConfigurator",
    "ProjectLogger",
    "StructuredJsonFormatter",
]
