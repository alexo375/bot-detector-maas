import logging
import socket
from typing import Any

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from project.core.config import config

logger = logging.getLogger(__name__)


class ClickHouseConnectionFactory:
    """Фабрика создания подключений к ClickHouse DWH с автоматическим DNS-разрешением."""

    @staticmethod
    def resolve_host(host: str) -> str:
        """Проверяет доступность хоста по DNS; при ошибке возвращает `127.0.0.1`."""
        if host in ("localhost", "127.0.0.1"):
            return host
        try:
            socket.gethostbyname(host)
            return host
        except Exception:  # noqa: BLE001  # Resilient fallback
            logger.debug("Хост '%s' недоступен по DNS, переключаемся на 127.0.0.1", host)
            return "127.0.0.1"

    @classmethod
    def create(
        cls,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
        connect_timeout: int = 5,
        send_receive_timeout: int = 30,
        **kwargs: Any,
    ) -> Client:
        """Создает типизированный клиент `clickhouse-connect`."""
        raw_host = host or config.clickhouse.host
        resolved_host = cls.resolve_host(raw_host)
        ch_port = port if port is not None else config.clickhouse.port
        ch_database = database or config.clickhouse.database
        ch_user = username or config.clickhouse.user
        ch_pass = password if password is not None else config.clickhouse.password

        return clickhouse_connect.get_client(
            host=resolved_host,
            port=ch_port,
            database=ch_database,
            username=ch_user,
            password=ch_pass,
            connect_timeout=connect_timeout,
            send_receive_timeout=send_receive_timeout,
            **kwargs,
        )


__all__ = ["ClickHouseConnectionFactory"]
