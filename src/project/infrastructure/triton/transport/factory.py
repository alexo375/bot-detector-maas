"""Фабрика создания сетевого транспорта Triton (HTTP / gRPC)."""

from typing import Any

import httpx

from project.core.config import config
from project.core.logging import ProjectLogger
from project.infrastructure.triton.transport.base import BaseTritonTransport
from project.infrastructure.triton.transport.http import HttpTritonTransport

logger = ProjectLogger.get_logger(__name__)


class TritonTransportFactory:
    """Фабрика создания HTTP или gRPC транспорта для Triton Inference Server."""

    @staticmethod
    def _resolve_protocol(
        protocol: str | None,
        url: str | None,
        client: httpx.Client | None,
        grpc_client: Any | None,
    ) -> str:
        if protocol:
            return protocol.lower()
        if url:
            if url.startswith("grpc://") or (not url.startswith(("http://", "https://")) and ":8001" in url):
                return "grpc"
            if url.startswith(("http://", "https://")) or ":8000" in url:
                return "http"
        if client is not None:
            return "http"
        if grpc_client is not None:
            return "grpc"
        return config.triton.protocol.lower()

    @classmethod
    def create(
        cls,
        protocol: str | None = None,
        url: str | None = None,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
        grpc_client: Any | None = None,
    ) -> BaseTritonTransport:
        """Создает запрошенный транспорт Triton (HTTP или gRPC) строго без тихого отката."""
        chosen_proto = cls._resolve_protocol(protocol, url, client, grpc_client)
        if chosen_proto == "grpc":
            from project.infrastructure.triton.transport.grpc import GrpcTritonTransport

            server_url = url or config.triton.grpc_url
            logger.info("Инициализирован Triton gRPC транспорт (%s)", server_url)
            return GrpcTritonTransport(server_url=server_url, grpc_client=grpc_client, timeout=timeout)
        if chosen_proto != "http":
            raise ValueError(f"Unsupported Triton transport protocol: {chosen_proto!r}")

        server_url = url or config.triton.url
        logger.info("Инициализирован Triton HTTP транспорт (%s)", server_url)
        return HttpTritonTransport(base_url=server_url, client=client, timeout=timeout)


__all__ = ["TritonTransportFactory"]
