"""Иерархия доменных исключений проекта.

Обеспечивает строгое разделение типов ошибок между доменной логикой,
валидацией данных, обучением моделей и инфраструктурными зависимостями,
сохраняя 100% совместимость со стандартными исключениями Python (ValueError, RuntimeError).
"""

from typing import Any


class ProjectError(Exception):
    """Базовое исключение для всех доменных ошибок проекта."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DomainValidationError(ProjectError, ValueError):
    """Ошибка валидации доменных данных (структура диалога, сообщения, участники)."""


class ModelArtifactError(ProjectError, RuntimeError):
    """Отсутствие или повреждение артефактов модели (веса, токенизатор, ONNX)."""


class InfrastructureError(ProjectError, RuntimeError):
    """Базовое исключение при сбоях внешних сервисов и хранилищ."""


class TritonServerError(InfrastructureError):
    """Сбой взаимодействия с Triton Inference Server (таймаут, отказ KServe v2)."""


class TrackerError(ProjectError, RuntimeError):
    """Сбой управления жизненным циклом экспериментов в MLflow."""


__all__ = [
    "DomainValidationError",
    "InfrastructureError",
    "ModelArtifactError",
    "ProjectError",
    "TrackerError",
    "TritonServerError",
]
