"""Модульные тесты глобальных обработчиков доменных ошибок FastAPI (на изолированном экземпляре)."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from project.api.app import create_app
from project.core.exceptions import (
    DomainValidationError,
    InfrastructureError,
    ModelArtifactError,
    ProjectError,
)


def _create_isolated_client() -> tuple[FastAPI, TestClient]:
    test_app = create_app()
    return test_app, TestClient(test_app, raise_server_exceptions=False)


def test_domain_validation_handler():
    """Проверка возврата HTTP 422 при DomainValidationError."""
    test_app, client = _create_isolated_client()

    @test_app.get("/test/domain-error")
    def trigger_domain_error():
        raise DomainValidationError("Невалидные данные диалога", details={"field": "messages"})

    resp = client.get("/test/domain-error")
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"] == "domain_validation_error"
    assert "Невалидные данные диалога" in data["message"]
    assert data["details"] == {"field": "messages"}


def test_model_artifact_handler():
    """Проверка возврата HTTP 503 при ModelArtifactError."""
    test_app, client = _create_isolated_client()

    @test_app.get("/test/model-error")
    def trigger_model_error():
        raise ModelArtifactError("Веса модели ONNX не найдены в репозитории")

    resp = client.get("/test/model-error")
    assert resp.status_code == 503
    data = resp.json()
    assert data["error"] == "model_unavailable"
    assert "Веса модели ONNX не найдены" in data["message"]


def test_infrastructure_handler():
    """Проверка возврата HTTP 503 при InfrastructureError."""
    test_app, client = _create_isolated_client()

    @test_app.get("/test/infra-error")
    def trigger_infra_error():
        raise InfrastructureError("Сбой соединения с RabbitMQ брокером")

    resp = client.get("/test/infra-error")
    assert resp.status_code == 503
    data = resp.json()
    assert data["error"] == "infrastructure_unavailable"
    assert "Сбой соединения" in data["message"]


def test_project_base_error_handler():
    """Проверка возврата HTTP 500 при общем базовом ProjectError."""
    test_app, client = _create_isolated_client()

    @test_app.get("/test/project-error")
    def trigger_project_error():
        raise ProjectError("Непредвиденная доменная ошибка")

    resp = client.get("/test/project-error")
    assert resp.status_code == 500
    data = resp.json()
    assert data["error"] == "internal_error"
    assert "Непредвиденная доменная ошибка" in data["message"]
