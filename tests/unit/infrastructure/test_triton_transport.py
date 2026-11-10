"""Модульные тесты для сетевых транспортов Triton (HTTP и gRPC) и фабрики транспортов."""

from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from tritonclient.utils import InferenceServerException

from project.core.exceptions import TritonServerError
from project.infrastructure.triton.circuit_breaker import CircuitBreaker, CircuitState
from project.infrastructure.triton.client import TritonEnsembleClient
from project.infrastructure.triton.transport.factory import TritonTransportFactory
from project.infrastructure.triton.transport.grpc import GrpcTritonTransport
from project.infrastructure.triton.transport.http import HttpTritonTransport


def test_http_transport_url_normalization() -> None:
    """HTTP транспорт корректно нормализует url без префикса схемы."""
    transport = HttpTritonTransport(base_url="my-triton:8000/")
    assert transport.base_url == "http://my-triton:8000"


def test_http_transport_infer_success() -> None:
    """HTTP транспорт успешно отправляет тензоры и парсит выходы."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "outputs": [
            {"name": "probabilities", "shape": [2, 2], "data": [0.8, 0.2, 0.1, 0.9]},
            {"name": "cb_probabilities", "shape": [2, 2], "data": [0.75, 0.25, 0.15, 0.85]},
            {"name": "lgb_probabilities", "shape": [2, 2], "data": [0.70, 0.30, 0.20, 0.80]},
            {"name": "xgb_probabilities", "shape": [2, 2], "data": [0.85, 0.15, 0.05, 0.95]},
        ]
    }
    mock_client.post.return_value = mock_resp

    transport = HttpTritonTransport(base_url="http://test:8000", client=mock_client)
    features = np.ones((2, 10), dtype=np.float32)

    p_meta, p_cb, p_lgb, p_xgb = transport.infer(features)

    assert p_meta.shape == (2,)
    assert np.isclose(p_meta[0], 0.2)
    assert np.isclose(p_meta[1], 0.9)
    assert np.isclose(p_cb[1], 0.85)
    assert np.isclose(p_lgb[1], 0.80)
    assert np.isclose(p_xgb[1], 0.95)
    mock_resp.raise_for_status.assert_called_once()


def test_http_transport_infer_error_handling() -> None:
    """HTTP транспорт транслирует сетевые сбои в TritonServerError."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = httpx.ConnectTimeout("Triton timeout")

    transport = HttpTritonTransport(base_url="http://test:8000", client=mock_client)
    with pytest.raises(TritonServerError, match="Сбой HTTP-запроса к Triton"):
        transport.infer(np.ones((1, 5), dtype=np.float32))


def test_http_transport_is_healthy() -> None:
    """HTTP транспорт проверяет эндпоинт здоровья."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client.get.return_value = mock_resp

    transport = HttpTritonTransport(base_url="http://test:8000", client=mock_client)
    assert transport.is_healthy() is True

    mock_client.get.side_effect = httpx.HTTPError("down")
    assert transport.is_healthy() is False


def test_http_transport_close() -> None:
    """HTTP транспорт корректно закрывает внутренний httpx.Client."""
    mock_client = MagicMock(spec=httpx.Client)
    transport = HttpTritonTransport(base_url="http://test:8000", client=mock_client)
    transport.close()
    mock_client.close.assert_called_once()


def test_grpc_transport_infer_success() -> None:
    """gRPC транспорт формирует бинарные тензоры и парсит np.ndarray выходы."""
    mock_grpc = MagicMock()
    mock_response = MagicMock()

    def mock_as_numpy(name: str):
        arrays = {
            "probabilities": np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
            "cb_probabilities": np.array([[0.75, 0.25], [0.15, 0.85]], dtype=np.float32),
            "lgb_probabilities": np.array([[0.7, 0.3], [0.2, 0.8]], dtype=np.float32),
            "xgb_probabilities": np.array([[0.85, 0.15], [0.05, 0.95]], dtype=np.float32),
        }
        return arrays.get(name)

    mock_response.as_numpy.side_effect = mock_as_numpy
    mock_grpc.infer.return_value = mock_response

    transport = GrpcTritonTransport(server_url="localhost:8001", grpc_client=mock_grpc)
    features = np.ones((2, 10), dtype=np.float32)

    p_meta, p_cb, p_lgb, p_xgb = transport.infer(features)

    assert p_meta.shape == (2,)
    assert np.isclose(p_meta[0], 0.2)
    assert np.isclose(p_meta[1], 0.9)
    assert np.isclose(p_cb[1], 0.85)
    assert np.isclose(p_lgb[1], 0.80)
    assert np.isclose(p_xgb[1], 0.95)
    mock_grpc.infer.assert_called_once()


def test_grpc_transport_infer_error_handling() -> None:
    """gRPC транспорт транслирует ошибки сервера в TritonServerError."""
    mock_grpc = MagicMock()
    mock_grpc.infer.side_effect = InferenceServerException("gRPC service unavailable")

    transport = GrpcTritonTransport(server_url="localhost:8001", grpc_client=mock_grpc)
    with pytest.raises(TritonServerError, match="Сбой gRPC-запроса к Triton"):
        transport.infer(np.ones((1, 5), dtype=np.float32))


def test_grpc_transport_is_healthy() -> None:
    """gRPC транспорт проверяет готовность через is_server_ready."""
    mock_grpc = MagicMock()
    mock_grpc.is_server_ready.return_value = True

    transport = GrpcTritonTransport(server_url="localhost:8001", grpc_client=mock_grpc)
    assert transport.is_healthy() is True

    mock_grpc.is_server_ready.side_effect = InferenceServerException("failed")
    assert transport.is_healthy() is False


def test_grpc_transport_close() -> None:
    """gRPC транспорт корректно закрывает соединение."""
    mock_grpc = MagicMock()
    transport = GrpcTritonTransport(server_url="localhost:8001", grpc_client=mock_grpc)
    transport.close()
    mock_grpc.close.assert_called_once()


def test_create_triton_transport_factory() -> None:
    """Фабрика TritonTransportFactory.create корректно инстанциирует транспорты."""
    # Явный протокол
    http_t = TritonTransportFactory.create(protocol="http")
    assert isinstance(http_t, HttpTritonTransport)

    grpc_t = TritonTransportFactory.create(protocol="grpc")
    assert isinstance(grpc_t, GrpcTritonTransport)

    # Авто-детект по URL
    auto_grpc = TritonTransportFactory.create(url="triton-server:8001")
    assert isinstance(auto_grpc, GrpcTritonTransport)

    auto_grpc_schema = TritonTransportFactory.create(url="grpc://triton-server:9000")
    assert isinstance(auto_grpc_schema, GrpcTritonTransport)

    auto_http = TritonTransportFactory.create(url="http://triton-server:8000")
    assert isinstance(auto_http, HttpTritonTransport)


def test_create_triton_transport_fails_fast_on_missing_grpc(monkeypatch) -> None:
    """При отсутствии зависимостей gRPC фабрика падает явно без отката на HTTP."""
    import sys

    # Имитируем сбой импорта grpc
    monkeypatch.setitem(sys.modules, "project.infrastructure.triton.transport.grpc", None)

    with pytest.raises(ImportError):
        TritonTransportFactory.create(protocol="grpc")


def test_triton_client_with_custom_transport() -> None:
    """TritonEnsembleClient принимает пользовательский транспорт и защищает его Circuit Breaker."""
    mock_transport = MagicMock(spec=GrpcTritonTransport)
    mock_transport.infer.side_effect = TritonServerError("Server down")

    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)
    client = TritonEnsembleClient(circuit_breaker=cb, transport=mock_transport)

    with pytest.raises(TritonServerError):
        client.infer(np.ones((1, 5), dtype=np.float32))

    with pytest.raises(TritonServerError):
        client.infer(np.ones((1, 5), dtype=np.float32))

    assert cb.state == CircuitState.OPEN
    # Проверка вызова close
    client.close()
    mock_transport.close.assert_called_once()


def test_triton_transport_http_client_and_scheme_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Явно переданный HTTP client или URL со схемой http:// не перетирается протоколом gRPC из конфига."""
    from project.core.config import config

    monkeypatch.setattr(config.triton, "protocol", "grpc")
    mock_http = MagicMock(spec=httpx.Client)

    t_client = TritonTransportFactory.create(client=mock_http)
    assert isinstance(t_client, HttpTritonTransport)

    t_scheme = TritonTransportFactory.create(url="http://triton:8001")
    assert isinstance(t_scheme, HttpTritonTransport)


def test_triton_transport_checks_model_readiness() -> None:
    """Оба транспорта требуют готовности как сервера, так и ансамбля bot_ensemble."""
    mock_http = MagicMock(spec=httpx.Client)
    resp_server_ok = MagicMock(status_code=200)
    resp_model_not_ready = MagicMock(status_code=503)
    mock_http.get.side_effect = [resp_server_ok, resp_model_not_ready]

    http_t = HttpTritonTransport(base_url="http://test:8000", client=mock_http)
    assert http_t.is_healthy() is False

    mock_grpc = MagicMock()
    mock_grpc.is_server_ready.return_value = True
    mock_grpc.is_model_ready.return_value = False
    grpc_t = GrpcTritonTransport(server_url="localhost:8001", grpc_client=mock_grpc)
    assert grpc_t.is_healthy() is False
