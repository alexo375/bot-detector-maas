"""Модульные тесты для паттерна Circuit Breaker и защиты Triton клиента."""

from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from project.core.exceptions import TritonServerError
from project.infrastructure.triton.circuit_breaker import CircuitBreaker, CircuitState
from project.infrastructure.triton.client import TritonEnsembleClient


def test_circuit_breaker_normal_closed_execution():
    """В нормальном режиме (CLOSED) запросы выполняются штатно."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    mock_func = MagicMock(return_value="success_result")

    result = cb.execute(mock_func, 123, key="value")
    assert result == "success_result"
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 0
    mock_func.assert_called_once_with(123, key="value")


def test_circuit_breaker_opens_after_consecutive_failures():
    """При серии сбоев цепь размыкается (OPEN)."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    mock_func = MagicMock(side_effect=RuntimeError("Triton timeout"))

    for _ in range(3):
        with pytest.raises(RuntimeError):
            cb.execute(mock_func)

    assert cb.consecutive_failures == 3
    assert cb.state == CircuitState.OPEN


def test_circuit_breaker_open_fails_fast_without_calling_target():
    """В состоянии OPEN запросы мгновенно отсекаются без обращения к целевому сервису."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)
    mock_func = MagicMock(side_effect=RuntimeError("connection error"))

    # Доводим до OPEN
    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.execute(mock_func)

    assert cb.state == CircuitState.OPEN
    mock_func.reset_mock()

    # Попытка вызова в OPEN
    with pytest.raises(TritonServerError) as exc_info:
        cb.execute(mock_func)

    assert "Circuit Breaker OPEN" in str(exc_info.value)
    mock_func.assert_not_called()


def test_circuit_breaker_half_open_recovery_success(monkeypatch):
    """После истечения таймаута цепь переходит в HALF_OPEN и возвращается в CLOSED при успехе."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=2.0)
    mock_func = MagicMock(side_effect=RuntimeError("fail"))

    # Размыкаем цепь
    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.execute(mock_func)
    assert cb.state == CircuitState.OPEN

    # Сдвигаем время вперёд на 2.5 секунды
    original_time = cb._last_failure_time
    monkeypatch.setattr("time.monotonic", lambda: original_time + 2.5)

    assert str(cb.state) == str(CircuitState.HALF_OPEN)

    # Успешный пробный запрос
    mock_func.side_effect = None
    mock_func.return_value = "probe_ok"

    res = cb.execute(mock_func)
    assert res == "probe_ok"
    assert str(cb.state) == str(CircuitState.CLOSED)
    assert cb.consecutive_failures == 0


def test_circuit_breaker_half_open_failure_increases_timeout(monkeypatch):
    """Если пробный запрос в HALF_OPEN провалился, цепь возвращается в OPEN с увеличенным таймаутом."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=2.0, backoff_factor=2.0)
    mock_func = MagicMock(side_effect=RuntimeError("fail"))

    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.execute(mock_func)

    original_time = cb._last_failure_time
    monkeypatch.setattr("time.monotonic", lambda: original_time + 2.5)

    assert str(cb.state) == str(CircuitState.HALF_OPEN)

    # Пробный запрос падает
    with pytest.raises(RuntimeError):
        cb.execute(mock_func)

    assert str(cb.state) == str(CircuitState.OPEN)
    assert cb.current_recovery_timeout == 4.0  # 2.0 * backoff_factor


def test_triton_client_with_circuit_breaker_integration():
    """Интеграция TritonEnsembleClient с Circuit Breaker транслирует сетевые ошибки в TritonServerError."""
    mock_http_client = MagicMock()
    mock_http_client.post.side_effect = httpx.ConnectError("Connection refused by Triton")

    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=5.0)
    client = TritonEnsembleClient(client=mock_http_client, circuit_breaker=cb)

    dummy_features = np.zeros((2, 10), dtype=np.float32)

    # 1-я ошибка
    with pytest.raises(TritonServerError) as exc1:
        client.infer(dummy_features)
    assert "Сбой HTTP-запроса к Triton" in str(exc1.value)

    # 2-я ошибка -> размыкание цепи
    with pytest.raises(TritonServerError):
        client.infer(dummy_features)

    assert cb.state == CircuitState.OPEN

    # 3-й вызов -> мгновенный fail-fast без сетевого вызова
    mock_http_client.reset_mock()
    with pytest.raises(TritonServerError) as exc3:
        client.infer(dummy_features)
    assert "Circuit Breaker OPEN" in str(exc3.value)
    mock_http_client.post.assert_not_called()


def test_circuit_breaker_half_open_single_concurrent_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    """В состоянии HALF_OPEN одновременно выполняется строго один пробный запрос."""
    import threading

    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
    with pytest.raises(RuntimeError):
        cb.execute(MagicMock(side_effect=RuntimeError("boom")))

    t0 = cb._last_failure_time
    monkeypatch.setattr("time.monotonic", lambda: t0 + 2.0)

    started = threading.Event()
    release = threading.Event()
    results: list[str] = []

    def slow_probe() -> str:
        started.set()
        release.wait(timeout=2.0)
        return "ok"

    t1 = threading.Thread(target=lambda: results.append(cb.execute(slow_probe)))
    t1.start()
    assert started.wait(timeout=1.0)

    with pytest.raises(TritonServerError, match="trial in progress"):
        cb.execute(lambda: "second")

    release.set()
    t1.join(timeout=2.0)
    assert results == ["ok"]
    assert cb.state == CircuitState.CLOSED
