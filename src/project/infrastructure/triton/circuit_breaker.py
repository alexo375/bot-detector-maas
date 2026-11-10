"""Реализация паттерна Circuit Breaker для изоляции сбоев Triton Inference Server.

Предотвращает каскадные сбои (cascade failure) и лавинообразные повторные запросы
(retry storm) при перезапуске или временной недоступности Triton.
"""

import threading
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any, TypeVar

from project.core.exceptions import TritonServerError
from project.core.logging import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

T = TypeVar("T")


class CircuitState(StrEnum):
    """Состояния автомата защиты (Circuit Breaker)."""

    CLOSED = "CLOSED"  # Нормальный рабочий режим: запросы разрешены
    OPEN = "OPEN"  # Аварийный режим: запросы мгновенно отсекаются (fail-fast)
    HALF_OPEN = "HALF_OPEN"  # Пробный режим: разрешается один тестовый запрос


class CircuitBreaker:
    """Управление состоянием доступности внешнего сервиса с автоматическим восстановлением."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 15.0,
        backoff_factor: float = 1.5,
        max_recovery_timeout: float = 120.0,
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.initial_recovery_timeout = recovery_timeout
        self.current_recovery_timeout = recovery_timeout
        self.backoff_factor = backoff_factor
        self.max_recovery_timeout = max_recovery_timeout

        self._lock = threading.RLock()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time: float = 0.0
        self._trial_in_progress = False

    def _refresh_state_locked(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.current_recovery_timeout:
                logger.info(
                    "CircuitBreaker: таймаут восстановления %.1fс истёк, переход в HALF_OPEN для проверки связи",
                    self.current_recovery_timeout,
                )
                self._state = CircuitState.HALF_OPEN
                self._trial_in_progress = False
        return self._state

    @property
    def state(self) -> CircuitState:
        """Динамическое вычисление текущего состояния защиты с учётом таймаута."""
        with self._lock:
            return self._refresh_state_locked()

    @property
    def consecutive_failures(self) -> int:
        """Количество последовательных сбоев."""
        with self._lock:
            return self._consecutive_failures

    def record_success(self) -> None:
        """Фиксация успешного ответа сервиса: сброс счётчиков и возврат в CLOSED."""
        with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info("CircuitBreaker: связь восстановлена, переход в CLOSED")
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self.current_recovery_timeout = self.initial_recovery_timeout
            self._trial_in_progress = False

    def record_failure(self, exc: Exception) -> None:
        """Фиксация ошибки сервиса и переход в OPEN при превышении порога сбоев."""
        with self._lock:
            self._consecutive_failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Пробный запрос не удался -> экспоненциальный откат таймаута
                self.current_recovery_timeout = min(
                    self.current_recovery_timeout * self.backoff_factor,
                    self.max_recovery_timeout,
                )
                self._state = CircuitState.OPEN
                self._trial_in_progress = False
                logger.warning(
                    "CircuitBreaker: пробный запрос провален (%s). Возврат в OPEN на %.1fс",
                    exc,
                    self.current_recovery_timeout,
                )
            elif self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.error(
                    "CircuitBreaker: зафиксировано %d сбоев подряд (%s). Размыкание цепи -> OPEN на %.1fс (fail-fast)",
                    self._consecutive_failures,
                    exc,
                    self.current_recovery_timeout,
                )

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Исполнение вызова с защитой Circuit Breaker."""
        claimed_trial = False
        with self._lock:
            current_state = self._refresh_state_locked()
            if current_state == CircuitState.OPEN:
                raise TritonServerError(
                    f"Triton Inference Server временно недоступен (Circuit Breaker OPEN). "
                    f"Повторите попытку через {self.current_recovery_timeout:.1f}с."
                )

            if current_state == CircuitState.HALF_OPEN:
                if self._trial_in_progress:
                    raise TritonServerError("Проверка доступности Triton уже выполняется (HALF_OPEN trial in progress)")
                self._trial_in_progress = True
                claimed_trial = True

        settled = False
        try:
            result = func(*args, **kwargs)
            self.record_success()
            settled = True
            return result
        except Exception as exc:
            self.record_failure(exc)
            settled = True
            raise
        finally:
            if claimed_trial and not settled:
                with self._lock:
                    self._trial_in_progress = False


__all__ = ["CircuitBreaker", "CircuitState"]
