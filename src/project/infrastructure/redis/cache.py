import json
import threading
import time
from typing import Any, cast

import redis

from project.core.config import config
from project.core.logging import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

_RECONNECT_COOLDOWN_SEC = 10.0


class DialogueCache:
    """Клиент распределенного кэша Redis."""

    def __init__(
        self,
        redis_url: str | None = None,
        client: redis.Redis | None = None,
        default_ttl_sec: int | None = None,
    ) -> None:
        """Инициализация подключения к Redis или инъекция готового клиента."""
        self.redis_url = redis_url or config.redis.url
        self.default_ttl = default_ttl_sec if default_ttl_sec is not None else config.redis.cache_default_ttl_sec
        self.redis_client = client
        self._owns_client = client is None
        self._lock = threading.Lock()
        self._last_reconnect_attempt = 0.0

        if self.redis_client is None:
            self.reconnect(force=True)

    def reconnect(self, force: bool = False) -> bool:
        """Попытка подключения или переподключения к Redis с ограничением частоты повторов."""
        if not self._owns_client:
            return self.redis_client is not None

        now = time.monotonic()
        with self._lock:
            if self.redis_client is not None:
                return True
            if not force and (now - self._last_reconnect_attempt) < _RECONNECT_COOLDOWN_SEC:
                return False
            self._last_reconnect_attempt = now
            try:
                c = redis.Redis.from_url(self.redis_url, socket_connect_timeout=0.5, decode_responses=True)
                c.ping()
                self.redis_client = c
                logger.info("Connected to Redis at %s", self.redis_url)
                return True
            except Exception as e:  # noqa: BLE001  # Resilient fallback
                logger.warning("Redis is not available at %s (%s). Caching is disabled.", self.redis_url, e)
                return False

    def _invalidate_client_on_error(self, exc: BaseException) -> None:
        """Сбрасывает собственный Redis-клиент при сетевой ошибке для последующего переподключения."""
        if (
            self._owns_client
            and isinstance(self.redis_client, redis.Redis)
            and isinstance(exc, (redis.ConnectionError, redis.TimeoutError))
        ):
            with self._lock:
                client_ref = self.redis_client
                self.redis_client = None
            if client_ref is not None and hasattr(client_ref, "close"):
                try:
                    client_ref.close()
                except Exception:
                    logger.debug("Ignoring close error after Redis disconnect", exc_info=True)

    def _get_client(self) -> redis.Redis | None:
        if self.redis_client is None and self._owns_client:
            self.reconnect(force=False)
        return self.redis_client

    @property
    def is_connected(self) -> bool:
        """Флаг доступности подключения к серверу Redis."""
        return self._get_client() is not None

    def close(self) -> None:
        """Закрытие соединения с сервером Redis."""
        if self.redis_client is not None and hasattr(self.redis_client, "close"):
            self.redis_client.close()
        self.redis_client = None

    @staticmethod
    def _serialize_payload(value: Any) -> str:
        dict_val = value.to_dict() if hasattr(value, "to_dict") else value
        return json.dumps(dict_val, ensure_ascii=False)

    def delete(self, dialogue_hash: str) -> None:
        """Удаление закэшированного вердикта по хэшу диалога."""
        client = self._get_client()
        if client is None:
            return
        key = f"dialog_score:{dialogue_hash}"
        try:
            client.delete(key)
        except Exception as e:  # noqa: BLE001  # Resilient fallback
            self._invalidate_client_on_error(e)
            logger.warning("Redis delete failed: %s", e)

    def get(self, dialogue_hash: str) -> dict[str, Any] | None:
        """Получение закэшированного вердикта по хэшу диалога."""
        client = self._get_client()
        if client is None:
            return None
        key = f"dialog_score:{dialogue_hash}"
        try:
            val = client.get(key)
            if val is not None:
                return cast("dict[str, Any]", json.loads(val))
        except (redis.RedisError, json.JSONDecodeError) as e:
            self._invalidate_client_on_error(e)
            logger.warning("Redis get failed: %s", e)
        return None

    def mget(self, dialogue_hashes: list[str]) -> list[dict[str, Any] | None]:
        """Пакетное получение закэшированных вердиктов (1 сетевой round-trip)."""
        client = self._get_client()
        if client is None or not dialogue_hashes:
            return [None] * len(dialogue_hashes)
        keys = [f"dialog_score:{h}" for h in dialogue_hashes]
        try:
            raw_vals = client.mget(keys)
            results: list[dict[str, Any] | None] = []
            for val in raw_vals:
                if val is not None:
                    try:
                        results.append(cast("dict[str, Any]", json.loads(val)))
                    except (json.JSONDecodeError, TypeError):
                        results.append(None)
                else:
                    results.append(None)
            return results
        except redis.RedisError as e:
            self._invalidate_client_on_error(e)
            logger.warning("Redis mget failed: %s", e)
            return [None] * len(dialogue_hashes)

    def set(
        self,
        dialogue_hash: str,
        verdict: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        """Сохранение вердикта инференса в кэш с заданным временем жизни."""
        client = self._get_client()
        if client is None:
            return
        key = f"dialog_score:{dialogue_hash}"
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        payload = self._serialize_payload(verdict)
        try:
            client.setex(key, ttl, payload)
        except Exception as e:  # noqa: BLE001  # Resilient fallback
            self._invalidate_client_on_error(e)
            logger.warning("Redis set failed: %s", e)

    def mset(
        self,
        items: list[tuple[str, Any]],
        ttl_seconds: int | None = None,
    ) -> None:
        """Пакетное сохранение вердиктов инференса через pipeline (1 сетевой round-trip)."""
        client = self._get_client()
        if client is None or not items:
            return
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        try:
            pipe = client.pipeline(transaction=False)
            for d_hash, verdict in items:
                key = f"dialog_score:{d_hash}"
                pipe.setex(key, ttl, self._serialize_payload(verdict))
            pipe.execute()
        except Exception as e:  # noqa: BLE001  # Resilient fallback
            self._invalidate_client_on_error(e)
            logger.warning("Redis mset pipeline failed: %s", e)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Получение текущего статуса и результата асинхронной задачи."""
        client = self._get_client()
        if client is None:
            return None
        key = f"async_task:{task_id}"
        try:
            val = client.get(key)
            if val is not None:
                return cast("dict[str, Any]", json.loads(val))
        except (redis.RedisError, json.JSONDecodeError) as e:
            self._invalidate_client_on_error(e)
            logger.warning("Redis get_task failed: %s", e)
        return None

    def set_task(self, task_id: str, task_data: Any, ttl_sec: int | None = None) -> bool:
        """Сохранение состояния асинхронной задачи в хранилище."""
        client = self._get_client()
        if client is None:
            return False
        key = f"async_task:{task_id}"
        effective_ttl = ttl_sec if ttl_sec is not None else config.redis.task_default_ttl_sec
        payload = self._serialize_payload(task_data)
        try:
            client.setex(key, effective_ttl, payload)
            return True
        except Exception as e:  # noqa: BLE001  # Resilient fallback
            self._invalidate_client_on_error(e)
            logger.warning("Redis set_task failed: %s", e)
            return False

    def delete_task(self, task_id: str) -> None:
        """Удаление состояния асинхронной задачи из хранилища."""
        client = self._get_client()
        if client is None:
            return
        try:
            client.delete(f"async_task:{task_id}")
        except Exception as e:  # noqa: BLE001  # Resilient fallback
            self._invalidate_client_on_error(e)
            logger.warning("Redis delete_task failed: %s", e)
