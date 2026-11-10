"""Тесты модульной архитектуры ядра (core) и инфраструктурных адаптеров (infrastructure)."""

import numpy as np
import pytest

from project.core.config import AppConfig, config
from project.core.paths import ProjectPaths, paths
from project.core.utils import DialogueHasher, LogitTransformer
from project.infrastructure.clickhouse.audit import ChatLogRecord
from project.infrastructure.clickhouse.audit.consumer import ClickHouseLogConsumer
from project.infrastructure.rabbitmq import AuditLogPublisher
from project.infrastructure.redis import DialogueCache


def test_core_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверка загрузки настроек конфигурации по умолчанию."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = AppConfig()
    assert cfg.mlflow_tracking_uri == "http://127.0.0.1:5000"
    assert cfg.redis.host == "localhost"
    assert cfg.redis.port == 6379
    assert cfg.clickhouse.port == 8123
    assert cfg.rabbitmq.url.startswith("amqp://")
    assert config is not None


def test_core_paths_structure():
    """Проверка корректности путей проекта."""
    p = ProjectPaths()
    assert p.root.is_dir()
    assert p.data.name == "data"
    assert p.deploy.name == "deploy"
    assert paths.root == p.root


def test_compute_dialogue_hash():
    """Проверка генерации детерминированного хеша диалога."""
    msgs1 = [{"participant_index": 0, "text": "Привет"}, {"participant_index": 1, "text": "Привет!"}]
    msgs2 = [{"participant_index": 0, "text": "Привет"}, {"participant_index": 1, "text": "Привет!"}]
    msgs3 = [{"participant_index": 0, "text": "Пока"}, {"participant_index": 1, "text": "До встречи!"}]

    assert DialogueHasher.compute(msgs1) == DialogueHasher.compute(msgs2)
    assert DialogueHasher.compute(msgs1) != DialogueHasher.compute(msgs3)


def test_redis_cache_disabled_when_disconnected():
    """Проверка отключения кэша при недоступности Redis во избежание утечек памяти."""
    cache = DialogueCache(redis_url="redis://invalid-host-for-testing:9999/0")
    assert not cache.is_connected

    key = "dialog_hash_test_123"
    val = {"verdicts": [{"bot_probability": 0.88}]}

    cache.set(key, val, ttl_seconds=60)
    assert cache.get(key) is None
    cache.delete(key)
    assert cache.get(key) is None


@pytest.mark.asyncio
async def test_audit_log_publisher_infrastructure_fallback():
    """Проверка паблишера RabbitMQ при недоступном сервере."""
    publisher = AuditLogPublisher(rabbitmq_url="amqp://invalid-host-for-testing:5672/")
    connected = await publisher.connect()
    assert not connected
    assert not publisher.is_connected

    # Публикация должна безопасно возвращать False
    success = await publisher.publish_event({"test": "data"})
    assert success is False
    await publisher.close()


def test_clickhouse_infrastructure_consumer_columns():
    """Проверка соответствия колонок схемы ClickHouse."""
    consumer = ClickHouseLogConsumer(ch_host="invalid-host")
    assert consumer.batch_size == 50
    cols = ChatLogRecord.column_names()
    assert "dialog_id" in cols
    assert "messages_json" in cols
    assert "latency_ms" in cols


def test_triton_prob_to_logit():
    """Проверка корректности формулы логистического преобразования вероятностей."""
    p = np.array([0.5, 0.0, 1.0])
    logits = LogitTransformer.transform(p)
    assert np.isclose(logits[0], 0.0, atol=1e-4)
    assert logits[1] < -10.0
    assert logits[2] > 10.0


def test_dialogue_cache_connected_task_methods() -> None:
    """Проверка методов set_task и get_task при активном Redis-клиенте."""
    from unittest.mock import MagicMock

    mock_redis = MagicMock()
    mock_redis.get.return_value = '{"status": "completed", "result": 123}'
    cache = DialogueCache(client=mock_redis)

    cache.set_task("task_001", {"status": "pending"})
    mock_redis.setex.assert_called_once()
    assert "async_task:task_001" in mock_redis.setex.call_args[0]

    record = cache.get_task("task_001")
    assert record is not None
    assert record["status"] == "completed"


def test_triton_is_healthy_trips_circuit_breaker_on_false() -> None:
    """Проверка, что возврат False из transport.is_healthy учитывается как сбой CircuitBreaker."""
    from unittest.mock import MagicMock

    from project.infrastructure.triton import CircuitBreaker, TritonEnsembleClient

    mock_transport = MagicMock()
    mock_transport.is_healthy.return_value = False
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
    client = TritonEnsembleClient(transport=mock_transport, circuit_breaker=cb)

    assert client.is_healthy() is False
    assert cb.consecutive_failures == 1
    assert cb.state.value == "CLOSED"

    assert client.is_healthy() is False
    assert cb.state.value == "OPEN"
    # Следующий вызов fail-fast не обращается к транспорту
    assert client.is_healthy() is False
    assert mock_transport.is_healthy.call_count == 2
