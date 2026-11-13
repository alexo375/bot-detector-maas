from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from project import __version__
from project.api.app import app
from project.api.schemas import AsyncTaskRecord, AsyncTaskStatus
from project.infrastructure.container import ServiceContainer
from project.infrastructure.rabbitmq.schemas.task import InferenceTask
from project.infrastructure.workers.inference_worker import InferenceWorker
from tests.fixtures.mock_fixtures import FakeRedisClient


@pytest.fixture(scope="module")
def client():
    services = ServiceContainer.get_instance()
    fake_redis = FakeRedisClient()
    if services.cache:
        services.cache.redis_client = fake_redis  # type: ignore[assignment]

    def mock_infer(features, shape=None):
        import numpy as np

        n = len(features) if hasattr(features, "__len__") else (shape[0] if shape else 2)
        probs = np.full(n, 0.2, dtype=np.float32)
        return probs, probs, probs, probs

    orig_infer = services.triton_client.infer
    orig_is_healthy = services.triton_client.is_healthy
    services.triton_client.infer = mock_infer  # type: ignore[method-assign]
    services.triton_client.is_healthy = lambda: True  # type: ignore[method-assign]

    orig_is_connected = services.publisher._is_connected
    orig_connection = services.publisher._connection
    orig_channel = services.publisher._channel
    orig_connect = services.publisher.connect
    orig_close = services.publisher.close
    orig_publish_task = services.publisher.publish_task

    mock_conn = MagicMock()
    mock_conn.is_closed = False
    services.publisher._is_connected = True
    services.publisher._connection = mock_conn
    services.publisher._channel = MagicMock()
    services.publisher.connect = AsyncMock(return_value=True)  # type: ignore[method-assign]
    services.publisher.close = AsyncMock()  # type: ignore[method-assign]
    test_worker = InferenceWorker(
        triton_client=services.triton_client,
        cache=services.cache,
    )

    async def mock_publish_task(task: InferenceTask) -> bool:
        eval_resp, _ = test_worker.process_task_sync(task)
        services.cache.set_task(
            task.task_id,
            AsyncTaskRecord(status=AsyncTaskStatus.COMPLETED, result=eval_resp).to_dict(),
        )
        return True

    services.publisher.publish_task = mock_publish_task  # type: ignore[method-assign]

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        services.triton_client.infer = orig_infer  # type: ignore[method-assign]
        services.triton_client.is_healthy = orig_is_healthy  # type: ignore[method-assign]
        services.publisher._is_connected = orig_is_connected
        services.publisher._connection = orig_connection
        services.publisher._channel = orig_channel
        services.publisher.connect = orig_connect  # type: ignore[method-assign]
        services.publisher.close = orig_close  # type: ignore[method-assign]
        services.publisher.publish_task = orig_publish_task  # type: ignore[method-assign]


def test_health_check(client):
    """Проверка эндпоинта /health."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    assert data["version"] == __version__


def test_liveness_and_readiness_probes(client):
    """Проверка Kubernetes liveness (/health/live) и readiness (/health/ready) проб."""
    resp_live = client.get("/health/live")
    assert resp_live.status_code == 200
    assert resp_live.json() == {"status": "alive"}

    resp_ready = client.get("/health/ready")
    assert resp_ready.status_code == 200
    data_ready = resp_ready.json()
    assert data_ready["status"] == "healthy"
    assert data_ready["models_loaded"] is True


def test_predict_sync_success(client):
    """Классификация диалога: возврат 202 при отсутствии в кэше и получение результата по task_id."""
    payload = {
        "dialogues": [
            {
                "dialog_id": "test-session-001",
                "messages": [
                    {"participant_index": 0, "text": "Привет! Ты бот или человек?"},
                    {"participant_index": 1, "text": "Привет! Человек конечно, а что?"},
                    {"participant_index": 0, "text": "А реши 45 на 3"},
                    {"participant_index": 1, "text": "135"},
                ],
            }
        ]
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 202
    task_id = response.json()["task_id"]

    resp_get = client.get(f"/predict/{task_id}")
    assert resp_get.status_code == 200
    data = resp_get.json()

    assert data["total"] == 1
    assert len(data["results"]) == 1
    item = data["results"][0]
    assert item["dialog_id"] == "test-session-001"
    assert item["dialogue_length"] == 4
    assert len(item["verdicts"]) == 2
    assert item["cached"] is False
    assert 0.0 <= item["verdicts"][0]["bot_probability"] <= 1.0
    assert 0.0 <= item["verdicts"][1]["bot_probability"] <= 1.0


def test_predict_cache_hit(client):
    """Проверка мгновенной отдачи из кэша (Redis / in-memory)."""
    payload = {
        "dialogues": [
            {
                "dialog_id": "cached-session-999",
                "messages": [
                    {"participant_index": 0, "text": "Какая погода сегодня?"},
                    {"participant_index": 1, "text": "Солнечно и тепло!"},
                ],
            }
        ]
    }

    # Первый запрос - промах кэша, постановка в очередь (202)
    resp1 = client.post("/predict", json=payload)
    assert resp1.status_code == 202
    task_id = resp1.json()["task_id"]

    # Завершение фоновой обработки
    resp_get = client.get(f"/predict/{task_id}")
    assert resp_get.status_code == 200
    assert resp_get.json()["results"][0]["cached"] is False

    # Второй запрос - 100% попадание в кэш (200 OK, O(1))
    resp2 = client.post("/predict", json=payload)
    assert resp2.status_code == 200
    assert resp2.json()["results"][0]["cached"] is True
    assert resp2.json()["batch_latency_ms"] < 15.0


def test_predict_batch_multiple_dialogues(client):
    """Проверка классификации пачки из нескольких диалогов в одном запросе."""
    payload = {
        "dialogues": [
            {
                "dialog_id": "batch-dialog-001",
                "messages": [
                    {"participant_index": 0, "text": "Первый диалог привет"},
                    {"participant_index": 1, "text": "Первый диалог ответ"},
                ],
            },
            {
                "dialog_id": "batch-dialog-002",
                "messages": [
                    {"participant_index": 0, "text": "Второй диалог вопрос"},
                    {"participant_index": 1, "text": "Второй диалог ответ"},
                ],
            },
        ]
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]

    resp_get = client.get(f"/predict/{task_id}")
    assert resp_get.status_code == 200
    data = resp_get.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["dialog_id"] == "batch-dialog-001"
    assert data["results"][1]["dialog_id"] == "batch-dialog-002"


def test_predict_pull_polling_model(client):
    """Проверка асинхронной Pull-модели (202 Accepted -> GET /predict/{task_id})."""
    payload = {
        "dialogues": [
            {
                "dialog_id": "pull-async-session-123",
                "messages": [
                    {"participant_index": 0, "text": "Привет"},
                    {"participant_index": 1, "text": "Привет"},
                ],
            }
        ]
    }

    # 1. Отправка в фоновую очередь
    resp_init = client.post("/predict", json=payload)
    assert resp_init.status_code == 202
    init_data = resp_init.json()
    task_id = init_data["task_id"]
    assert init_data["status"] == "processing"

    # 2. Опрос результата через Pull GET
    resp_get = client.get(f"/predict/{task_id}")
    assert resp_get.status_code == 200
    result_data = resp_get.json()
    assert result_data["total"] == 1
    assert result_data["results"][0]["dialog_id"] == "pull-async-session-123"
    assert len(result_data["results"][0]["verdicts"]) == 2


def test_predict_task_not_found_or_expired_404(client):
    """Проверка 404 при запросе несуществующей или истекшей по TTL задачи."""
    resp = client.get("/predict/non-existent-task-id-999")
    assert resp.status_code == 404
    data = resp.json()
    assert "not found or expired" in data["detail"]


def test_predict_push_webhook_model(client):
    """Проверка Push-модели (Webhook)."""
    payload = {
        "dialogues": [
            {
                "dialog_id": "webhook-session-456",
                "messages": [
                    {"participant_index": 0, "text": "Привет, проверка вебхука"},
                    {"participant_index": 1, "text": "Да, все работает"},
                ],
            }
        ],
        "webhook_url": "https://httpbin.org/post",
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 202
    data = resp.json()
    assert data["webhook_url"] == "https://httpbin.org/post"
    assert "task_id" in data


def test_predict_pairwise_probability_constraint(client):
    """Проверка соблюдения парного ограничения вероятностей P(Bot0) + P(Bot1) <= 1.0 через PairwisePostprocessor."""
    payload = {
        "dialogues": [
            {
                "dialog_id": "pairwise-session-check",
                "messages": [
                    {"participant_index": 0, "text": "привет"},
                    {"participant_index": 1, "text": "привет"},
                ],
            }
        ]
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]

    resp_get = client.get(f"/predict/{task_id}")
    assert resp_get.status_code == 200
    data = resp_get.json()
    item = data["results"][0]
    assert item["dialogue_length"] == 2
    p0 = item["verdicts"][0]["bot_probability"]
    p1 = item["verdicts"][1]["bot_probability"]
    assert 0.0 <= p0 <= 1.0
    assert 0.0 <= p1 <= 1.0
    # Физическое ограничение: сумма вероятностей двух участников диалога не превышает 1.0
    assert (p0 + p1) <= 1.0


def test_predict_empty_messages_validation_error(client):
    """Проверка 422 ошибки при пустом списке сообщений."""
    payload = {
        "dialogues": [
            {
                "dialog_id": "empty-session",
                "messages": [],
            }
        ]
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_prometheus_metrics_endpoint(client):
    """Проверка доступности и формата эндпоинта сбора метрик /metrics."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "bot_detector" in body or "http_requests" in body
    assert "bot_detector_cache_hits_total" in body
    assert "bot_detector_inference_latency_seconds" in body
    assert "bot_detector_prediction_drift_psi" in body
    assert "bot_detector_ensemble_disagreement_rate" in body


def test_predict_queue_unavailable_fail_fast(client):
    """Проверка 503 Service Unavailable (Fail-Fast) при отключенном брокере очередей."""
    services = ServiceContainer.get_instance()
    orig_connected = services.publisher._is_connected
    orig_connect = services.publisher.connect
    services.publisher._is_connected = False
    services.publisher.connect = AsyncMock(return_value=False)  # type: ignore[method-assign]
    try:
        payload = {
            "dialogues": [
                {
                    "dialog_id": "failfast-session-001",
                    "messages": [
                        {"participant_index": 0, "text": "тест"},
                        {"participant_index": 1, "text": "ответ"},
                    ],
                }
            ]
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()
    finally:
        services.publisher._is_connected = orig_connected
        services.publisher.connect = orig_connect  # type: ignore[method-assign]


def test_swagger_and_openapi_documentation(client):
    """Проверка доступности интерактивной документации Swagger UI, ReDoc и схемы OpenAPI."""
    resp_docs = client.get("/docs")
    assert resp_docs.status_code == 200
    assert "swagger" in resp_docs.text.lower() or "html" in resp_docs.text.lower()

    resp_redoc = client.get("/redoc")
    assert resp_redoc.status_code == 200

    resp_openapi = client.get("/openapi.json")
    assert resp_openapi.status_code == 200
    schema = resp_openapi.json()
    assert schema["info"]["title"] == "Bot Detection Service API"
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]


def test_health_returns_503_when_dependency_unhealthy(client):
    """Эндпоинты /health и /health/ready возвращают 503 Service Unavailable при деградации зависимостей."""
    services = ServiceContainer.get_instance()
    orig_is_healthy = services.triton_client.is_healthy
    services.triton_client.is_healthy = lambda: False  # type: ignore[method-assign]
    try:
        resp_h = client.get("/health")
        resp_r = client.get("/health/ready")
        assert resp_h.status_code == 503
        assert resp_h.json()["status"] == "unhealthy"
        assert resp_r.status_code == 503
    finally:
        services.triton_client.is_healthy = orig_is_healthy  # type: ignore[method-assign]


def test_predict_publish_failure_cleans_up_redis_task(client):
    """При ошибке постановки задачи в RabbitMQ временная запись PROCESSING удаляется из Redis."""
    services = ServiceContainer.get_instance()
    orig_publish = services.publisher.publish_task
    services.publisher.publish_task = AsyncMock(return_value=False)  # type: ignore[method-assign]
    try:
        fake_store = cast("FakeRedisClient", services.cache.redis_client).store
        keys_before = len(fake_store)
        resp = client.post(
            "/predict",
            json={
                "dialogues": [
                    {
                        "dialog_id": "cleanup-on-publish-fail",
                        "messages": [
                            {"participant_index": 0, "text": "уникальное сообщение тест очистки 1"},
                            {"participant_index": 1, "text": "уникальное сообщение тест очистки 2"},
                        ],
                    }
                ]
            },
        )
        assert resp.status_code == 503
        assert len(fake_store) == keys_before
    finally:
        services.publisher.publish_task = orig_publish  # type: ignore[method-assign]
