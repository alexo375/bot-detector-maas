"""Сквозные (E2E) тесты сервиса инференса: HTTP Gateway -> RabbitMQ -> InferenceWorker -> Triton -> Redis."""

import time
import uuid
from collections.abc import Generator

import httpx
import pytest

GATEWAY_BASE_URL = "http://localhost:8080"

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def http_client() -> Generator[httpx.Client]:
    """Клиент HTTP для взаимодействия с живым API Gateway."""
    with httpx.Client(base_url=GATEWAY_BASE_URL, timeout=10.0) as client:
        yield client


def test_e2e_full_async_inference_serving_lifecycle(http_client: httpx.Client) -> None:
    """Сквозной тест полного цикла инференса: POST -> 202 Accepted -> очередь -> Worker -> Triton -> GET 200 OK."""
    unique_dialog_id = f"e2e_dialog_{uuid.uuid4().hex[:8]}"

    payload = {
        "dialogues": [
            {
                "dialog_id": unique_dialog_id,
                "messages": [
                    {"participant_index": 0, "text": f"Здравствуйте {unique_dialog_id}, где мой заказ?"},
                    {"participant_index": 1, "text": f"Заказ {unique_dialog_id} передан курьеру и будет доставлен."},
                ],
            }
        ],
        "audit_mode": "off",
    }

    # 1. Отправляем запрос на инференс диалога
    post_resp = http_client.post("/predict", json=payload)
    assert post_resp.status_code == 202, f"Ожидался HTTP 202, получен {post_resp.status_code}: {post_resp.text}"

    data_202 = post_resp.json()
    assert "task_id" in data_202
    assert data_202["status"] == "processing"
    task_id = data_202["task_id"]

    # 2. Опрашиваем GET /predict/{task_id} до получения готового результата
    max_wait_seconds = 10.0
    poll_interval = 0.2
    start_time = time.monotonic()
    final_resp_data = None

    while time.monotonic() - start_time < max_wait_seconds:
        get_resp = http_client.get(f"/predict/{task_id}")
        if get_resp.status_code == 200:
            final_resp_data = get_resp.json()
            break
        assert get_resp.status_code == 202, f"Неожиданный статус при опросе: {get_resp.status_code}"
        time.sleep(poll_interval)

    assert final_resp_data is not None, f"Таймаут ожидания задачи {task_id} ({max_wait_seconds} сек)"

    # 3. Валидация структуры ответа
    assert final_resp_data["total"] == 1
    assert len(final_resp_data["results"]) == 1

    res = final_resp_data["results"][0]
    assert res["dialog_id"] == unique_dialog_id
    assert res["dialogue_length"] == 2
    assert res["cached"] is False
    assert len(res["verdicts"]) == 2

    v0, v1 = res["verdicts"][0], res["verdicts"][1]
    assert v0["participant_index"] == 0
    assert v1["participant_index"] == 1
    assert 0.0 <= v0["bot_probability"] <= 1.0
    assert 0.0 <= v1["bot_probability"] <= 1.0
    # Проверка физического ограничения парного постпроцессора
    assert v0["bot_probability"] + v1["bot_probability"] <= 1.0 + 1e-4

    # 4. Проверка мгновенного O(1) попадания в кэш Redis при повторном запросе того же диалога
    repeat_post_resp = http_client.post("/predict", json=payload)
    assert repeat_post_resp.status_code == 200, "Повторный запрос должен мгновенно возвращать 200 OK из кэша"

    cached_data = repeat_post_resp.json()
    assert cached_data["total"] == 1
    cached_res = cached_data["results"][0]
    assert cached_res["dialog_id"] == unique_dialog_id
    assert cached_res["cached"] is True
    assert cached_data["batch_latency_ms"] <= 10.0


def test_e2e_multi_dialogue_batch_serving(http_client: httpx.Client) -> None:
    """Сквозной тест пакетного инференса нескольких диалогов в одном запросе."""
    d1 = f"batch_d1_{uuid.uuid4().hex[:6]}"
    d2 = f"batch_d2_{uuid.uuid4().hex[:6]}"

    payload = {
        "dialogues": [
            {
                "dialog_id": d1,
                "messages": [
                    {"participant_index": 0, "text": f"Первый диалог {d1}, привет"},
                    {"participant_index": 1, "text": f"Привет первому диалогу {d1}"},
                ],
            },
            {
                "dialog_id": d2,
                "messages": [
                    {"participant_index": 0, "text": f"Второй диалог {d2}, как дела?"},
                    {"participant_index": 1, "text": f"Нормально все {d2}"},
                ],
            },
        ],
        "audit_mode": "off",
    }

    resp = http_client.post("/predict", json=payload)
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]

    # Ожидаем завершения батча
    max_wait = 10.0
    start = time.monotonic()
    result = None

    while time.monotonic() - start < max_wait:
        poll = http_client.get(f"/predict/{task_id}")
        if poll.status_code == 200:
            result = poll.json()
            break
        time.sleep(0.2)

    assert result is not None
    assert result["total"] == 2
    assert len(result["results"]) == 2
    assert {r["dialog_id"] for r in result["results"]} == {d1, d2}


def test_e2e_non_existent_task_returns_404(http_client: httpx.Client) -> None:
    """Опрос несуществующей задачи возвращает HTTP 404."""
    resp = http_client.get("/predict/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()
