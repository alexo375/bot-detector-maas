"""Locust нагрузочные тесты API Gateway (синхронные и pull-модели инференса)."""

import os

from locust import HttpUser, between, task

from tests.fixtures.load_test_utils import (
    generate_unique_dialogue,
    make_cache_payload,
    make_payload,
)

HOST = os.getenv("LOAD_TEST_HOST", "http://127.0.0.1:8080")

# Детерминированный диалог для проверки O(1) чтения из Redis-кэша.
# Первый запрос прогревает кэш (202/200), последующие — мгновенный хит.
CACHE_PAYLOAD = make_cache_payload()

# Максимальное число попыток поллинга результата асинхронной задачи.
POLL_MAX_ATTEMPTS = 30


class BotDetectorUser(HttpUser):
    host = HOST
    # wait_time задаёт темп генерации новых итераций задач каждым пользователем.
    wait_time = between(0.5, 1.5)

    def on_start(self) -> None:
        # Прогрев кэша детерминированным диалогом: первый запрос кладёт вердикт в Redis.
        self.client.post("/predict", json=CACHE_PAYLOAD, name="POST /predict [Cache Warmup]")

    @task(6)
    def predict_cache_miss(self) -> None:
        """60% трафика: уникальный диалог — гарантированный промах кэша, постановка в очередь."""
        with self.client.post(
            "/predict",
            json=make_payload(generate_unique_dialogue()),
            catch_response=True,
            name="POST /predict [Queue Enqueue]",
        ) as response:
            if response.status_code == 202:
                data = response.json()
                if data.get("task_id") and data.get("poll_url"):
                    response.success()
                else:
                    response.failure(f"Missing task_id/poll_url in 202 response: {response.text}")
            elif response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unexpected status code {response.status_code}: {response.text}")

    @task(2)
    def predict_cache_hit(self) -> None:
        """20% трафика: детерминированный диалог — O(1) чтение готового вердикта из Redis."""
        with self.client.post(
            "/predict",
            json=CACHE_PAYLOAD,
            catch_response=True,
            name="POST /predict [Cache Hit / Redis]",
        ) as response:
            if response.status_code in (200, 202):
                response.success()
            else:
                response.failure(f"Cache request failed with {response.status_code}: {response.text}")

    @task(2)
    def async_pull(self) -> None:
        """20% трафика: pull-модель — постановка задачи и поллинг GET /predict/{task_id} до 200 OK."""
        task_id: str | None = None
        with self.client.post(
            "/predict",
            json=make_payload(generate_unique_dialogue()),
            catch_response=True,
            name="POST /predict [Pull Init]",
        ) as response:
            if response.status_code == 202:
                task_id = response.json().get("task_id")
                if not task_id:
                    response.failure(f"Missing task_id in 202 response: {response.text}")
                    return
                response.success()
            elif response.status_code == 200:
                response.success()
                return
            else:
                response.failure(f"Failed to submit task: {response.status_code}: {response.text}")
                return

        for attempt in range(POLL_MAX_ATTEMPTS):
            with self.client.get(
                f"/predict/{task_id}",
                catch_response=True,
                name="GET /predict/{task_id} [Pull Poll]",
            ) as poll_resp:
                if poll_resp.status_code == 200:
                    poll_resp.success()
                    return
                if poll_resp.status_code != 202:
                    poll_resp.failure(f"Unexpected poll status: {poll_resp.status_code}")
                    return
                if attempt == POLL_MAX_ATTEMPTS - 1:
                    poll_resp.failure(f"Task still processing after {POLL_MAX_ATTEMPTS} attempts")
                else:
                    poll_resp.success()  # 202 — задача ещё обрабатывается, это норма
            self.wait()
