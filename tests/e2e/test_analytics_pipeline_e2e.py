"""Сквозные (E2E) тесты аналитического пайплайна: API Gateway -> Worker -> RabbitMQ -> ClickHouse (тестовая БД)."""

import time
import uuid
from collections.abc import Generator

import httpx
import pytest

from project.infrastructure.clickhouse import ClickHouseConnectionFactory, FeatureDWH

GATEWAY_BASE_URL = "http://localhost:8080"
TEST_DB = "bot_detection_test"

pytestmark = [pytest.mark.live, pytest.mark.xdist_group("clickhouse_test_db"), pytest.mark.xdist_group("live_e2e")]


@pytest.fixture(scope="module")
def ch_client() -> Generator[object]:
    """Изолированная тестовая БД ClickHouse для E2E проверки аналитики без касания prod."""
    admin = ClickHouseConnectionFactory.create()
    admin.command(f"CREATE DATABASE IF NOT EXISTS {TEST_DB}")
    client = ClickHouseConnectionFactory.create(database=TEST_DB)
    FeatureDWH(client).init_schema()
    try:
        yield client
    finally:
        client.close()
        admin.command(f"DROP DATABASE IF EXISTS {TEST_DB}")
        admin.close()


def test_e2e_inference_to_clickhouse_audit_logging(ch_client):
    """Сквозной E2E тест аудита: audit_mode='test' маршрутизируется в bot_detection_test.chat_logs."""
    unique_dialog_id = f"e2e_audit_{uuid.uuid4().hex[:10]}"

    payload = {
        "dialogues": [
            {
                "dialog_id": unique_dialog_id,
                "messages": [
                    {"participant_index": 0, "text": f"Добрый день {unique_dialog_id}, как активировать промокод?"},
                    {"participant_index": 1, "text": f"Промокод {unique_dialog_id} можно ввести в поле 'Купон'."},
                ],
            }
        ],
        "audit_mode": "test",
    }

    # 1. Отправляем запрос на инференс диалога через шлюз
    with httpx.Client(base_url=GATEWAY_BASE_URL, timeout=10.0) as http_client:
        post_resp = http_client.post("/predict", json=payload)
        assert post_resp.status_code == 202
        task_id = post_resp.json()["task_id"]

        # Опрашиваем результат инференса
        max_wait = 10.0
        start = time.monotonic()
        infer_result = None
        while time.monotonic() - start < max_wait:
            poll_resp = http_client.get(f"/predict/{task_id}")
            if poll_resp.status_code == 200:
                infer_result = poll_resp.json()
                break
            time.sleep(0.2)

    assert infer_result is not None, "Инференс не завершился в отведенное время"
    item = infer_result["results"][0]
    expected_p0 = item["verdicts"][0]["bot_probability"]
    expected_p1 = item["verdicts"][1]["bot_probability"]

    # 2. Ожидаем появления записи в ClickHouse (микробатч consumer сбрасывает каждые 2 сек)
    max_dwh_wait = 10.0
    dwh_start = time.monotonic()
    row_found = False
    ch_record = None

    while time.monotonic() - dwh_start < max_dwh_wait:
        res = ch_client.query(
            f"SELECT dialog_id, num_messages, prob_p0, prob_p1, is_bot_p0, is_bot_p1, verdict, latency_ms "
            f"FROM chat_logs WHERE dialog_id = '{unique_dialog_id}'"
        )
        if len(res.result_rows) > 0:
            row_found = True
            ch_record = res.result_rows[0]
            break
        time.sleep(0.5)

    assert row_found, f"Событие аудита диалога {unique_dialog_id} не появилось в ClickHouse chat_logs"
    assert ch_record is not None

    # 3. Валидация точности сохраненных данных
    assert ch_record[0] == unique_dialog_id
    assert ch_record[1] == 2  # num_messages
    assert pytest.approx(ch_record[2], abs=1e-3) == expected_p0
    assert pytest.approx(ch_record[3], abs=1e-3) == expected_p1
    assert ch_record[4] in (0, 1)  # is_bot_p0
    assert ch_record[5] in (0, 1)  # is_bot_p1
    assert ch_record[6] in ("both_humans", "p0_bot", "p1_bot", "both_bots")
    assert ch_record[7] > 0.0  # latency_ms
