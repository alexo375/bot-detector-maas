"""Сквозные (E2E) тесты мониторинга: Prometheus (targets, metrics, scrape) и Grafana (health, dashboards, drift panels)."""

from collections.abc import Generator

import httpx
import pytest

PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3000"

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def prom_client() -> Generator[httpx.Client]:
    """Клиент HTTP для взаимодействия с Prometheus."""
    with httpx.Client(base_url=PROMETHEUS_URL, timeout=5.0) as client:
        yield client


@pytest.fixture(scope="module")
def grafana_client() -> Generator[httpx.Client]:
    """Клиент HTTP для взаимодействия с Grafana API."""
    with httpx.Client(base_url=GRAFANA_URL, auth=("admin", "admin"), timeout=5.0) as client:
        yield client


def test_e2e_prometheus_health_and_targets(prom_client: httpx.Client) -> None:
    """Сквозной тест доступности Prometheus и активности скрейп-таргета API Gateway."""
    # 1. Проверка готовности сервера Prometheus
    ready_resp = prom_client.get("/-/ready")
    assert ready_resp.status_code == 200

    # 2. Проверка состояния таргетов скрейпинга
    targets_resp = prom_client.get("/api/v1/targets")
    assert targets_resp.status_code == 200
    targets_data = targets_resp.json()
    assert targets_data["status"] == "success"

    active_targets = targets_data["data"]["activeTargets"]
    assert len(active_targets) > 0, "Нет активных таргетов скрейпинга в Prometheus"

    # Ищем таргет api-gateway
    gateway_targets = [t for t in active_targets if "api-gateway" in t["labels"].get("job", "")]
    assert len(gateway_targets) > 0, "Таргет 'api-gateway' не найден среди скрейп-джобов Prometheus"
    assert gateway_targets[0]["health"] == "up", f"Таргет api-gateway не в состоянии 'up': {gateway_targets[0]}"


def test_e2e_prometheus_metric_queries(prom_client: httpx.Client) -> None:
    """Сквозной тест выполнения PromQL запросов к метрикам приложения."""
    # Запрашиваем счетчик HTTP запросов или метрики сервиса
    query_resp = prom_client.get("/api/v1/query", params={"query": "up"})
    assert query_resp.status_code == 200
    res = query_resp.json()
    assert res["status"] == "success"
    assert len(res["data"]["result"]) > 0


def test_e2e_grafana_health_and_dashboards_provisioning(grafana_client: httpx.Client) -> None:
    """Сквозной тест доступности Grafana и автоматического провижининга дашбордов."""
    # 1. Проверка состояния сервиса Grafana
    health_resp = grafana_client.get("/api/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["database"] == "ok"

    # 2. Проверка зарегистрированных дашбордов через Search API
    search_resp = grafana_client.get("/api/search")
    assert search_resp.status_code == 200
    dashboards = search_resp.json()

    uids = {d.get("uid") for d in dashboards}
    assert "bot-detector-prod-overview" in uids, "Дашборд 'bot-detector-prod-overview' не найден в Grafana"
    assert "data-drift-overview" in uids, "Дашборд 'data-drift-overview' не найден в Grafana"


def test_e2e_grafana_data_drift_dashboard_panels(grafana_client: httpx.Client) -> None:
    """Сквозной тест структуры дашборда Data Drift: наличие панелей с метриками дрифта признаков."""
    resp = grafana_client.get("/api/dashboards/uid/data-drift-overview")
    assert resp.status_code == 200
    dash_data = resp.json()["dashboard"]

    assert dash_data["title"] == "Data Drift & Feature Quality"
    panels = dash_data.get("panels", [])
    assert len(panels) > 0

    # Проверяем, что панели содержат PromQL запросы к метрикам дрифта признаков
    all_targets_exprs: list[str] = []
    for p in panels:
        for t in p.get("targets", []):
            if "expr" in t:
                all_targets_exprs.append(t["expr"])

    drift_exprs_str = " ".join(all_targets_exprs)
    assert "bot_detector_data_drift_share" in drift_exprs_str or "bot_detector_feature_drift" in drift_exprs_str, (
        f"Дашборд Data Drift не содержит запросов к метрикам дрифта: {drift_exprs_str}"
    )


def test_e2e_grafana_production_overview_dashboard_panels(grafana_client: httpx.Client) -> None:
    """Сквозной тест структуры продакшен-дашборда: наличие панелей задержки и кэш-хитов."""
    resp = grafana_client.get("/api/dashboards/uid/bot-detector-prod-overview")
    assert resp.status_code == 200
    dash_data = resp.json()["dashboard"]

    assert dash_data["title"] == "Bot Detection Production Overview"
    panels = dash_data.get("panels", [])
    assert len(panels) > 0

    all_targets_exprs = [t["expr"] for p in panels for t in p.get("targets", []) if "expr" in t]
    overview_str = " ".join(all_targets_exprs)
    assert "bot_detector" in overview_str or "http_requests" in overview_str
