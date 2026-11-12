"""Юнит-тесты URL-реестра ServiceUrls и скрипта инъекции дрифта."""

import json
from unittest.mock import MagicMock, patch

import pytest

from project.core.urls import ServiceUrls, service_urls
from scripts.inject_drift import _select_hard_short_train_dialogues


def test_service_urls_resolution() -> None:
    """Проверка генерации ссылок менеджером ServiceUrls."""
    urls = ServiceUrls(public_host="127.0.0.1", grafana_port=3000, prefect_port=4200, mlflow_port=5000)
    assert urls.grafana_drift_dashboard == "http://127.0.0.1:3000/d/data-drift-overview"
    assert urls.prefect_deployments == "http://127.0.0.1:4200/v2/deployments"
    assert urls.prefect_retrain_flow == "http://127.0.0.1:4200/v2/deployments"
    assert urls.mlflow_models == "http://127.0.0.1:5000/#/models"
    assert "data-drift-overview" in service_urls.grafana_drift_dashboard


def test_select_hard_short_train_dialogues_strict_clickhouse() -> None:
    """Отбор самых коротких/сложных диалогов строго из ClickHouse; при сбое — ошибка без локального фолбэка."""
    mock_client = MagicMock()
    mock_client.query.return_value.result_rows = [
        (
            "short_dlg",
            json.dumps(
                [
                    {"participant_index": 0, "text": "да"},
                    {"participant_index": 1, "text": "ок"},
                ]
            ),
        )
    ]
    with patch("scripts.inject_drift._get_ch_client", return_value=mock_client):
        selected = _select_hard_short_train_dialogues(count=1)

    assert len(selected) == 1
    assert selected[0]["dialog_id"].startswith("drift_inject_")
    assert [m["text"] for m in selected[0]["messages"]] == ["да", "ок"]

    with (
        patch("scripts.inject_drift._get_ch_client", side_effect=RuntimeError("no ch")),
        pytest.raises(RuntimeError, match="no ch"),
    ):
        _select_hard_short_train_dialogues(count=1)
