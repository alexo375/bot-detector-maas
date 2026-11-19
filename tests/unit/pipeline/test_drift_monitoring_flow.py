from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from prefect.deployments.runner import EntrypointType

from project.drift.data_drift import DataDriftSummary
from project.pipeline.drift_monitoring.flow import (
    DriftMonitoringDeployer,
    drift_monitoring_flow,
)
from project.pipeline.drift_monitoring.schemas import FeatureDistributionData

pytestmark = [pytest.mark.xdist_group("prefect"), pytest.mark.usefixtures("prefect_isolated_server")]


def test_drift_monitoring_flow_healthy() -> None:
    """Проверка выполнения флоу мониторинга при здоровом распределении данных."""
    ref_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, 0.2, 0.3]})
    cur_df = pd.DataFrame({"f1": [1.1, 2.1, 2.9], "f2": [0.12, 0.19, 0.31]})

    with (
        patch("project.pipeline.drift_monitoring.flow.fetch_dwh_data_task") as mock_fetch,
        patch("project.pipeline.drift_monitoring.flow.detect_feature_drift_task") as mock_detect,
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.trigger_api_gateway_metrics") as mock_trigger,
        patch(
            "project.drift.data_drift.fetcher.DWHFeatureFetcher.fetch_prediction_data",
            return_value=(np.array([0.2, 0.8] * 10), np.array([0.2, 0.8] * 10), ["human", "bot"] * 10),
        ),
        patch(
            "project.drift.data_drift.fetcher.DWHFeatureFetcher.fetch_concept_drift_probs",
            return_value={
                "catboost": np.array([0.1, 0.9] * 5),
                "rubert": np.array([0.1, 0.9] * 5),
            },
        ),
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.count_recent_dialogues", return_value=10),
        patch(
            "project.drift.model_age.ChampionMetadataReader.read",
            return_value={"age_days": 1.0, "version": "1"},
        ),
        patch(
            "project.pipeline.drift_monitoring.decision.DriftDecisionEngine.is_retrain_in_progress", return_value=False
        ),
        patch(
            "project.pipeline.drift_monitoring.decision.DriftDecisionEngine.is_retrain_in_cooldown", return_value=False
        ),
        patch("redis.Redis.from_url"),
    ):
        mock_fetch.return_value = FeatureDistributionData(ref_df=ref_df, cur_df=cur_df)
        mock_detect.return_value = DataDriftSummary(
            dataset_drift=False,
            drift_share=0.05,
            number_of_drifted_columns=0,
            number_of_columns=2,
            drift_threshold=0.20,
            retrain_recommended=False,
            drifted_features=[],
        )

        result = drift_monitoring_flow(days=7, drift_threshold=0.20)

        assert result["dataset_drift"] is False
        assert result["retrain_recommended"] is False
        assert result["drift_share"] == 0.05
        assert result["number_of_columns"] == 2
        mock_trigger.assert_called_once()


def test_drift_monitoring_flow_drift_detected() -> None:
    """Проверка поведения флоу при детекции значительного дрифта и достаточном объеме данных."""
    ref_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, 0.2, 0.3]})
    cur_df = pd.DataFrame({"f1": [10.0, 20.0, 30.0], "f2": [5.0, 6.0, 7.0]})

    with (
        patch("project.pipeline.drift_monitoring.flow.fetch_dwh_data_task") as mock_fetch,
        patch("project.pipeline.drift_monitoring.flow.detect_feature_drift_task") as mock_detect,
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.trigger_api_gateway_metrics"),
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.count_recent_dialogues", return_value=500),
        patch(
            "project.drift.data_drift.fetcher.DWHFeatureFetcher.fetch_prediction_data",
            return_value=(np.array([]), np.array([]), []),
        ),
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.fetch_concept_drift_probs", return_value={}),
        patch(
            "project.drift.model_age.ChampionMetadataReader.read",
            return_value={"age_days": 1.0, "version": "1"},
        ),
        patch(
            "project.pipeline.drift_monitoring.decision.DriftDecisionEngine.is_retrain_in_progress", return_value=False
        ),
        patch(
            "project.pipeline.drift_monitoring.decision.DriftDecisionEngine.is_retrain_in_cooldown", return_value=False
        ),
        patch("prefect.deployments.run_deployment") as mock_run_dep,
        patch("redis.Redis.from_url"),
    ):
        mock_fetch.return_value = FeatureDistributionData(ref_df=ref_df, cur_df=cur_df)
        mock_detect.return_value = DataDriftSummary(
            dataset_drift=True,
            drift_share=0.80,
            number_of_drifted_columns=2,
            number_of_columns=2,
            drift_threshold=0.20,
            retrain_recommended=True,
            drifted_features=["f1", "f2"],
        )

        result = drift_monitoring_flow(days=3, drift_threshold=0.20, auto_trigger=True)

        assert result["dataset_drift"] is True
        assert result["retrain_recommended"] is True
        assert result["retrain_status"] == 2
        assert result["decision_reason"] == "drift_detected"
        assert result["drift_share"] == 0.80
        assert result["drifted_features"] == ["f1", "f2"]
        mock_run_dep.assert_called_once_with(
            name="bot_detection_retrain_flow/bot-detection-retraining",
            timeout=0,
            parameters={"force_retrain": True},
        )


def test_drift_monitoring_flow_manual_mode_by_default() -> None:
    """По умолчанию авто-триггер ретрейна выключен: дрифт меняет статус на 2 (RETRAIN NEEDED), но Flow не стартует."""
    ref_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, 0.2, 0.3]})
    cur_df = pd.DataFrame({"f1": [10.0, 20.0, 30.0], "f2": [5.0, 6.0, 7.0]})

    with (
        patch("project.pipeline.drift_monitoring.flow.fetch_dwh_data_task") as mock_fetch,
        patch("project.pipeline.drift_monitoring.flow.detect_feature_drift_task") as mock_detect,
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.trigger_api_gateway_metrics"),
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.count_recent_dialogues", return_value=500),
        patch(
            "project.drift.data_drift.fetcher.DWHFeatureFetcher.fetch_prediction_data",
            return_value=(np.array([]), np.array([]), []),
        ),
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.fetch_concept_drift_probs", return_value={}),
        patch(
            "project.drift.model_age.ChampionMetadataReader.read",
            return_value={"age_days": 1.0, "version": "1"},
        ),
        patch(
            "project.pipeline.drift_monitoring.decision.DriftDecisionEngine.is_retrain_in_progress", return_value=False
        ),
        patch("prefect.deployments.run_deployment") as mock_run_dep,
        patch("redis.Redis.from_url"),
    ):
        mock_fetch.return_value = FeatureDistributionData(ref_df=ref_df, cur_df=cur_df)
        mock_detect.return_value = DataDriftSummary(
            dataset_drift=True,
            drift_share=0.80,
            number_of_drifted_columns=2,
            number_of_columns=2,
            drift_threshold=0.20,
            retrain_recommended=True,
            drifted_features=["f1", "f2"],
        )

        result = drift_monitoring_flow(days=3, drift_threshold=0.20)

        assert result["dataset_drift"] is True
        assert result["retrain_recommended"] is True
        assert result["retrain_status"] == 2
        assert result["decision_reason"] == "drift_detected"
        mock_run_dep.assert_not_called()


def test_drift_monitoring_flow_skips_when_retrain_in_progress() -> None:
    """Мониторинг не ставит дублирующие ретрейны и держит статус 3 (RETRAINING), пока обучение активно."""
    ref_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, 0.2, 0.3]})
    cur_df = pd.DataFrame({"f1": [10.0, 20.0, 30.0], "f2": [5.0, 6.0, 7.0]})

    with (
        patch("project.pipeline.drift_monitoring.flow.fetch_dwh_data_task") as mock_fetch,
        patch("project.pipeline.drift_monitoring.flow.detect_feature_drift_task") as mock_detect,
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.trigger_api_gateway_metrics"),
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.count_recent_dialogues", return_value=500),
        patch(
            "project.drift.data_drift.fetcher.DWHFeatureFetcher.fetch_prediction_data",
            return_value=(np.array([]), np.array([]), []),
        ),
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.fetch_concept_drift_probs", return_value={}),
        patch(
            "project.drift.model_age.ChampionMetadataReader.read",
            return_value={"age_days": 1.0, "version": "1"},
        ),
        patch(
            "project.pipeline.drift_monitoring.decision.DriftDecisionEngine.is_retrain_in_progress", return_value=True
        ),
        patch("prefect.deployments.run_deployment") as mock_run_dep,
        patch("redis.Redis.from_url"),
    ):
        mock_fetch.return_value = FeatureDistributionData(ref_df=ref_df, cur_df=cur_df)
        mock_detect.return_value = DataDriftSummary(
            dataset_drift=True,
            drift_share=0.80,
            number_of_drifted_columns=2,
            number_of_columns=2,
            drift_threshold=0.20,
            retrain_recommended=True,
            drifted_features=["f1", "f2"],
        )

        result = drift_monitoring_flow(days=3, drift_threshold=0.20)

        assert result["retrain_status"] == 3
        assert result["retrain_recommended"] is False
        assert result["decision_reason"] == "retraining_in_progress"
        mock_run_dep.assert_not_called()


def test_drift_monitoring_flow_drift_awaiting_data() -> None:
    """Проверка перехода в статус AWAITING_DATA при дрифте, но малом объеме новых данных."""
    ref_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, 0.2, 0.3]})
    cur_df = pd.DataFrame({"f1": [10.0, 20.0, 30.0], "f2": [5.0, 6.0, 7.0]})

    with (
        patch("project.pipeline.drift_monitoring.flow.fetch_dwh_data_task") as mock_fetch,
        patch("project.pipeline.drift_monitoring.flow.detect_feature_drift_task") as mock_detect,
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.trigger_api_gateway_metrics"),
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.count_recent_dialogues", return_value=15),
        patch(
            "project.drift.data_drift.fetcher.DWHFeatureFetcher.fetch_prediction_data",
            return_value=(np.array([]), np.array([]), []),
        ),
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.fetch_concept_drift_probs", return_value={}),
        patch(
            "project.drift.model_age.ChampionMetadataReader.read",
            return_value={"age_days": 2.0, "version": "1"},
        ),
        patch(
            "project.pipeline.drift_monitoring.decision.DriftDecisionEngine.is_retrain_in_progress", return_value=False
        ),
        patch("redis.Redis.from_url"),
    ):
        mock_fetch.return_value = FeatureDistributionData(ref_df=ref_df, cur_df=cur_df)
        mock_detect.return_value = DataDriftSummary(
            dataset_drift=True,
            drift_share=0.80,
            number_of_drifted_columns=2,
            number_of_columns=2,
            drift_threshold=0.20,
            retrain_recommended=True,
            drifted_features=["f1", "f2"],
        )

        result = drift_monitoring_flow(days=3, drift_threshold=0.20)

        assert result["dataset_drift"] is True
        assert result["retrain_recommended"] is False
        assert result["retrain_status"] == 1
        assert result["decision_reason"] == "drift_detected_awaiting_data"


def test_deploy_drift_monitoring_applies_deployment() -> None:
    """Проверка регистрации расписания и лимита конкурентности через deploy_drift_monitoring."""
    from prefect.client.schemas.objects import ConcurrencyLimitConfig, ConcurrencyLimitStrategy

    from project.core.config import config

    mock_dep = MagicMock()
    mock_dep.apply.return_value = "fake-uuid-1234"

    with patch.object(drift_monitoring_flow, "to_deployment", return_value=mock_dep) as mock_to_dep:
        dep_id = DriftMonitoringDeployer.deploy()

        assert dep_id == "fake-uuid-1234"
        mock_to_dep.assert_called_once_with(
            name="drift-monitor",
            work_pool_name="default-agent-pool",
            cron=config.schedule.drift_monitoring_cron,
            concurrency_limit=ConcurrencyLimitConfig(
                limit=1,
                collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW,
                grace_period_seconds=60,
            ),
            tags=["monitoring", "data-drift", "decision-engine"],
            description="Регулярный автоматический аудит распределений признаков, объема данных и адаптивного ретрейна",
            entrypoint_type=EntrypointType.MODULE_PATH,
        )
        assert mock_dep.entrypoint == "project.pipeline.drift_monitoring.flow:drift_monitoring_flow"
        mock_dep.apply.assert_called_once()


def test_drift_monitoring_scopes_window_to_champion_timestamp() -> None:
    """Мониторинг фильтрует chat_logs по created_at_ts активного чемпиона из MLflow."""
    ref_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, 0.2, 0.3]})
    cur_df = pd.DataFrame({"f1": [1.1, 2.1, 2.9], "f2": [0.12, 0.19, 0.31]})
    champ_ts = 1700000123.0

    with (
        patch("project.pipeline.drift_monitoring.flow.fetch_dwh_data_task") as mock_fetch,
        patch("project.pipeline.drift_monitoring.flow.detect_feature_drift_task") as mock_detect,
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.trigger_api_gateway_metrics"),
        patch(
            "project.drift.data_drift.fetcher.DWHFeatureFetcher.fetch_prediction_data",
            return_value=(np.array([0.2, 0.8] * 10), np.array([0.2, 0.8] * 10), ["human", "bot"] * 10),
        ) as mock_pred_fetch,
        patch(
            "project.drift.data_drift.fetcher.DWHFeatureFetcher.fetch_concept_drift_probs", return_value={}
        ) as mock_cd,
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.count_recent_dialogues", return_value=12) as mock_cnt,
        patch(
            "project.drift.model_age.ChampionMetadataReader.read",
            return_value={"age_days": 0.0, "version": "2", "created_at_ts": champ_ts},
        ),
        patch(
            "project.pipeline.drift_monitoring.decision.DriftDecisionEngine.is_retrain_in_progress", return_value=False
        ),
        patch("redis.Redis.from_url"),
    ):
        mock_fetch.return_value = FeatureDistributionData(ref_df=ref_df, cur_df=cur_df)
        mock_detect.return_value = {
            "dataset_drift": False,
            "drift_share": 0.0,
            "retrain_recommended": False,
            "drifted_features": [],
            "number_of_drifted_columns": 0,
            "number_of_columns": 2,
        }

        drift_monitoring_flow(days=7, drift_threshold=0.20)

        mock_fetch.assert_called_once_with(days=7, since_ts=champ_ts)
        mock_pred_fetch.assert_called_once_with(days=7, since_ts=champ_ts)
        mock_cd.assert_called_once_with(days=7, since_ts=champ_ts, limit=200)
        mock_cnt.assert_called_once_with(days=7, since_ts=champ_ts)


def test_drift_monitoring_no_traffic_stays_healthy() -> None:
    """При пустой chat_logs (< 10 диалогов) мониторинг не гоняет fallback-заглушки и остаётся в HEALTHY."""
    with (
        patch("project.pipeline.drift_monitoring.flow.fetch_dwh_data_task") as mock_fetch,
        patch("project.pipeline.drift_monitoring.flow.detect_feature_drift_task") as mock_detect,
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.trigger_api_gateway_metrics") as mock_trigger,
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.count_recent_dialogues", return_value=0),
        patch(
            "project.drift.model_age.ChampionMetadataReader.read",
            return_value={"age_days": 0.0, "version": "1", "created_at_ts": None},
        ),
        patch("redis.Redis.from_url"),
    ):
        result = drift_monitoring_flow(days=7, drift_threshold=0.20)

        mock_fetch.assert_not_called()
        mock_detect.assert_not_called()
        mock_trigger.assert_called_once()
        prediction_drift_psi = result["prediction_drift_psi"]
        prediction_entropy = result["prediction_entropy"]
        ensemble_disagreement_rate = result["ensemble_disagreement_rate"]
        assert result["dataset_drift"] is False
        assert result["drift_share"] == 0.0
        assert prediction_drift_psi == 0.0
        assert prediction_entropy == 0.0
        assert ensemble_disagreement_rate == 0.0
        assert result["retrain_status"] == 0
        assert result["retrain_recommended"] is False
        assert result["decision_reason"] == "system_healthy"


def test_drift_monitoring_restores_baseline_if_promoted_mid_run() -> None:
    """Если чемпион сменился в MLflow во время прогона мониторинга, устаревший дрифт сбрасывается в baseline."""
    ref_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, 0.2, 0.3]})
    cur_df = pd.DataFrame({"f1": [10.0, 20.0, 30.0], "f2": [5.0, 6.0, 7.0]})

    with (
        patch("project.pipeline.drift_monitoring.flow.fetch_dwh_data_task") as mock_fetch,
        patch("project.pipeline.drift_monitoring.flow.detect_feature_drift_task") as mock_detect,
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.trigger_api_gateway_metrics") as mock_trigger,
        patch(
            "project.drift.data_drift.fetcher.DWHFeatureFetcher.fetch_prediction_data",
            return_value=(np.array([0.2, 0.8] * 10), np.array([0.9, 0.1] * 10), ["human", "bot"] * 10),
        ),
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.fetch_concept_drift_probs", return_value={}),
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.count_recent_dialogues", side_effect=[200, 0]),
        patch(
            "project.drift.model_age.ChampionMetadataReader.read",
            side_effect=[
                {"age_days": 1.0, "version": "10", "created_at_ts": 1700000000.0},
                {"age_days": 0.0, "version": "11", "created_at_ts": 1700001000.0},
            ],
        ),
        patch(
            "project.pipeline.drift_monitoring.decision.DriftDecisionEngine.is_retrain_in_progress", return_value=False
        ),
        patch("prefect.deployments.run_deployment") as mock_run_dep,
        patch("redis.Redis.from_url"),
    ):
        mock_fetch.return_value = FeatureDistributionData(ref_df=ref_df, cur_df=cur_df)
        mock_detect.return_value = DataDriftSummary(
            dataset_drift=True,
            drift_share=0.80,
            number_of_drifted_columns=2,
            number_of_columns=2,
            drift_threshold=0.20,
            retrain_recommended=True,
            drifted_features=["f1", "f2"],
        )

        result = drift_monitoring_flow(days=7, drift_threshold=0.20)

        assert result["drift_share"] == 0.0
        assert result["prediction_drift_psi"] == 0.0
        assert result["retrain_status"] == 0
        assert result["retrain_recommended"] is False
        assert result["decision_reason"] == "promoted"
        mock_run_dep.assert_not_called()
        mock_trigger.assert_called_once()


def test_drift_monitoring_flow_handles_missing_feature_store_state() -> None:
    """При отсутствии feature_store_state.json flow не падает, а возвращает awaiting_model_initialization."""
    with (
        patch(
            "project.drift.data_drift.fetcher.DWHFeatureFetcher.count_recent_dialogues",
            side_effect=FileNotFoundError("feature_store_state.json not found"),
        ),
        patch(
            "project.drift.model_age.ChampionMetadataReader.read",
            return_value={"age_days": 0.0, "version": None, "created_at_ts": None},
        ),
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.persist_adaptive_decision") as mock_persist,
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.trigger_api_gateway_metrics") as mock_trigger,
    ):
        result = drift_monitoring_flow(days=7, drift_threshold=0.20)

    assert result["retrain_status"] == 0
    assert result["retrain_recommended"] is False
    assert result["decision_reason"] == "awaiting_model_initialization"
    mock_persist.assert_called_once_with(0, 0.0, 0, "awaiting_model_initialization")
    mock_trigger.assert_called_once()
