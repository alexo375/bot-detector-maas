"""Модульные тесты оркестратора дообучения моделей (Prefect, Triton Hot Reload, Champion/Challenger)."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import onnx
import pandas as pd
import pytest

from project.core.config import config
from project.infrastructure.triton.version_manager import TritonModelVersionManager
from project.pipeline.mlflow_retrainer import (
    FeatureDistributionData,
    FlowExecutionResult,
    FlowStatus,
    GatekeeperDecision,
    GatekeeperMetrics,
    RetrainEvaluationResult,
    bot_detection_retrain_flow,
)
from project.pipeline.mlflow_retrainer.components.gatekeeper import RetrainingGatekeeper
from project.pipeline.serving import ModelPublisher, TritonReloader

pytestmark = [pytest.mark.xdist_group("prefect"), pytest.mark.usefixtures("prefect_isolated_server")]


def test_triton_hot_reload_mock() -> None:
    """Проверка отправки команд Hot Reload в Triton Server с мокированием HTTP."""
    reloader = TritonReloader(triton_url="http://mock-triton:8000")
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        success = reloader.reload_model("catboost_detector")
        assert success is True

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Network error")
        mock_client_cls.return_value.__enter__.return_value = mock_client

        with pytest.raises(RuntimeError, match="Could not reach Triton"):
            reloader.reload_model("catboost_detector")

    with patch.object(TritonReloader, "reload_model", return_value=True) as mock_hr:
        results = TritonReloader().reload_all()
        assert results["catboost_detector"] is True
        assert results["lightgbm_detector"] is True
        assert results["xgboost_detector"] is True
        assert results["meta_stacker"] is True
        assert results["bot_ensemble"] is True
        assert "rubert_detector" not in results
        assert mock_hr.call_count == 5


def test_retrain_flow_skips_when_no_drift() -> None:
    """Проверка, что пайплайн Prefect пропускает переобучение, если дрифт отсутствует."""
    feature_names = [f"feature_{i}" for i in range(5)]
    np.random.seed(42)
    synthetic_df = pd.DataFrame(np.random.normal(0, 1, size=(50, 5)), columns=feature_names)

    with (
        patch("project.pipeline.mlflow_retrainer.flow.fetch_dwh_data_task") as mock_fetch,
        patch("project.pipeline.mlflow_retrainer.flow.detect_feature_drift_task") as mock_drift,
        patch("project.pipeline.mlflow_retrainer.flow.retrain_models_task") as mock_retrain,
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.count_recent_dialogues", return_value=200),
    ):
        mock_fetch.return_value = {"ref_df": synthetic_df, "cur_df": synthetic_df}
        mock_drift.return_value = {"retrain_recommended": False}

        result = bot_detection_retrain_flow(force_retrain=False, drift_threshold=0.50)

        assert result.status == "skipped"
        assert result.reason == "no_drift_detected"
        assert not mock_retrain.called


def test_onnx_artifacts_validity() -> None:
    """Проверка валидности экспортированных ONNX графов."""
    triton_repo = Path(config.triton.model_repository)
    ver_mgr = TritonModelVersionManager(repository_path=triton_repo)
    for model_name in ("catboost_detector", "lightgbm_detector", "xgboost_detector", "meta_stacker"):
        p = ver_mgr.get_model_path(model_name)
        assert p is not None, f"ONNX path for {model_name} missing"
        assert p.exists(), f"ONNX file for {model_name} does not exist"
        m = onnx.load(str(p))
        onnx.checker.check_model(m)
        if model_name == "meta_stacker":
            input_shape = m.graph.input[0].type.tensor_type.shape.dim
            assert input_shape[1].dim_value == 3


def test_champion_challenger_promotion() -> None:
    """Проверка промоушена Challenger, когда его LogLoss лучше Champion."""
    with (
        patch("project.pipeline.mlflow_retrainer.flow.fetch_dwh_data_task") as mock_fetch,
        patch("project.pipeline.mlflow_retrainer.flow.detect_feature_drift_task") as mock_drift,
        patch("project.pipeline.mlflow_retrainer.flow.retrain_models_task") as mock_retrain,
        patch("project.pipeline.mlflow_retrainer.flow.export_onnx_task") as mock_export,
        patch("project.pipeline.mlflow_retrainer.flow.hot_reload_triton_task") as mock_reload,
    ):
        mock_fetch.return_value = {"ref_df": None, "cur_df": None}
        mock_drift.return_value = {"retrain_recommended": True}
        mock_retrain.return_value = {
            "metrics": {
                "champion_loss": 0.1350,
                "challenger_loss": 0.1250,
                "promoted": True,
                "decision": "PROMOTED",
                "reason": "challenger_improved_champion",
            },
            "diff_data": {"features": []},
        }
        mock_reload.return_value = {"catboost_detector": True}

        result = bot_detection_retrain_flow(force_retrain=True)

        assert result.status == "completed"
        assert result.promoted is True
        assert result.decision == "PROMOTED"
        assert mock_export.called
        assert mock_reload.called


def test_champion_challenger_rejection() -> None:
    """Проверка отклонения Challenger, когда его LogLoss хуже Champion."""
    with (
        patch("project.pipeline.mlflow_retrainer.flow.fetch_dwh_data_task") as mock_fetch,
        patch("project.pipeline.mlflow_retrainer.flow.detect_feature_drift_task") as mock_drift,
        patch("project.pipeline.mlflow_retrainer.flow.retrain_models_task") as mock_retrain,
        patch("project.pipeline.mlflow_retrainer.flow.export_onnx_task") as mock_export,
        patch("project.pipeline.mlflow_retrainer.flow.hot_reload_triton_task") as mock_reload,
    ):
        mock_fetch.return_value = {"ref_df": None, "cur_df": None}
        mock_drift.return_value = {"retrain_recommended": True}
        mock_retrain.return_value = {
            "metrics": {
                "champion_loss": 0.1275,
                "challenger_loss": 0.1650,
                "promoted": False,
                "decision": "REJECTED",
                "reason": "challenger_loss_degraded",
            },
            "diff_data": {"features": []},
        }

        result = bot_detection_retrain_flow(force_retrain=True)

        assert result.status == "completed"
        assert result.promoted is False
        assert result.decision == "REJECTED"
        assert not mock_export.called
        assert not mock_reload.called
        assert result.triton_reload is not None
        assert result.triton_reload["skipped"] is True


def test_mlflow_retrainer_dataclasses() -> None:
    """Проверка структуры и методов датаклассов модуля mlflow_retrainer."""
    metrics = GatekeeperMetrics(
        champion_loss=0.12,
        challenger_loss=0.11,
        catboost_val_loss=0.13,
        lightgbm_val_loss=0.12,
        xgboost_val_loss=0.12,
        meta_stacker_val_loss=0.11,
        promoted=True,
        decision=GatekeeperDecision.PROMOTED,
        reason="challenger_improved_champion",
    )
    assert metrics.promoted is True
    assert metrics.decision is GatekeeperDecision.PROMOTED
    assert metrics.decision == "PROMOTED"
    assert metrics.to_dict()["champion_loss"] == 0.12

    eval_result = RetrainEvaluationResult(
        metrics=metrics,
        diff_data={"diff": []},
        selected_features=["f1"],
    )
    assert eval_result.selected_features == ["f1"]
    assert eval_result.to_dict()["metrics"]["decision"] == "PROMOTED"

    flow_res = FlowExecutionResult(
        status=FlowStatus.COMPLETED,
        retrained=True,
        promoted=True,
        decision=GatekeeperDecision.PROMOTED,
        metrics=metrics,
        feature_importance_diff=eval_result.diff_data,
    )
    assert flow_res.status == "completed"
    assert flow_res.status is FlowStatus.COMPLETED
    assert flow_res.decision is GatekeeperDecision.PROMOTED
    assert flow_res.retrained is True
    assert flow_res.to_dict()["decision"] == "PROMOTED"

    from_dict_res = FlowExecutionResult.from_dict({"status": "completed", "decision": "PROMOTED", "retrained": True})
    assert from_dict_res.status is FlowStatus.COMPLETED
    assert from_dict_res.decision is GatekeeperDecision.PROMOTED

    dist_data = FeatureDistributionData(ref_df=None, cur_df=None)
    assert dist_data.ref_df is None
    assert dist_data.cur_df is None
    assert dist_data.to_dict()["ref_df"] is None


def test_model_publisher_versioning_and_rotation(tmp_path: Path) -> None:
    """Проверка создания инкрементальных версий, ротации и строгой синхронизации `feature_store_state.json`."""
    import json
    from types import SimpleNamespace

    vm = TritonModelVersionManager(repository_path=tmp_path, keep_versions=3)
    publisher = ModelPublisher(version_manager=vm)

    (tmp_path / "catboost_detector" / "1").mkdir(parents=True)
    (tmp_path / "catboost_detector" / "2").mkdir(parents=True)
    for m_name, inp_name in (
        ("catboost_detector", "features"),
        ("lightgbm_detector", "float_input"),
        ("xgboost_detector", "float_input"),
        ("bot_ensemble", "features"),
    ):
        (tmp_path / m_name).mkdir(parents=True, exist_ok=True)
        (tmp_path / m_name / "config.pbtxt").write_text(
            f'input [ {{ name: "{inp_name}" dims: [ 95 ] }} ]',
            encoding="utf-8",
        )
    assert publisher.get_existing_versions("catboost_detector") == [1, 2]

    mock_src_dir = tmp_path / "mlflow_download"
    mock_src_dir.mkdir()
    (mock_src_dir / "model.onnx").write_bytes(b"dummy_onnx_bytes")
    state_file = tmp_path / "feature_store_state.json"
    state_file.write_text(json.dumps({"version": 1, "active_features": ["f0", "f1", "f2"]}), encoding="utf-8")

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = SimpleNamespace(run_id="champ_run_1")
    mock_client.get_run.return_value = SimpleNamespace(outputs=None)

    def _download(artifact_uri: str, **_kwargs: object) -> str:
        return str(state_file) if artifact_uri.endswith(".json") else str(mock_src_dir)

    with (
        patch("project.pipeline.serving.publisher.MlflowClient", return_value=mock_client),
        patch("project.tracking.experiment_tracker.loggers.onnx_resolver.MlflowClient", return_value=mock_client),
        patch("mlflow.artifacts.download_artifacts", side_effect=_download),
    ):
        pub_versions = publisher.publish_models()
        assert pub_versions["catboost_detector"] == 3
        assert publisher.get_existing_versions("catboost_detector") == [1, 2, 3]
        assert (tmp_path / "bot_ensemble" / "1" / "feature_store_state.json").is_file()
        assert "dims: [ 3 ]" in (tmp_path / "lightgbm_detector" / "config.pbtxt").read_text(encoding="utf-8")
        assert "dims: [ 3 ]" in (tmp_path / "xgboost_detector" / "config.pbtxt").read_text(encoding="utf-8")

        pub_versions_2 = publisher.publish_models()
        assert pub_versions_2["catboost_detector"] == 4
        assert publisher.get_existing_versions("catboost_detector") == [2, 3, 4]
        assert not (tmp_path / "catboost_detector" / "1").exists()


def test_gatekeeper_fast_mode_deterministic_feature_selection(isolated_mlflow_env: str) -> None:
    """Проверка легковесного режима Fast Mode: быстрый детерминированный отбор фичей и пропуск RuBERT."""
    import polars as pl

    from project.pipeline.mlflow_retrainer.components.data_loader import TrainingBundle
    from project.pipeline.mlflow_retrainer.components.gatekeeper import RetrainingGatekeeper

    mlflow.set_experiment("bot_detection_retraining")

    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 10)).astype(np.float32)
    y = rng.choice([0, 1], size=40)
    feat_names = [f"feature_{i}" for i in range(10)]
    texts_df = pl.DataFrame(
        {
            "dialog_id": np.repeat([f"d_{i}" for i in range(20)], 2).tolist(),
            "participant_index": np.tile([0, 1], 20).tolist(),
            "text": [f"msg_{i}" for i in range(40)],
            "is_bot": y.tolist(),
            "is_pseudo": [False] * 40,
        }
    )

    mock_loader = MagicMock()
    mock_loader.load_retraining_bundle.return_value = TrainingBundle(
        X=X,
        y=y,
        feature_names=feat_names,
        texts_df=texts_df,
        dialogues=[],
        is_pseudo=np.zeros(40, dtype=bool),
    )

    gk = RetrainingGatekeeper(
        data_loader=mock_loader,
        fast_mode=True,
        n_estimators=10,
        feature_keep_ratio=0.70,
    )

    with patch.object(gk, "fetch_champion_loss", return_value=1.5):
        res = gk.retrain_and_evaluate()

    metrics = res.metrics if isinstance(res.metrics, dict) else res.metrics.to_dict()
    assert metrics.get("decision") in (GatekeeperDecision.PROMOTED, GatekeeperDecision.REJECTED, "PROMOTED", "REJECTED")
    assert res.selected_features_by_model is not None
    # Проверяем, что для каждой модели отобрано подмножество фичей (детерминированный прунинг)
    cb_feats = res.selected_features_by_model["catboost"]
    lgb_feats = res.selected_features_by_model["lightgbm"]
    xgb_feats = res.selected_features_by_model["xgboost"]

    assert 5 <= len(cb_feats) <= 10
    assert 5 <= len(lgb_feats) <= 10
    assert 5 <= len(xgb_feats) <= 10
    assert res.diff_data is not None


def test_bot_detection_retrain_flow_with_fast_mode_flags() -> None:
    """Проверка проброса параметров fast_mode и n_estimators через bot_detection_retrain_flow."""
    with (
        patch("project.pipeline.mlflow_retrainer.flow.fetch_dwh_data_task") as mock_fetch,
        patch("project.pipeline.mlflow_retrainer.flow.detect_feature_drift_task") as mock_drift,
        patch("project.pipeline.mlflow_retrainer.flow.retrain_models_task") as mock_retrain,
        patch("project.pipeline.mlflow_retrainer.flow.export_onnx_task"),
        patch("project.pipeline.mlflow_retrainer.flow.hot_reload_triton_task", return_value={"catboost": True}),
    ):
        mock_fetch.return_value = {"ref_df": None, "cur_df": None}
        mock_drift.return_value = {"retrain_recommended": True, "dataset_drift": True}
        mock_retrain.return_value = {
            "metrics": {"promoted": True, "decision": "PROMOTED"},
            "diff_data": {"report": {}},
        }

        flow_res = bot_detection_retrain_flow(
            force_retrain=True,
            fast_mode=True,
            n_estimators=20,
        )

        assert flow_res.status is FlowStatus.COMPLETED
        assert flow_res.retrained is True
        mock_retrain.assert_called_once_with(
            fast_mode=True,
            n_estimators=20,
        )


def test_gatekeeper_regular_vs_fast_hyperparameters() -> None:
    """Проверка канонических ноутбучных настроек Optuna (300/50, 750 итераций) для регулярного и быстрого режимов."""
    mock_loader = MagicMock()

    # 1. Регулярный режим (ноутбучный бюджет: n_trials=300, patience=50, n_estimators=750)
    regular_gk = RetrainingGatekeeper(
        fast_mode=False,
        data_loader=mock_loader,
    )
    assert regular_gk.n_trials == 300
    assert regular_gk.patience == 50
    assert regular_gk.n_estimators == 750
    assert regular_gk.early_stopping_rounds == 40
    assert regular_gk.n_splits == 5
    assert regular_gk.sample_size == 3032
    assert regular_gk.corr_threshold == 0.95

    # 2. Быстрый проверочный режим (меняет только объем данных/итераций, а не логику CV)
    fast_gk = RetrainingGatekeeper(
        fast_mode=True,
        data_loader=mock_loader,
    )
    assert fast_gk.n_trials == 3
    assert fast_gk.patience == 1
    assert fast_gk.n_estimators <= 25
    assert fast_gk.n_splits == 5
    assert fast_gk.sample_size == 300


def test_retrain_flow_skips_when_insufficient_data() -> None:
    """Проверка срабатывания Data Volume Guard (пропуск при N < 100)."""
    with (
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.count_recent_dialogues", return_value=15),
        patch("project.pipeline.mlflow_retrainer.flow.retrain_models_task") as mock_retrain,
    ):
        result = bot_detection_retrain_flow(force_retrain=False)

        assert result.status == "skipped"
        assert result.reason == "insufficient_data"
        assert not mock_retrain.called


def test_fetch_champion_optuna_params_parses_typed_search_space() -> None:
    """Параметры чемпиона и Level-1 ранов парсятся из MLflow-строк и фильтруются под search space Optuna."""
    from types import SimpleNamespace

    from project.pipeline.mlflow_retrainer.components.champion_params import ChampionParamResolver

    champ_run = SimpleNamespace(
        info=SimpleNamespace(run_id="champ_run", experiment_id="7", start_time=200, run_name="logistic_meta_stacker"),
        data=SimpleNamespace(
            tags={
                "level1_catboost_run_id": "cb_run",
                "level1_lightgbm_run_id": "lgb_run",
                "level1_xgboost_run_id": "xgb_run",
            },
            params={"C": "0.07", "fit_intercept": "True", "n_trials": "300", "solver": "lbfgs"},
        ),
    )
    cb_run = SimpleNamespace(
        info=SimpleNamespace(run_id="cb_run", run_name="catboost_detector", start_time=190),
        data=SimpleNamespace(
            params={"learning_rate": "0.036", "depth": "7", "l2_leaf_reg": "4.8", "iterations": "750", "seed": "42"}
        ),
    )
    lgb_run = SimpleNamespace(
        info=SimpleNamespace(run_id="lgb_run", run_name="lightgbm_rubert_hybrid", start_time=180),
        data=SimpleNamespace(params={"learning_rate": "0.02", "num_leaves": "54", "min_child_samples": "42"}),
    )
    xgb_run = SimpleNamespace(
        info=SimpleNamespace(run_id="xgb_run", run_name="xgboost_rubert_hybrid", start_time=170),
        data=SimpleNamespace(params={"max_depth": "6", "learning_rate": "0.015", "subsample": "0.85"}),
    )

    run_map = {"champ_run": champ_run, "cb_run": cb_run, "lgb_run": lgb_run, "xgb_run": xgb_run}
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = SimpleNamespace(run_id="champ_run")
    mock_client.get_run.side_effect = lambda rid: run_map[rid]

    with patch("project.pipeline.mlflow_retrainer.components.champion_params.MlflowClient", return_value=mock_client):
        parsed = ChampionParamResolver.fetch("http://mlflow:5000")

    assert parsed["stacker"] == {"C": 0.07, "fit_intercept": True}
    assert parsed["catboost"] == {"learning_rate": 0.036, "depth": 7, "l2_leaf_reg": 4.8}
    assert parsed["lightgbm"] == {"learning_rate": 0.02, "num_leaves": 54, "min_child_samples": 40}
    assert parsed["xgboost"] == {"max_depth": 6, "learning_rate": 0.015, "subsample": 0.85}

    champ_run.data.tags = {"level1_catboost_run_id": "cb_run"}
    with (
        patch("project.pipeline.mlflow_retrainer.components.champion_params.MlflowClient", return_value=mock_client),
        pytest.raises(FileNotFoundError, match="missing feature_tree.json"),
    ):
        ChampionParamResolver.fetch("http://mlflow:5000")


class _FakeRedisStore:
    def __init__(self, data: dict[str, str]) -> None:
        self.data = data

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def set(self, key: str, value: str, **_kwargs: object) -> None:
        self.data[key] = value

    def mget(self, keys: list[str]) -> list[str | None]:
        return [self.data.get(k) for k in keys]

    def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                removed += 1
        return removed

    def close(self) -> None:
        pass


def test_retrain_flow_syncs_all_green_metrics_on_promotion(monkeypatch: pytest.MonkeyPatch) -> None:
    """После финального промоушена в MLflow/Triton синхронно сбрасываются все drift-ключи в Redis и /metrics."""
    import json

    from project.api.metrics import (
        ADAPTIVE_RETRAIN_STATUS,
        DATA_DRIFT_SHARE,
        MODEL_AGE_DAYS,
        PRODUCTION_DIALOGUES_COUNT,
        metrics_manager,
    )
    from project.drift.concept_drift.prometheus_exporter import ConceptDriftPrometheusExporter
    from project.drift.feature_importance_diff.prometheus_exporter import FeatureImportanceDiffPrometheusExporter
    from project.drift.prediction_drift.prometheus_exporter import PredictionDriftPrometheusExporter

    fake_data: dict[str, str] = {
        "bot_detector:drift_metrics": json.dumps(
            {"drift_share": 0.66, "dataset_drift": True, "column_p_values": {"f1": 0.0}}
        ),
        "bot_detector:prediction_drift_metrics": json.dumps(
            {"psi": 0.35, "mean_entropy": 0.72, "is_drifted": True, "verdict_shares": {"bot": 0.9}}
        ),
        "bot_detector:concept_drift_metrics": json.dumps(
            {"disagreement_rate": 0.41, "feature_rank_correlation": 0.6, "concept_drift_detected": True}
        ),
        "bot_detector:feature_importance_metrics": json.dumps(
            {"features": [{"feature": "fuzzy_echo", "importance_delta": 3.5, "rank_shift": 2}]}
        ),
        "bot_detector:adaptive_decision": json.dumps(
            {"dialogues_count": 200, "model_age_days": 1.5, "retrain_status": 2, "reason": "drift_detected"}
        ),
        "bot_detector:retrain_cooldown_until": "9999999999.0",
    }
    fake_redis = _FakeRedisStore(fake_data)

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(metrics_manager, "_redis_client", None)
    with (
        patch(
            "project.pipeline.mlflow_retrainer.flow.fetch_dwh_data_task", return_value={"ref_df": None, "cur_df": None}
        ),
        patch(
            "project.pipeline.mlflow_retrainer.flow.detect_feature_drift_task",
            return_value={"retrain_recommended": True},
        ),
        patch(
            "project.pipeline.mlflow_retrainer.flow.retrain_models_task",
            return_value={
                "metrics": {"promoted": True, "decision": "PROMOTED", "reason": "challenger_improved_champion"},
                "diff_data": {"features": []},
            },
        ),
        patch("project.pipeline.mlflow_retrainer.flow.export_onnx_task"),
        patch("project.pipeline.mlflow_retrainer.flow.hot_reload_triton_task", return_value={"meta_stacker": True}),
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.trigger_api_gateway_metrics") as mock_sync,
        patch("redis.Redis.from_url", return_value=fake_redis),
    ):
        result = bot_detection_retrain_flow(force_retrain=True)
        metrics_manager.sync_from_redis()

    assert result.promoted is True
    assert mock_sync.call_count == 2
    assert "bot_detector:retrain_in_progress" not in fake_data
    assert "bot_detector:retrain_cooldown_until" not in fake_data
    assert "bot_detector:feature_importance_metrics" in fake_data

    adaptive = json.loads(fake_data["bot_detector:adaptive_decision"])
    assert adaptive == {"dialogues_count": 0, "model_age_days": 0.0, "retrain_status": 0, "reason": "promoted"}

    assert ADAPTIVE_RETRAIN_STATUS._value.get() == 0.0
    assert PRODUCTION_DIALOGUES_COUNT._value.get() == 0.0
    assert MODEL_AGE_DAYS._value.get() == 0.0
    assert DATA_DRIFT_SHARE._value.get() == 0.0
    assert PredictionDriftPrometheusExporter.PREDICTION_DRIFT_PSI._value.get() == 0.0
    assert PredictionDriftPrometheusExporter.PREDICTION_ENTROPY._value.get() == 0.0
    assert ConceptDriftPrometheusExporter.ENSEMBLE_DISAGREEMENT._value.get() == 0.0
    assert len(next(iter(FeatureImportanceDiffPrometheusExporter.FEATURE_IMPORTANCE_DELTA.collect())).samples) >= 1


def test_retrain_flow_keeps_drift_metrics_when_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """При отклонении Challenger drift-метрики в Redis не обнуляются, а лишь выставляется cooldown."""
    import json

    fake_data: dict[str, str] = {
        "bot_detector:drift_metrics": json.dumps({"drift_share": 0.66, "dataset_drift": True}),
        "bot_detector:prediction_drift_metrics": json.dumps({"psi": 0.35, "mean_entropy": 0.72, "is_drifted": True}),
        "bot_detector:adaptive_decision": json.dumps(
            {"dialogues_count": 200, "model_age_days": 1.5, "retrain_status": 2, "reason": "drift_detected"}
        ),
    }
    fake_redis = _FakeRedisStore(fake_data)

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with (
        patch(
            "project.pipeline.mlflow_retrainer.flow.fetch_dwh_data_task", return_value={"ref_df": None, "cur_df": None}
        ),
        patch(
            "project.pipeline.mlflow_retrainer.flow.detect_feature_drift_task",
            return_value={"retrain_recommended": True},
        ),
        patch(
            "project.pipeline.mlflow_retrainer.flow.retrain_models_task",
            return_value={
                "metrics": {"promoted": False, "decision": "REJECTED", "reason": "challenger_loss_degraded"},
                "diff_data": {"features": []},
            },
        ),
        patch("project.pipeline.mlflow_retrainer.flow.export_onnx_task"),
        patch("project.pipeline.mlflow_retrainer.flow.hot_reload_triton_task"),
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.trigger_api_gateway_metrics"),
        patch("redis.Redis.from_url", return_value=fake_redis),
    ):
        result = bot_detection_retrain_flow(force_retrain=True)

    assert result.promoted is False
    assert json.loads(fake_data["bot_detector:drift_metrics"])["drift_share"] == 0.66
    assert json.loads(fake_data["bot_detector:prediction_drift_metrics"])["psi"] == 0.35
    adaptive = json.loads(fake_data["bot_detector:adaptive_decision"])
    assert adaptive["retrain_status"] == 0
    assert adaptive["reason"] == "retrain_completed_cooldown"
    assert "bot_detector:retrain_cooldown_until" in fake_data


def test_fetch_champion_optuna_params_supports_detector_tags(tmp_path: Path) -> None:
    """Теги level1_{key}_detector_run_id и feature_tree.json используются до fallback через search_runs."""
    import json
    from types import SimpleNamespace

    from project.pipeline.mlflow_retrainer.components.champion_params import ChampionParamResolver

    champ_run = SimpleNamespace(
        info=SimpleNamespace(experiment_id="12", start_time=500, run_id="champ_v10", run_name="logistic_meta_stacker"),
        data=SimpleNamespace(
            tags={
                "level1_catboost_detector_run_id": "cb_10",
                "level1_lightgbm_detector_run_id": "lgb_10",
                "level1_xgboost_detector_run_id": "xgb_10",
            },
            params={"C": "0.15", "fit_intercept": "True"},
        ),
    )
    runs_by_id = {
        "champ_v10": champ_run,
        "cb_10": SimpleNamespace(
            info=SimpleNamespace(run_name="catboost_detector"),
            data=SimpleNamespace(params={"learning_rate": "0.04", "depth": "6"}),
        ),
        "lgb_10": SimpleNamespace(
            info=SimpleNamespace(run_name="lightgbm_detector"),
            data=SimpleNamespace(params={"learning_rate": "0.03", "num_leaves": "31"}),
        ),
        "xgb_10": SimpleNamespace(
            info=SimpleNamespace(run_name="xgboost_detector"),
            data=SimpleNamespace(params={"max_depth": "5", "learning_rate": "0.025"}),
        ),
    }

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = SimpleNamespace(run_id="champ_v10")
    mock_client.get_run.side_effect = lambda rid: runs_by_id[rid]

    with patch("project.pipeline.mlflow_retrainer.components.champion_params.MlflowClient", return_value=mock_client):
        parsed = ChampionParamResolver.fetch("http://mlflow:5000")

    mock_client.search_runs.assert_not_called()
    assert parsed["catboost"] == {"learning_rate": 0.04, "depth": 6}
    assert parsed["lightgbm"] == {"learning_rate": 0.03, "num_leaves": 31}
    assert parsed["xgboost"] == {"max_depth": 5, "learning_rate": 0.025}

    # Проверка чтения run_id из feature_tree.json при отсутствии тегов
    tree_dir = tmp_path / "tree_art"
    tree_dir.mkdir()
    tree_path = tree_dir / "feature_tree.json"
    tree_path.write_text(
        json.dumps(
            {
                "level1_base_models": {
                    "catboost_detector": {"run_id": "cb_10"},
                    "lightgbm_detector": {"run_id": "lgb_10"},
                    "xgboost_detector": {"run_id": "xgb_10"},
                }
            }
        ),
        encoding="utf-8",
    )
    champ_no_tags = SimpleNamespace(
        info=SimpleNamespace(experiment_id="12", start_time=500, run_id="champ_tree", run_name="logistic_meta_stacker"),
        data=SimpleNamespace(tags={}, params={"C": "0.2", "fit_intercept": "False"}),
    )
    runs_by_id["champ_tree"] = champ_no_tags
    mock_client2 = MagicMock()
    mock_client2.get_model_version_by_alias.return_value = SimpleNamespace(run_id="champ_tree")
    mock_client2.get_run.side_effect = lambda rid: runs_by_id[rid]
    mock_client2.download_artifacts.return_value = str(tree_path)

    with patch("project.pipeline.mlflow_retrainer.components.champion_params.MlflowClient", return_value=mock_client2):
        parsed_tree = ChampionParamResolver.fetch("http://mlflow:5000")

    mock_client2.search_runs.assert_not_called()
    assert parsed_tree["catboost"] == {"learning_rate": 0.04, "depth": 6}
    assert parsed_tree["stacker"] == {"C": 0.2, "fit_intercept": False}


def test_dataset_splitter_and_pruner_isolate_holdout() -> None:
    """DatasetSplitter изолирует holdout строго на gold-парах, а FeaturePruner удаляет константы и дубли > 0.95."""
    import polars as pl

    from project.pipeline.mlflow_retrainer.components.data_loader import TrainingBundle
    from project.pipeline.mlflow_retrainer.components.dataset_splitter import DatasetSplitter
    from project.pipeline.mlflow_retrainer.stages.feature_selection import FeaturePruner

    rng = np.random.default_rng(7)
    base_col = rng.normal(size=12)
    dup_col = base_col * 2.0 + 1e-6  # |r| == 1.0 > 0.95
    const_col = np.ones(12)
    other_col = rng.normal(size=12)
    X_all = np.column_stack([base_col, dup_col, const_col, other_col]).astype(np.float32)
    y_all = np.array([0, 1] * 6, dtype=np.int64)
    is_pseudo = np.array([False] * 8 + [True] * 4, dtype=bool)
    texts_df = pl.DataFrame(
        {"dialog_id": [f"d{i // 2}" for i in range(12)], "participant_index": [i % 2 for i in range(12)]}
    )
    bundle = TrainingBundle(
        X=X_all,
        y=y_all,
        feature_names=["base", "dup", "const", "other"],
        texts_df=texts_df,
        dialogues=[],
        is_pseudo=is_pseudo,
    )
    split = DatasetSplitter.split_dialogue_pairs(bundle)
    assert len(split.y_tr) == 10
    assert len(split.y_te) == 2
    assert bool(np.any(split.is_pseudo_tr))

    pruner = FeaturePruner(corr_threshold=0.95)
    X_pruned = pruner.fit_transform(split.X_tr, feature_names=bundle.feature_names)
    kept = pruner.get_feature_names_out()
    assert kept == ["base", "other"]
    assert X_pruned.shape == (10, 2)
    assert "const" in pruner.dropped_constant_features_
    assert "dup" in pruner.dropped_correlated_features_


def test_stacking_stage_logs_canonical_metrics_tags_and_feature_store(isolated_mlflow_env: str) -> None:
    """StackingAndPromotionStage пишет канонические метрики (без fi_*), Gatekeeper теги и FeatureStore JSON."""
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from mlflow.tracking import MlflowClient

    from project.pipeline.mlflow_retrainer.stages.ensemble_runner import Level1Results
    from project.pipeline.mlflow_retrainer.stages.feature_selection import SelectedFeatures
    from project.pipeline.mlflow_retrainer.stages.stacking_stage import StackingAndPromotionStage

    rng = np.random.default_rng(11)
    n_tr, n_te, n_feats = 24, 8, 6
    X_tr = rng.normal(size=(n_tr, n_feats)).astype(np.float32)
    y_tr = np.tile([0, 1], n_tr // 2)
    y_te = np.tile([0, 1], n_te // 2)
    cb_full = CatBoostClassifier(iterations=5, depth=3, verbose=0).fit(X_tr, y_tr)
    lgb_full = lgb.LGBMClassifier(n_estimators=5, max_depth=3, verbose=-1).fit(X_tr, y_tr)
    xgb_full = xgb.XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss").fit(X_tr, y_tr)

    l1 = Level1Results(
        level1_runs={"catboost_detector": "cb_r1", "lightgbm_detector": "lgb_r1", "xgboost_detector": "xgb_r1"},
        cb_oof=np.clip(rng.uniform(0.1, 0.9, n_tr), 1e-3, 1 - 1e-3),
        cb_test=np.clip(rng.uniform(0.1, 0.9, n_te), 1e-3, 1 - 1e-3),
        cb_loss=0.14,
        lgb_oof=np.clip(rng.uniform(0.1, 0.9, n_tr), 1e-3, 1 - 1e-3),
        lgb_test=np.clip(rng.uniform(0.1, 0.9, n_te), 1e-3, 1 - 1e-3),
        lgb_loss=0.15,
        xgb_oof=np.clip(rng.uniform(0.1, 0.9, n_tr), 1e-3, 1 - 1e-3),
        xgb_test=np.clip(rng.uniform(0.1, 0.9, n_te), 1e-3, 1 - 1e-3),
        xgb_loss=0.145,
        cb_full_model=cb_full,
        lgb_full_model=lgb_full,
        xgb_full_model=xgb_full,
        cb_feature_importances={"f0": 0.42, "f1": 0.11},
    )
    selected = SelectedFeatures(
        selected_by_model={"catboost": ["f0", "f1"], "lightgbm": ["f0", "f1"], "xgboost": ["f0", "f1"]},
        k_cb=2,
        m_lgb=2,
        p_xgb=2,
        cb_feature_indices=[0, 1],
        lgb_feature_indices=[0, 1],
        xgb_feature_indices=[0, 1],
        total_features=n_feats,
        all_feature_names=[f"f{i}" for i in range(n_feats)],
    )
    stage = StackingAndPromotionStage(
        experiment_name="bot_detection_retraining", n_trials=0, patience=1, n_splits=2, n_estimators=5
    )
    Z_tr, Z_te = stage._logits_from_l1(l1)
    d_ids_tr = np.repeat([f"tr_{i}" for i in range(n_tr // 2)], 2)
    p_idx_tr = np.tile([0, 1], n_tr // 2)
    d_ids_te = np.repeat([f"te_{i}" for i in range(n_te // 2)], 2)
    p_idx_te = np.tile([0, 1], n_te // 2)

    res, gk_metrics, _ = stage.train_and_evaluate_stacker(
        Z_tr=Z_tr,
        Z_te=Z_te,
        y_tr=y_tr,
        groups_tr=d_ids_tr,
        dialog_ids_tr=d_ids_tr,
        participant_indices_tr=p_idx_tr,
        dialog_ids_te=d_ids_te,
        participant_indices_te=p_idx_te,
        y_te=y_te,
        is_pseudo_tr=np.zeros(n_tr, dtype=bool),
        test_ids=np.arange(n_te),
        meta_feature_names=["logit_catboost", "logit_lightgbm", "logit_xgboost"],
        l1=l1,
        selected=selected,
        champion_loss=9.99,
        champion_version="10",
        champion_run_id="champ_r10",
        champion_eval_mode="holdout_onnx",
        feature_store_state={"version": 1, "active_features": selected.all_feature_names},
    )

    assert gk_metrics.promoted is True
    client = MlflowClient(tracking_uri=isolated_mlflow_env)
    run = client.get_run(res.run_id)
    assert {"val_logloss", "val_logloss_raw", "val_roc_auc", "val_accuracy", "val_f1"}.issubset(run.data.metrics)
    assert not any(k.startswith("fi_") for k in run.data.metrics)
    assert "brier_score" not in run.data.metrics
    assert run.data.tags["champion_version"] == "10"
    assert run.data.tags["champion_run_id"] == "champ_r10"
    assert run.data.tags["champion_eval_mode"] == "holdout_onnx"
    assert run.data.tags["level1_catboost_detector_run_id"] == "cb_r1"
    feat_artifacts = {a.path for a in client.list_artifacts(res.run_id, "features")}
    assert "features/feature_store_state.json" in feat_artifacts


def test_champion_evaluator_fetch_importance_from_feature_tree(tmp_path: Path) -> None:
    """ChampionEvaluator.fetch_champion_importance извлекает важности признаков CatBoost из feature_tree.json."""
    import json
    from types import SimpleNamespace

    from project.pipeline.mlflow_retrainer.components.champion_evaluator import ChampionEvaluator

    tree_file = tmp_path / "feature_tree.json"
    tree_file.write_text(
        json.dumps(
            {
                "architecture": "TwoLevelStackingEnsemble",
                "level1_base_models": {
                    "catboost_detector": {
                        "features": [
                            {"name": "echo_all", "importance": 12.5},
                            {"name": "cyr_ratio", "importance": 3.0},
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = SimpleNamespace(version="10", run_id="r10")
    mock_client.download_artifacts.return_value = str(tree_file)

    with (
        patch("project.pipeline.mlflow_retrainer.components.champion_evaluator.MlflowClient", return_value=mock_client),
        patch("mlflow.artifacts.download_artifacts", return_value=str(tree_file)),
    ):
        imp = ChampionEvaluator.fetch_champion_importance("http://mlflow:5000")

    assert imp == {"echo_all": 12.5, "cyr_ratio": 3.0}


def test_champion_evaluator_holdout_uses_runs_uri_onnx(tmp_path: Path) -> None:
    """ChampionEvaluator.evaluate_champion_on_holdout строго загружает `feature_store_state.json` и 4 ONNX."""
    import json
    from types import SimpleNamespace

    from project.pipeline.mlflow_retrainer.components.champion_evaluator import ChampionEvaluator

    models_dir = tmp_path / "downloaded"
    for name in ("catboost_detector", "lightgbm_detector", "xgboost_detector", "meta_stacker"):
        mdir = models_dir / name
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "model.onnx").write_bytes(b"dummy-onnx")

    state_file = tmp_path / "feature_store_state.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "active_features": ["is_last_sender", "n_messages", "msg_share", "max_consecutive_msgs"],
            }
        ),
        encoding="utf-8",
    )

    def _mock_download_artifacts(
        *, artifact_uri: str, tracking_uri: str | None = None, dst_path: str | None = None
    ) -> str:
        _ = (tracking_uri, dst_path)
        if artifact_uri.endswith("feature_store_state.json"):
            return str(state_file)
        model_name = artifact_uri.rsplit("/", maxsplit=1)[-1]
        return str(models_dir / model_name)

    def _make_session(path_str: str, **_kwargs: object) -> MagicMock:
        sess = MagicMock()
        if "meta_stacker" in path_str:
            sess.get_inputs.return_value = [SimpleNamespace(name="float_input", shape=[None, 3])]
            probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.85, 0.15], [0.15, 0.85]], dtype=np.float32)
        else:
            sess.get_inputs.return_value = [SimpleNamespace(name="float_input", shape=[None, 4])]
            probs = np.array([[0.8, 0.2], [0.2, 0.8], [0.75, 0.25], [0.25, 0.75]], dtype=np.float32)
        sess.run.return_value = [None, probs]
        return sess

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = SimpleNamespace(version="10", run_id="champ_r10")
    mock_client.get_run.return_value = SimpleNamespace(
        outputs=None, data=SimpleNamespace(metrics={"val_logloss": 0.20})
    )

    X_te = np.ones((4, 4), dtype=np.float32)
    y_te = np.array([0, 1, 0, 1], dtype=np.int64)
    dlgs = [
        [{"participant_index": 0, "text": "a"}, {"participant_index": 1, "text": "b"}],
        [{"participant_index": 0, "text": "c"}, {"participant_index": 1, "text": "d"}],
    ]

    with (
        patch("project.pipeline.mlflow_retrainer.components.champion_evaluator.MlflowClient", return_value=mock_client),
        patch("project.tracking.experiment_tracker.loggers.onnx_resolver.MlflowClient", return_value=mock_client),
        patch(
            "project.tracking.experiment_tracker.loggers.onnx_resolver.mlflow.artifacts.download_artifacts",
            side_effect=_mock_download_artifacts,
        ) as mock_dl,
        patch("onnxruntime.InferenceSession", side_effect=_make_session),
    ):
        loss = ChampionEvaluator.evaluate_champion_on_holdout(
            X_te=X_te,
            y_te=y_te,
            dialogues_te=dlgs,
            tracking_uri="http://mlflow:5000",
        )

    assert isinstance(loss, float)
    assert 0.0 < loss < 0.5
    downloaded_uris = [
        c.kwargs["artifact_uri"] for c in mock_dl.call_args_list if not str(c.kwargs["artifact_uri"]).endswith(".json")
    ]
    assert downloaded_uris == [
        "runs:/champ_r10/catboost_detector",
        "runs:/champ_r10/lightgbm_detector",
        "runs:/champ_r10/xgboost_detector",
        "runs:/champ_r10/meta_stacker",
    ]


def test_retrain_flow_force_retrain_ignores_drift_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """При force_retrain=True ошибка недостатка chat_logs на этапе дрейфа не валит дообучение."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    fake_redis = _FakeRedisStore({})
    with (
        patch(
            "project.pipeline.mlflow_retrainer.flow.fetch_dwh_data_task",
            side_effect=RuntimeError("ClickHouse chat_logs contains only 0 dialogues"),
        ),
        patch(
            "project.pipeline.mlflow_retrainer.flow.retrain_models_task",
            return_value={
                "metrics": {"promoted": False, "decision": "REJECTED", "reason": "no_gain"},
                "diff_data": {"features": []},
            },
        ),
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.trigger_api_gateway_metrics"),
        patch("redis.Redis.from_url", return_value=fake_redis),
    ):
        result = bot_detection_retrain_flow(force_retrain=True)

    assert result.status == FlowStatus.COMPLETED
    assert result.promoted is False
    assert isinstance(result.drift_summary, dict)
    assert result.drift_summary["reason"] == "skipped_on_force_retrain"


def test_gatekeeper_fetches_champion_importance_before_promotion() -> None:
    """Проверка, что важность признаков старого @champion читается ДО промоушена новой модели."""
    from project.pipeline.mlflow_retrainer.components.gatekeeper import RetrainingGatekeeper
    from project.pipeline.mlflow_retrainer.stages.ensemble_runner import Level1Results
    from project.pipeline.mlflow_retrainer.stages.feature_selection import SelectedFeatures

    gk = RetrainingGatekeeper(fast_mode=True, n_trials=0, n_estimators=2, dwh=MagicMock())
    call_order: list[str] = []
    old_importance = {"f1": 10.0, "f2": 2.0}

    def _fake_fetch_importance() -> dict[str, float]:
        call_order.append("fetch_champion_importance")
        return old_importance if "train_and_evaluate_stacker" not in call_order else {"f1": 99.0}

    def _fake_train_stacker(**_kwargs: Any) -> tuple[Any, Any, Any]:
        call_order.append("train_and_evaluate_stacker")
        return (
            MagicMock(oof_calibrated=np.array([0.2, 0.8]), run_id="new_run"),
            MagicMock(promoted=True, decision="PROMOTED", reason="improved"),
            {},
        )

    def _fake_analyze_drift(
        *, l1: Any, reference_importance: dict[str, float], **_kwargs: Any
    ) -> tuple[dict[str, Any], Any]:
        call_order.append("analyze_drift")
        assert reference_importance == old_importance
        return {"features": [{"feature": "f1", "importance_delta": -5.0}]}, None

    fake_split = MagicMock(
        tr_dlgs=[],
        te_dlgs=[],
        y_tr=np.array([0, 1]),
        y_te=np.array([0, 1]),
        groups_tr=np.array([0, 1]),
        dialog_ids_tr=np.array(["d0", "d1"]),
        participant_indices_tr=np.array([0, 1]),
        dialog_ids_te=np.array(["d2", "d3"]),
        participant_indices_te=np.array([0, 1]),
        is_pseudo_tr=np.array([False, False]),
    )
    fake_l1 = Level1Results(
        cb_oof=np.array([0.1, 0.9]),
        cb_test=np.array([0.1, 0.9]),
        lgb_oof=np.array([0.1, 0.9]),
        lgb_test=np.array([0.1, 0.9]),
        xgb_oof=np.array([0.1, 0.9]),
        xgb_test=np.array([0.1, 0.9]),
        cb_loss=0.2,
        lgb_loss=0.2,
        xgb_loss=0.2,
        cb_feature_importances={"f1": 5.0, "f2": 7.0},
        level1_runs={},
    )
    fake_selected = SelectedFeatures(
        selected_by_model={"catboost": ["f1", "f2"], "lightgbm": ["f1", "f2"], "xgboost": ["f1", "f2"]},
        cb_feature_indices=[0, 1],
        lgb_feature_indices=[0, 1],
        xgb_feature_indices=[0, 1],
        k_cb=2,
        m_lgb=2,
        p_xgb=2,
    )

    with (
        patch.object(gk, "_load_bundle", return_value=MagicMock(feature_names=["f1", "f2"])),
        patch(
            "project.pipeline.mlflow_retrainer.components.gatekeeper.DatasetSplitter.split_dialogue_pairs",
            return_value=fake_split,
        ),
        patch.object(
            gk, "_prepare_pruned_features", return_value=(np.zeros((2, 2)), np.zeros((2, 2)), ["f1", "f2"], {})
        ),
        patch(
            "project.pipeline.mlflow_retrainer.components.gatekeeper.ChampionEvaluator.resolve_champion_info",
            return_value=None,
        ),
        patch.object(gk, "fetch_champion_loss", return_value=0.5),
        patch.object(gk, "fetch_champion_importance", side_effect=_fake_fetch_importance),
        patch(
            "project.pipeline.mlflow_retrainer.components.gatekeeper.FeatureSelectionStage.compute_rankings",
            return_value={},
        ),
        patch(
            "project.pipeline.mlflow_retrainer.components.gatekeeper.FeatureSelectionStage.select_features",
            return_value=fake_selected,
        ),
        patch("project.pipeline.mlflow_retrainer.components.gatekeeper.ChampionParamResolver.fetch", return_value={}),
        patch("project.pipeline.mlflow_retrainer.components.gatekeeper.Level1Stage.run", return_value=fake_l1),
        patch(
            "project.pipeline.mlflow_retrainer.components.gatekeeper.StackingAndPromotionStage.load_meta_matrices",
            return_value=(
                np.zeros((2, 3)),
                np.zeros((2, 3)),
                np.array([0, 1]),
                np.array([0, 1]),
                np.array(["d0", "d1"]),
                np.array([0, 1]),
                np.array([False, False]),
            ),
        ),
        patch(
            "project.pipeline.mlflow_retrainer.components.gatekeeper.StackingAndPromotionStage.train_and_evaluate_stacker",
            side_effect=_fake_train_stacker,
        ),
        patch(
            "project.pipeline.mlflow_retrainer.components.gatekeeper.StackingAndPromotionStage.analyze_drift",
            side_effect=_fake_analyze_drift,
        ),
    ):
        res = gk.retrain_and_evaluate()

    assert call_order == ["fetch_champion_importance", "train_and_evaluate_stacker", "analyze_drift"]
    assert res.diff_data["features"][0]["importance_delta"] == -5.0
