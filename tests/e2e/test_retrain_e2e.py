"""Сквозные (End-to-End) тесты еженедельного пайплайна переобучения (Prefect + MLflow + ONNX + Triton).

Проверяют полный сквозной цикл:
1. Реальную регистрацию эксперимента `bot_detection_retraining` и запись ранов в MLflow.
2. Полноценное переобучение всех 3 градиентных бустингов (CatBoost, LightGBM, XGBoost)
   с индивидуальным отбором признаков (Feature Pruning) через Optuna для каждой модели.
3. Обучение мета-стекера второго уровня (TwoLevelStackingClassifier) над логитами OOF-предсказаний
   и условным промоушеном только при улучшении валидационного LogLoss.
4. Компиляцию и публикацию всех 4 ONNX-графов (`catboost_detector`, `lightgbm_detector`,
   `xgboost_detector`, `meta_stacker`) в Triton Model Repository и проверку инференса через ONNX Runtime.
5. Сценарий отклонения (Rejection): когда Challenger хуже Champion, продуктовые модели не перезаписываются.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from project.infrastructure.triton.version_manager import TritonModelVersionManager
from project.pipeline.mlflow_retrainer import (
    FeatureDistributionData,
    FlowStatus,
    GatekeeperDecision,
    GatekeeperMetrics,
    bot_detection_retrain_flow,
)
from project.pipeline.mlflow_retrainer.components.data_loader import TrainingDataLoader
from project.pipeline.mlflow_retrainer.components.gatekeeper import RetrainingGatekeeper
from project.pipeline.serving import ModelPublisher, TritonReloader

pytestmark = [pytest.mark.xdist_group("prefect"), pytest.mark.usefixtures("prefect_isolated_server")]


def test_e2e_retrain_promotion_with_mlflow_and_onnx(
    isolated_mlflow_env: str,
    synthetic_training_loader: TrainingDataLoader,
    tmp_path: Path,
) -> None:
    """Полный E2E тест: 3 GBDT с прунингом -> Meta-Stacker -> MLflow -> 4 ONNX в Triton -> Hot-Reload."""
    triton_repo = tmp_path / "triton_models"
    triton_repo.mkdir(parents=True, exist_ok=True)
    for m_name, inp_name in (
        ("catboost_detector", "features"),
        ("lightgbm_detector", "float_input"),
        ("xgboost_detector", "float_input"),
        ("bot_ensemble", "features"),
    ):
        (triton_repo / m_name).mkdir(parents=True, exist_ok=True)
        (triton_repo / m_name / "config.pbtxt").write_text(
            f'input [ {{ name: "{inp_name}" dims: [ 95 ] }} ]',
            encoding="utf-8",
        )
    version_manager = TritonModelVersionManager(repository_path=triton_repo, keep_versions=3)
    publisher = ModelPublisher(version_manager=version_manager)

    rng = np.random.default_rng(100)
    ref_df = pd.DataFrame(rng.normal(0.0, 1.0, size=(50, 8)), columns=[f"f_{i}" for i in range(8)])
    cur_df = pd.DataFrame(rng.normal(5.0, 1.5, size=(50, 8)), columns=[f"f_{i}" for i in range(8)])

    real_gatekeeper = RetrainingGatekeeper(
        n_trials=3,
        data_loader=synthetic_training_loader,
        n_estimators=15,
    )

    reloaded_models: list[str] = []

    def fake_reload_model(model_name: str) -> bool:
        reloaded_models.append(model_name)
        return True

    with (
        patch("project.pipeline.mlflow_retrainer.flow.fetch_dwh_data_task") as mock_fetch,
        patch(
            "project.pipeline.mlflow_retrainer.tasks.RetrainingGatekeeper",
            return_value=real_gatekeeper,
        ),
        patch(
            "project.pipeline.serving.tasks.ModelPublisher",
            return_value=publisher,
        ),
        patch.object(TritonReloader, "reload_model", side_effect=fake_reload_model),
        patch("project.drift.data_drift.fetcher.DWHFeatureFetcher.count_recent_dialogues", return_value=200),
    ):
        mock_fetch.return_value = FeatureDistributionData(ref_df=ref_df, cur_df=cur_df)

        with patch.object(real_gatekeeper, "fetch_champion_loss", return_value=0.85):
            flow_result = bot_detection_retrain_flow(
                force_retrain=False,
                drift_threshold=0.20,
            )

    # 1. Проверяем статус выполнения E2E Flow
    assert flow_result.status is FlowStatus.COMPLETED
    assert flow_result.retrained is True
    assert flow_result.promoted is True
    assert flow_result.decision is GatekeeperDecision.PROMOTED
    assert flow_result.metrics is not None
    assert isinstance(flow_result.metrics, GatekeeperMetrics)
    assert flow_result.metrics.challenger_loss < 0.85

    # 2. Проверяем, что раны и артефакты реально записаны в MLflow Experiment в каноническом формате
    client = MlflowClient(tracking_uri=isolated_mlflow_env)
    exp = client.get_experiment_by_name("bot_detection_retraining")
    assert exp is not None
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.run_name = 'logistic_meta_stacker'",
    )
    assert len(runs) == 1

    latest_run = runs[0]
    for metric_key in (
        "val_logloss",
        "val_logloss_raw",
        "val_roc_auc",
        "val_accuracy",
        "val_f1",
        "execution_time_seconds",
    ):
        assert metric_key in latest_run.data.metrics
    assert not any(k.startswith("fi_") for k in latest_run.data.metrics)
    assert "brier_score" not in latest_run.data.metrics
    assert latest_run.data.tags.get("decision") == "PROMOTED"
    assert latest_run.data.tags.get("promoted") == "True"
    assert "level1_catboost_detector_run_id" in latest_run.data.tags

    tree_path = client.download_artifacts(latest_run.info.run_id, "feature_tree.json")
    with open(tree_path, encoding="utf-8") as f:
        tree_data = json.load(f)
    base_nodes = tree_data["level1_base_models"]
    k_cb = int(base_nodes["catboost_detector"]["n_features"])
    m_lgb = int(base_nodes["lightgbm_detector"]["n_features"])
    p_xgb = int(base_nodes["xgboost_detector"]["n_features"])
    assert k_cb > 0
    assert m_lgb > 0
    assert p_xgb > 0

    graphics_arts = {a.path for a in client.list_artifacts(latest_run.info.run_id, "graphics")}
    assert graphics_arts == {
        "graphics/confusion_matrix.png",
        "graphics/roc_curve.png",
        "graphics/calibration_curve.png",
        "graphics/prediction_distribution.png",
    }

    # 3. Проверяем регистрацию модели в MLflow Model Registry и алиас @champion
    prod_mv = client.get_model_version_by_alias("bot_detector", "champion")
    assert prod_mv is not None

    # 4. Проверяем, что все 4 ONNX-модели (3 бустинга + мета-стекер) опубликованы в Triton и валидны
    expected_models = ["catboost_detector", "lightgbm_detector", "xgboost_detector", "meta_stacker"]
    for model_name in expected_models:
        versions = version_manager.get_versions(model_name)
        assert versions == [1], f"Ожидалась версия [1] для {model_name}, получено {versions}"
        onnx_path = triton_repo / model_name / "1" / "model.onnx"
        assert onnx_path.exists()
        onnx_proto = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_proto)

    # Проверяем реальный инференс через ONNX Runtime для каждого из 3 бустингов (полный вход -> Gather) и мета-стекера
    cb_sess = ort.InferenceSession(
        str(triton_repo / "catboost_detector" / "1" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    total_feats = int(cb_sess.get_inputs()[0].shape[1])
    assert k_cb <= total_feats
    assert m_lgb <= total_feats
    assert p_xgb <= total_feats

    cb_out = cb_sess.run(None, {"features": np.random.randn(4, total_feats).astype(np.float32)})
    assert cb_out[1].shape == (4, 2)

    lgb_sess = ort.InferenceSession(
        str(triton_repo / "lightgbm_detector" / "1" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    lgb_out = lgb_sess.run(None, {"float_input": np.random.randn(4, total_feats).astype(np.float32)})
    assert lgb_out[1].shape == (4, 2)

    xgb_sess = ort.InferenceSession(
        str(triton_repo / "xgboost_detector" / "1" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    xgb_out = xgb_sess.run(None, {"float_input": np.random.randn(4, total_feats).astype(np.float32)})
    assert xgb_out[1].shape == (4, 2)

    meta_sess = ort.InferenceSession(
        str(triton_repo / "meta_stacker" / "1" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    meta_out = meta_sess.run(None, {"meta_input": np.random.randn(4, 3).astype(np.float32)})
    assert meta_out[1].shape == (4, 2)

    # 5. Проверяем, что Triton Hot-Reload был вызван для всех моделей
    assert set(expected_models).issubset(set(reloaded_models))


def test_e2e_retrain_rejection_preserves_champion(
    isolated_mlflow_env: str,
    synthetic_training_loader: TrainingDataLoader,
    tmp_path: Path,
) -> None:
    """Проверка E2E сценария, когда Challenger не обыгрывает Champion: модели Triton не обновляются."""
    triton_repo = tmp_path / "triton_models_reject"
    triton_repo.mkdir(parents=True, exist_ok=True)
    version_manager = TritonModelVersionManager(repository_path=triton_repo, keep_versions=3)
    publisher = ModelPublisher(version_manager=version_manager)

    real_gatekeeper = RetrainingGatekeeper(
        n_trials=2,
        data_loader=synthetic_training_loader,
        n_estimators=10,
    )

    reloaded_models: list[str] = []

    def fake_reload(model_name: str) -> bool:
        reloaded_models.append(model_name)
        return True

    with (
        patch("project.pipeline.mlflow_retrainer.flow.fetch_dwh_data_task") as mock_fetch,
        patch("project.pipeline.mlflow_retrainer.flow.detect_feature_drift_task") as mock_drift,
        patch(
            "project.pipeline.mlflow_retrainer.tasks.RetrainingGatekeeper",
            return_value=real_gatekeeper,
        ),
        patch(
            "project.pipeline.serving.tasks.ModelPublisher",
            return_value=publisher,
        ),
        patch.object(TritonReloader, "reload_model", side_effect=fake_reload),
    ):
        mock_fetch.return_value = FeatureDistributionData(ref_df=None, cur_df=None)
        mock_drift.return_value = {"dataset_drift": True, "drift_share": 0.5, "retrain_recommended": True}

        with patch.object(real_gatekeeper, "fetch_champion_loss", return_value=0.0001):
            flow_result = bot_detection_retrain_flow(
                force_retrain=True,
                drift_threshold=0.20,
            )

    assert flow_result.status is FlowStatus.COMPLETED
    assert flow_result.retrained is True
    assert flow_result.promoted is False
    assert flow_result.decision is GatekeeperDecision.REJECTED
    assert flow_result.triton_reload is not None
    assert flow_result.triton_reload["skipped"] is True

    assert reloaded_models == []
    for model_name in ("catboost_detector", "lightgbm_detector", "xgboost_detector", "meta_stacker"):
        assert version_manager.get_versions(model_name) == []

    client = MlflowClient(tracking_uri=isolated_mlflow_env)
    exp = client.get_experiment_by_name("bot_detection_retraining")
    assert exp is not None
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert any(r.data.tags.get("decision") == "REJECTED" for r in runs)
