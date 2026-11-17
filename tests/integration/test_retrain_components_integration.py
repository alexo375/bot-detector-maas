"""Интеграционные тесты взаимодействия компонентов еженедельного дообучения.

Проверяют:
1. Связку `RetrainingGatekeeper` + `OnnxModelLogger`: обучение 3 бустингов (CatBoost, LightGBM, XGBoost)
   с индивидуальным отбором признаков через Optuna, обучение мета-стекера и компиляцию
   всех 4 моделей в валидные ONNX-графы с соответствующими размерностями входов.
2. Связку `RetrainingGatekeeper` (промоушн в MLflow `LoggedModels`) + `ModelPublisher` +
   `TritonModelVersionManager`: реальный экспорт и загрузку 4 ONNX-моделей из MLflow 3 в Triton.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import polars as pl
from mlflow.tracking import MlflowClient

from project.infrastructure.triton.version_manager import TritonModelVersionManager
from project.pipeline.mlflow_retrainer.components.data_loader import TrainingBundle, TrainingDataLoader
from project.pipeline.mlflow_retrainer.components.gatekeeper import RetrainingGatekeeper
from project.pipeline.mlflow_retrainer.schemas import GatekeeperMetrics
from project.pipeline.serving.publisher import ModelPublisher
from project.tracking.experiment_tracker.loggers.onnx_model import OnnxModelLogger


def test_integration_gatekeeper_and_onnx_logger_three_gbdts_and_stacker(isolated_mlflow_env: str) -> None:
    """Интеграционный тест связки RetrainingGatekeeper + OnnxModelLogger (без публикации в Triton)."""
    _ = isolated_mlflow_env
    rng = np.random.default_rng(7)
    n_samples, n_features = 48, 12
    y = rng.integers(0, 2, size=n_samples).astype(np.int32)
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    for i in range(3):
        X[:, i] += (y * 2.0 - 1.0).astype(np.float32)
    feature_names = [f"f_{i}" for i in range(n_features)]

    n_pairs = n_samples // 2
    texts_df = pl.DataFrame(
        {
            "dialog_id": np.repeat([f"int_dlg_{i}" for i in range(n_pairs)], 2).tolist(),
            "participant_index": np.tile([0, 1], n_pairs).tolist(),
            "text": [f"msg_{i}" for i in range(n_samples)],
            "is_bot": y.tolist(),
            "is_pseudo": [False] * n_samples,
        }
    )
    mock_loader = MagicMock(spec=TrainingDataLoader)
    mock_loader.load_retraining_bundle.return_value = TrainingBundle(
        X=X,
        y=y,
        feature_names=feature_names,
        texts_df=texts_df,
        dialogues=[],
        is_pseudo=np.zeros(n_samples, dtype=bool),
    )

    gatekeeper = RetrainingGatekeeper(
        n_trials=2,
        data_loader=mock_loader,
        n_estimators=10,
    )

    captured_onnx: dict[str, onnx.ModelProto] = {}

    def capture_log_onnx(
        _self_logger: OnnxModelLogger,
        model: onnx.ModelProto,
        artifact_path: str,
        registered_model_name: str | None = None,
    ) -> MagicMock:
        _ = registered_model_name
        onnx.checker.check_model(model)
        captured_onnx[artifact_path] = model
        return MagicMock()

    with (
        patch.object(gatekeeper, "fetch_champion_loss", return_value=0.90),
        patch.object(gatekeeper, "fetch_champion_importance", return_value={}),
        patch.object(OnnxModelLogger, "log_onnx_model", autospec=True, side_effect=capture_log_onnx),
    ):
        result = gatekeeper.retrain_and_evaluate()

    assert result.selected_features_by_model is not None
    assert set(result.selected_features_by_model.keys()) == {"catboost", "lightgbm", "xgboost"}
    for m_key in ("catboost", "lightgbm", "xgboost"):
        assert 5 <= len(result.selected_features_by_model[m_key]) <= n_features

    assert set(captured_onnx.keys()) == {
        "catboost_detector",
        "lightgbm_detector",
        "xgboost_detector",
        "meta_stacker",
    }
    for gbdt_key in ("catboost_detector", "lightgbm_detector", "xgboost_detector"):
        model_proto = captured_onnx[gbdt_key]
        assert model_proto.graph.input[0].type.tensor_type.shape.dim[1].dim_value == n_features
        assert any(node.op_type == "Gather" for node in model_proto.graph.node)
    assert captured_onnx["meta_stacker"].graph.input[0].type.tensor_type.shape.dim[1].dim_value == 3


def test_integration_stacker_promotion_and_publisher_with_version_manager(
    isolated_mlflow_env: str,
    synthetic_training_loader: TrainingDataLoader,
    tmp_path: Path,
) -> None:
    """Интеграционный тест: RetrainingGatekeeper (MLflow LoggedModels) -> ModelPublisher -> Triton."""
    gatekeeper = RetrainingGatekeeper(
        n_trials=2,
        data_loader=synthetic_training_loader,
        n_estimators=10,
    )
    with patch.object(gatekeeper, "fetch_champion_loss", return_value=0.95):
        eval_result = gatekeeper.retrain_and_evaluate()

    assert isinstance(eval_result.metrics, GatekeeperMetrics)
    assert eval_result.metrics.promoted is True
    client = MlflowClient(tracking_uri=isolated_mlflow_env)
    exp = client.get_experiment_by_name("bot_detection_retraining")
    assert exp is not None
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.run_name = 'logistic_meta_stacker'",
    )
    assert len(runs) == 1
    meta_run = runs[0]
    assert meta_run.data.tags.get("promoted") == "True"
    assert "val_logloss" in meta_run.data.metrics
    assert not any(k.startswith("fi_") for k in meta_run.data.metrics)

    # Проверяем реальную загрузку 4 ONNX моделей из MLflow 3 (LoggedModels) в версионированный Triton
    triton_repo = tmp_path / "triton_int_repo"
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
    vm = TritonModelVersionManager(repository_path=triton_repo, keep_versions=2)
    publisher = ModelPublisher(version_manager=vm, tracking_uri=isolated_mlflow_env)
    published = publisher.publish_models()

    assert published == {
        "catboost_detector": 1,
        "lightgbm_detector": 1,
        "xgboost_detector": 1,
        "meta_stacker": 1,
    }
    for m_name in published:
        onnx_file = triton_repo / m_name / "1" / "model.onnx"
        assert onnx_file.exists()
        onnx.checker.check_model(onnx.load(str(onnx_file)))
