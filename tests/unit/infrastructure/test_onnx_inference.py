from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from project.core.config import config
from project.core.utils import LogitTransformer
from project.features.store import FeatureStore
from project.infrastructure.triton.version_manager import TritonModelVersionManager


@pytest.fixture(scope="module")
def onnx_sessions():
    triton_repo = Path(config.triton.model_repository)
    vm = TritonModelVersionManager(repository_path=triton_repo)

    def _latest_onnx(name: str) -> Path:
        ver = vm.get_latest_version(name)
        assert ver is not None, f"Missing versions for {name} in {triton_repo}"
        path = triton_repo / name / str(ver) / "model.onnx"
        assert path.exists(), f"Missing {path}"
        return path

    cb_path = _latest_onnx("catboost_detector")
    lgb_path = _latest_onnx("lightgbm_detector")
    xgb_path = _latest_onnx("xgboost_detector")
    meta_path = _latest_onnx("meta_stacker")

    for p in (cb_path, lgb_path, xgb_path):
        proto = onnx.load(str(p))
        assert any(node.op_type == "Gather" for node in proto.graph.node), f"Missing Gather selector in {p}"

    sess_cb = ort.InferenceSession(str(cb_path))
    sess_lgb = ort.InferenceSession(str(lgb_path))
    sess_xgb = ort.InferenceSession(str(xgb_path))
    sess_meta = ort.InferenceSession(str(meta_path))

    return {"catboost": sess_cb, "lightgbm": sess_lgb, "xgboost": sess_xgb, "meta": sess_meta}


def test_onnx_sessions_load(onnx_sessions):
    """Проверка успешной загрузки всех ONNX сессий и согласованности размерностей входов."""
    cb_dim = onnx_sessions["catboost"].get_inputs()[0].shape[1]
    assert isinstance(cb_dim, int)
    assert cb_dim >= 5
    assert onnx_sessions["lightgbm"].get_inputs()[0].shape[1] == cb_dim
    assert onnx_sessions["xgboost"].get_inputs()[0].shape[1] == cb_dim
    assert onnx_sessions["meta"].get_inputs()[0].shape[1] == 3


def test_onnx_pipeline_inference(onnx_sessions):
    """Сквозная проверка инференса признаков диалога через ONNX пайплайн с 3-входовым мета-стекером."""
    cb_dim = int(onnx_sessions["catboost"].get_inputs()[0].shape[1])
    store = FeatureStore(auto_load_from_repo=True)
    sample_dialogue = [
        {"participant_index": 0, "text": "Привет! Ты настоящий?"},
        {"participant_index": 1, "text": "Конечно, я человек! А ты?"},
        {"participant_index": 0, "text": "Я тоже, давай поболтаем."},
    ]

    assert store.active_features is not None, "feature_store_state.json must be synced alongside ONNX models"
    features = store.transform([sample_dialogue])
    assert features.shape == (2, cb_dim)
    features_f32 = features.astype(np.float32)

    cb_input_name = onnx_sessions["catboost"].get_inputs()[0].name
    cb_probs = onnx_sessions["catboost"].run(None, {cb_input_name: features_f32})[1]

    lgb_input_name = onnx_sessions["lightgbm"].get_inputs()[0].name
    lgb_probs = onnx_sessions["lightgbm"].run(None, {lgb_input_name: features_f32})[1]

    xgb_input_name = onnx_sessions["xgboost"].get_inputs()[0].name
    xgb_probs = onnx_sessions["xgboost"].run(None, {xgb_input_name: features_f32})[1]

    meta_input = np.column_stack(
        [
            LogitTransformer.transform(cb_probs[:, 1]),
            LogitTransformer.transform(lgb_probs[:, 1]),
            LogitTransformer.transform(xgb_probs[:, 1]),
        ]
    ).astype(np.float32)

    meta_input_name = onnx_sessions["meta"].get_inputs()[0].name
    meta_probs = onnx_sessions["meta"].run(None, {meta_input_name: meta_input})[1]

    assert meta_probs.shape == (2, 2)
    assert np.all(meta_probs >= 0.0)
    assert np.all(meta_probs <= 1.0)
    np.testing.assert_allclose(np.sum(meta_probs, axis=1), [1.0, 1.0], rtol=1e-5)
