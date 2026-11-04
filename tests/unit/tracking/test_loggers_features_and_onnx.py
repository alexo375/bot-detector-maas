"""Модульные тесты для утилит извлечения важности признаков и ONNX-логера."""

from unittest.mock import MagicMock

import numpy as np
from onnx import TensorProto, helper

from project.tracking.experiment_tracker.loggers.features import FeatureImportanceExtractor
from project.tracking.experiment_tracker.loggers.onnx_model import OnnxModelLogger


def test_extract_model_importances_ndarray() -> None:
    """Проверка извлечения, когда вход уже является numpy-массивом."""
    arr = np.array([0.1, 0.4, 0.5])
    res = FeatureImportanceExtractor.extract(arr)
    assert res is not None
    assert np.array_equal(res, arr)


def test_extract_model_importances_feature_importances_attr() -> None:
    """Проверка извлечения через атрибут feature_importances_."""
    model = MagicMock()
    model.feature_importances_ = np.array([0.3, 0.7])
    del model.get_feature_importance
    res = FeatureImportanceExtractor.extract(model)
    assert res is not None
    assert np.array_equal(res, np.array([0.3, 0.7]))


def test_extract_model_importances_catboost_method() -> None:
    """Проверка извлечения через метод get_feature_importance()."""
    model = MagicMock(spec=["get_feature_importance"])
    model.get_feature_importance.return_value = np.array([0.2, 0.8])
    res = FeatureImportanceExtractor.extract(model)
    assert res is not None
    assert np.array_equal(res, np.array([0.2, 0.8]))


def test_extract_model_importances_linear_coef() -> None:
    """Проверка извлечения через атрибут coef_ (1D и 2D веса)."""
    # 1D coef
    model_1d = MagicMock(spec=["coef_"])
    model_1d.coef_ = np.array([-0.5, 0.8])
    res_1d = FeatureImportanceExtractor.extract(model_1d)
    assert res_1d is not None
    assert np.allclose(res_1d, np.array([0.5, 0.8]))

    # 2D coef (например LogisticRegression multiclass/binary)
    model_2d = MagicMock(spec=["coef_"])
    model_2d.coef_ = np.array([[-0.2, 0.9]])
    res_2d = FeatureImportanceExtractor.extract(model_2d)
    assert res_2d is not None
    assert np.allclose(res_2d, np.array([0.2, 0.9]))


def test_extract_model_importances_unsupported() -> None:
    """Проверка возврата None для неподдерживаемых объектов."""
    assert FeatureImportanceExtractor.extract("non_model_object") is None


def test_onnx_model_logger_strip_zipmap() -> None:
    """Проверка удаления узла ZipMap и нормализации выходного тензора вероятностей."""
    logger = OnnxModelLogger(n_features=10)

    # Создаем минимальный ONNX-граф с узлом ZipMap
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 10])
    zipmap_out = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 2])

    dummy_node = helper.make_node("Identity", inputs=["X"], outputs=["raw_probs"])
    zipmap_node = helper.make_node("ZipMap", inputs=["raw_probs"], outputs=["probabilities"])

    graph = helper.make_graph(
        nodes=[dummy_node, zipmap_node],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[zipmap_out],
    )
    model = helper.make_model(graph)

    # Вызываем strip_zipmap
    stripped = logger.strip_zipmap(model)

    op_types = [n.op_type for n in stripped.graph.node]
    assert "ZipMap" not in op_types
    assert any(o.name == "probabilities" for o in stripped.graph.output)


def test_onnx_model_logger_prepend_feature_selector() -> None:
    """Проверка встраивания Gather(axis=1) для отбора подмножества фичей из 95 признаков."""
    import onnxruntime as ort
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(42)
    X_full = rng.normal(size=(20, 95)).astype(np.float32)
    idx = [10, 2, 45, 77]
    y = (X_full[:, 10] > 0).astype(int)

    clf = LogisticRegression().fit(X_full[:, idx], y)
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    raw_onnx = convert_sklearn(
        clf,
        initial_types=[("float_input", FloatTensorType([None, len(idx)]))],
        options={"zipmap": False},
        target_opset=14,
    )
    logger = OnnxModelLogger(n_features=95)
    cleaned = logger.strip_zipmap(raw_onnx)
    wrapped = logger.prepend_feature_selector(cleaned, feature_indices=idx, total_features=95)

    assert len(wrapped.graph.input) == 1
    assert wrapped.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 95
    assert wrapped.graph.node[0].op_type == "Gather"

    sess = ort.InferenceSession(wrapped.SerializeToString())
    outs = sess.run(None, {"float_input": X_full[:5]})
    assert np.allclose(outs[1], clf.predict_proba(X_full[:5, idx]), atol=1e-5)
