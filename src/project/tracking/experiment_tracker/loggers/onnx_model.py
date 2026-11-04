import logging
import tempfile
from typing import Any

import catboost as cb
import lightgbm as lgb
import mlflow
import mlflow.onnx
import onnx
import onnxmltools
from onnx import TensorProto, helper
from onnxmltools.convert.common.data_types import FloatTensorType as LgbFloatTensorType
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SklFloatTensorType

logger = logging.getLogger(__name__)


class OnnxModelLogger:
    """Логер и конвертер моделей ансамбля в ONNX для сохранения в MLflow Model Registry."""

    def __init__(self, n_features: int | None = None) -> None:
        self.n_features = n_features

    @staticmethod
    def _ensure_unique_opset_imports(model: onnx.ModelProto, default_opset: int = 14) -> bool:
        """Удаляет дублирующиеся записи opset_import по domain и гарантирует наличие default domain."""
        by_domain: dict[str, int] = {}
        had_duplicates = False
        for op in model.opset_import:
            if op.domain in by_domain:
                had_duplicates = True
                by_domain[op.domain] = max(by_domain[op.domain], int(op.version))
            else:
                by_domain[op.domain] = int(op.version)

        missing_default = "" not in by_domain
        if missing_default:
            by_domain[""] = default_opset

        if had_duplicates or missing_default:
            del model.opset_import[:]
            for domain, version in by_domain.items():
                model.opset_import.append(helper.make_operatorsetid(domain, version))
            return True
        return False

    @staticmethod
    def _strip_zipmap_nodes(model: onnx.ModelProto) -> bool:
        modified = False
        for n in list(model.graph.node):
            if n.op_type != "ZipMap":
                continue
            prob_input, prob_output = n.input[0], n.output[0]
            model.graph.node.remove(n)
            for prev in model.graph.node:
                for i, out in enumerate(prev.output):
                    if out == prob_input:
                        prev.output[i] = prob_output
            for i, graph_out in enumerate(model.graph.output):
                if graph_out.name == prob_output:
                    new_out = helper.make_tensor_value_info(prob_output, TensorProto.FLOAT, [None, 2])
                    model.graph.output[i].CopyFrom(new_out)
            modified = True
        return modified

    @staticmethod
    def _reshape_label_outputs(model: onnx.ModelProto) -> bool:
        modified = False
        for label_name in ("label", "output_label"):
            has_label = any(o.name == label_name for o in model.graph.output)
            already_reshaped = any(n.op_type == "Reshape" and n.output[0] == label_name for n in model.graph.node)
            if not has_label or already_reshaped:
                continue
            for n in model.graph.node:
                for i, node_out in enumerate(n.output):
                    if node_out == label_name:
                        n.output[i] = f"{label_name}_raw"
            model.graph.initializer.append(helper.make_tensor(f"{label_name}_shape", TensorProto.INT64, [2], [-1, 1]))
            model.graph.node.append(
                helper.make_node("Reshape", inputs=[f"{label_name}_raw", f"{label_name}_shape"], outputs=[label_name])
            )
            for i, graph_out in enumerate(model.graph.output):
                if graph_out.name == label_name:
                    model.graph.output[i].CopyFrom(
                        helper.make_tensor_value_info(label_name, TensorProto.INT64, [None, 1])
                    )
            modified = True
        return modified

    def strip_zipmap(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Удаление узла ZipMap и стандартизация тензорного ввода-вывода графа ONNX."""
        modified = self._strip_zipmap_nodes(model)
        if self._reshape_label_outputs(model):
            modified = True
        if self._ensure_unique_opset_imports(model):
            modified = True
        if model.ir_version < 7:
            model.ir_version = 7
            modified = True
        if modified:
            onnx.checker.check_model(model)
        return model

    def prepend_feature_selector(
        self,
        model: onnx.ModelProto,
        feature_indices: list[int] | None,
        total_features: int | None = None,
    ) -> onnx.ModelProto:
        """Встраивает узел `Gather(axis=1)` для отбора нужных признаков из входного вектора."""
        if not feature_indices or not model.graph.input:
            return model

        total_feats = int(total_features or self.n_features or len(feature_indices))
        orig_input = model.graph.input[0]
        input_name = orig_input.name
        selected_name = f"{input_name}_selected"

        for node in model.graph.node:
            for idx, inp in enumerate(node.input):
                if inp == input_name:
                    node.input[idx] = selected_name

        indices_name = f"{input_name}_indices"
        indices_tensor = helper.make_tensor(
            indices_name,
            TensorProto.INT64,
            [len(feature_indices)],
            [int(i) for i in feature_indices],
        )
        model.graph.initializer.append(indices_tensor)

        gather_node = helper.make_node(
            "Gather",
            inputs=[input_name, indices_name],
            outputs=[selected_name],
            axis=1,
        )
        model.graph.node.insert(0, gather_node)

        new_input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, total_feats])
        model.graph.input[0].CopyFrom(new_input)

        self._ensure_unique_opset_imports(model)
        if model.ir_version < 7:
            model.ir_version = 7

        onnx.checker.check_model(model)
        return model

    def convert_catboost(
        self,
        model: cb.CatBoostClassifier,
        feature_indices: list[int] | None = None,
        total_features: int | None = None,
    ) -> onnx.ModelProto:
        """Конвертация обученного классификатора CatBoost в объект ONNX."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as tmp_file:
            model.save_model(tmp_file.name, format="onnx")
            onnx_model = onnx.load(tmp_file.name)
        onnx_model = self.strip_zipmap(onnx_model)
        return self.prepend_feature_selector(onnx_model, feature_indices, total_features)

    def convert_lightgbm(
        self,
        booster: lgb.Booster,
        n_features: int | None = None,
        feature_indices: list[int] | None = None,
        total_features: int | None = None,
    ) -> onnx.ModelProto:
        """Конвертация бустера LightGBM в объект ONNX."""
        n_feats = (
            n_features
            or (len(feature_indices) if feature_indices else None)
            or (booster.num_feature() if hasattr(booster, "num_feature") else None)
            or self.n_features
        )
        if n_feats is None:
            raise ValueError("Cannot determine n_features for LightGBM ONNX conversion")
        initial_types = [("float_input", LgbFloatTensorType([None, int(n_feats)]))]
        onnx_model = onnxmltools.convert_lightgbm(booster, initial_types=initial_types, target_opset=14)
        onnx_model = self.strip_zipmap(onnx_model)
        return self.prepend_feature_selector(onnx_model, feature_indices, total_features)

    def convert_xgboost(
        self,
        model: Any,
        n_features: int | None = None,
        feature_indices: list[int] | None = None,
        total_features: int | None = None,
    ) -> onnx.ModelProto:
        """Конвертация классификатора XGBoost в объект ONNX."""
        n_feats = (
            n_features
            or (len(feature_indices) if feature_indices else None)
            or getattr(model, "n_features_in_", None)
            or (
                len(model.get_booster().feature_names)
                if hasattr(model, "get_booster") and getattr(model.get_booster(), "feature_names", None)
                else None
            )
            or self.n_features
        )
        if n_feats is None:
            raise ValueError("Cannot determine n_features for XGBoost ONNX conversion")
        initial_types = [("float_input", LgbFloatTensorType([None, int(n_feats)]))]
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_types, target_opset=14)
        onnx_model = self.strip_zipmap(onnx_model)
        return self.prepend_feature_selector(onnx_model, feature_indices, total_features)

    def convert_meta_stacker(self, meta_clf: Any) -> onnx.ModelProto:
        """Конвертация мета-классификатора Stacker в объект ONNX."""
        if hasattr(meta_clf, "meta_model_") and meta_clf.meta_model_ is not None:
            meta_estimator = meta_clf.meta_model_
        elif hasattr(meta_clf, "meta_model") and meta_clf.meta_model is not None:
            meta_estimator = meta_clf.meta_model
        else:
            meta_estimator = meta_clf

        n_inputs = meta_estimator.coef_.shape[1] if hasattr(meta_estimator, "coef_") else 3
        initial_types = [("meta_input", SklFloatTensorType([None, n_inputs]))]
        onnx_model = convert_sklearn(
            meta_estimator, initial_types=initial_types, options={"zipmap": False}, target_opset=14
        )
        return self.strip_zipmap(onnx_model)

    def log_onnx_model(
        self,
        model: onnx.ModelProto,
        artifact_path: str,
        registered_model_name: str | None = None,
    ) -> Any:
        """Логирование графа ONNX напрямую в MLflow Model Registry."""
        return mlflow.onnx.log_model(
            model,
            name=artifact_path,
            registered_model_name=registered_model_name,
        )

    def log_ensemble_models(
        self,
        catboost_model: cb.CatBoostClassifier,
        lightgbm_booster: lgb.Booster,
        meta_stacker: Any,
        xgboost_model: Any | None = None,
        models_prefix: str = "",
        register_individual: bool = False,
        cb_feature_indices: list[int] | None = None,
        lgb_feature_indices: list[int] | None = None,
        xgb_feature_indices: list[int] | None = None,
        total_features: int | None = None,
    ) -> dict[str, Any]:
        """Конвертация моделей ансамбля в ONNX и логирование в MLflow."""
        cb_onnx = self.convert_catboost(
            catboost_model, feature_indices=cb_feature_indices, total_features=total_features
        )
        lgb_onnx = self.convert_lightgbm(
            lightgbm_booster, feature_indices=lgb_feature_indices, total_features=total_features
        )
        meta_onnx = self.convert_meta_stacker(meta_stacker)

        prefix = f"{models_prefix.rstrip('/')}/" if models_prefix else ""
        res = {
            "catboost": self.log_onnx_model(
                cb_onnx, f"{prefix}catboost_detector", "catboost_detector" if register_individual else None
            ),
            "lightgbm": self.log_onnx_model(
                lgb_onnx, f"{prefix}lightgbm_detector", "lightgbm_detector" if register_individual else None
            ),
            "meta_stacker": self.log_onnx_model(
                meta_onnx, f"{prefix}meta_stacker", "meta_stacker" if register_individual else None
            ),
        }

        if xgboost_model is not None:
            xgb_onnx = self.convert_xgboost(
                xgboost_model, feature_indices=xgb_feature_indices, total_features=total_features
            )
            res["xgboost"] = self.log_onnx_model(
                xgb_onnx, f"{prefix}xgboost_detector", "xgboost_detector" if register_individual else None
            )

        logger.info(
            "Successfully exported ONNX models to MLflow (prefix=%r, register_individual=%s).",
            models_prefix,
            register_individual,
        )
        return res
