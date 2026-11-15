"""Сервис извлечения метрик, артефактов и строгой holdout-оценки `@champion` из MLflow."""

from typing import Any

import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import log_loss

from project.core.config import config
from project.core.logging import ProjectLogger
from project.core.utils import LogitTransformer
from project.features.store import FEATURE_STORE_STATE_FILENAME, FeatureStore
from project.infrastructure.clickhouse import FeatureDWH
from project.tracking.experiment_tracker.loggers.onnx_resolver import OnnxArtifactResolver

logger = ProjectLogger.get_logger(__name__)


class ChampionEvaluator:
    """Строгая оценка текущей продакшн-модели Champion из MLflow Registry без фолбэков."""

    @staticmethod
    def _extract_imp_list(features: Any) -> dict[str, float]:
        if not isinstance(features, list):
            return {}
        return {
            str(item["name"]): float(item["importance"])
            for item in features
            if isinstance(item, dict) and "name" in item and item.get("importance") is not None
        }

    @staticmethod
    def _get_champion_version(client: MlflowClient) -> Any | None:
        try:
            return client.get_model_version_by_alias("bot_detector", "champion")
        except MlflowException as exc:
            if "RESOURCE_DOES_NOT_EXIST" in str(getattr(exc, "error_code", "")) or "not found" in str(exc).lower():
                return None
            raise

    @classmethod
    def resolve_champion_info(cls, tracking_uri: str | None = None) -> dict[str, Any] | None:
        """Возвращает метаданные `@champion` из MLflow Registry или `None`, если чемпион не зарегистрирован."""
        uri = tracking_uri or config.mlflow_tracking_uri
        client = MlflowClient(tracking_uri=uri)
        mv = cls._get_champion_version(client)
        if mv is None:
            return None

        run_id = str(mv.run_id or "")
        if not run_id:
            raise RuntimeError(f"Champion bot_detector v{mv.version} has no associated MLflow run_id")
        run = client.get_run(run_id)
        val = run.data.metrics.get("val_logloss")
        if val is None:
            raise RuntimeError(f"Champion run {run_id} is missing required metric 'val_logloss'")
        return {"version": str(mv.version), "run_id": run_id, "loss": float(val)}

    @classmethod
    def fetch_champion_loss(cls, tracking_uri: str | None = None) -> float | None:
        """Получение валидационного LogLoss текущего `@champion` или `None` до первого промоушена."""
        info = cls.resolve_champion_info(tracking_uri=tracking_uri)
        return float(info["loss"]) if info is not None else None

    @classmethod
    def fetch_champion_importance(cls, tracking_uri: str | None = None) -> dict[str, float]:
        """Чтение карты важности признаков CatBoost строго из `feature_tree.json` текущего `@champion`."""
        uri = tracking_uri or config.mlflow_tracking_uri
        client = MlflowClient(tracking_uri=uri)
        mv = cls._get_champion_version(client)
        if mv is None:
            return {}

        run_id = str(mv.run_id or "")
        if not run_id:
            raise RuntimeError(f"Champion bot_detector v{mv.version} has no run_id")
        tree = OnnxArtifactResolver.download_json_artifact(
            run_id=run_id, artifact_path="feature_tree.json", tracking_uri=uri
        )
        base_models = tree.get("level1_base_models") or {}
        if not isinstance(base_models, dict) or "catboost_detector" not in base_models:
            raise RuntimeError(
                f"Champion run {run_id} feature_tree.json missing 'level1_base_models.catboost_detector'"
            )
        cb_node = base_models["catboost_detector"]
        imp = cls._extract_imp_list(cb_node.get("features") if isinstance(cb_node, dict) else None)
        if not imp:
            raise RuntimeError(f"Champion run {run_id} feature_tree.json has empty CatBoost features")
        return imp

    @staticmethod
    def save_predictions_to_dwh(
        *,
        dwh: FeatureDWH | None,
        run_id: str | None,
        run_name: str,
        dataset_split: str,
        dialog_ids: np.ndarray,
        participant_indices: np.ndarray,
        prob_bot: np.ndarray,
        prob_bot_calibrated: np.ndarray | None = None,
        is_bot: np.ndarray | None = None,
        fold_indices: np.ndarray | None = None,
    ) -> None:
        """Строгое сохранение OOF/Test предсказаний в ClickHouse `model_predictions`."""
        if dwh is None:
            return
        if not run_id or len(dialog_ids) == 0:
            raise ValueError(f"Cannot save empty predictions for {run_name} ({dataset_split})")
        dwh.save_predictions(
            run_id=str(run_id),
            run_name=run_name,
            dataset_split=dataset_split,
            dialog_ids=np.asarray(dialog_ids, dtype=str),
            participant_indices=np.asarray(participant_indices, dtype=np.int32),
            prob_bot=np.asarray(prob_bot, dtype=np.float64),
            prob_bot_calibrated=(
                np.asarray(prob_bot_calibrated, dtype=np.float64)
                if prob_bot_calibrated is not None
                else np.asarray(prob_bot, dtype=np.float64)
            ),
            is_bot=np.asarray(is_bot) if is_bot is not None else None,
            fold_indices=np.asarray(fold_indices, dtype=np.int32) if fold_indices is not None else None,
        )

    @classmethod
    def _build_champion_eval_matrix(
        cls,
        run_id: str,
        X_te: np.ndarray,
        dialogues_te: list[list[dict[str, Any]]],
        uri: str,
    ) -> np.ndarray:
        """Строит матрицу признаков строго в схеме `@champion` из `features/feature_store_state.json`."""
        state_dict = OnnxArtifactResolver.download_json_artifact(
            run_id=run_id,
            artifact_path=f"features/{FEATURE_STORE_STATE_FILENAME}",
            tracking_uri=uri,
        )
        active = state_dict.get("active_features")
        if not isinstance(active, list) or not active:
            raise RuntimeError(f"Champion run {run_id} feature_store_state.json has empty 'active_features'")

        if not dialogues_te or len(dialogues_te) * 2 != len(X_te):
            raise ValueError(
                f"Cannot evaluate champion run {run_id}: holdout dialogues ({len(dialogues_te)}) mismatch rows ({len(X_te)})"
            )
        champ_store = FeatureStore(auto_load_from_repo=False)
        champ_store.load_state_dict(state_dict)
        return champ_store.transform(dialogues_te).astype(np.float32)

    @classmethod
    def _infer_meta_onnx_probs(
        cls,
        run_id: str,
        X_eval: np.ndarray,
        uri: str,
    ) -> np.ndarray:
        import onnxruntime as ort

        probs: list[np.ndarray] = []
        for name in ("catboost_detector", "lightgbm_detector", "xgboost_detector"):
            onnx_path = OnnxArtifactResolver.resolve_model_onnx(model_name=name, run_id=run_id, tracking_uri=uri)
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            expected_dim = inp.shape[1] if len(inp.shape) > 1 else None
            if isinstance(expected_dim, int) and expected_dim != X_eval.shape[1]:
                raise RuntimeError(
                    f"Champion '{name}' expected feature_dim={expected_dim}, got X_eval.shape[1]={X_eval.shape[1]}"
                )
            probs.append(sess.run(None, {inp.name: X_eval})[1][:, 1])

        meta_path = OnnxArtifactResolver.resolve_model_onnx(model_name="meta_stacker", run_id=run_id, tracking_uri=uri)
        sess_meta = ort.InferenceSession(str(meta_path), providers=["CPUExecutionProvider"])
        meta_inp = sess_meta.get_inputs()[0]
        if len(meta_inp.shape) > 1 and isinstance(meta_inp.shape[1], int) and meta_inp.shape[1] != 3:
            raise RuntimeError(f"Champion 'meta_stacker' expected 3 logits, got {meta_inp.shape[1]}")
        meta_in = np.column_stack([LogitTransformer.transform(p) for p in probs]).astype(np.float32)
        return np.asarray(sess_meta.run(None, {meta_inp.name: meta_in})[1][:, 1], dtype=np.float64)

    @classmethod
    def evaluate_champion_on_holdout(
        cls,
        X_te: np.ndarray,
        y_te: np.ndarray,
        dialogues_te: list[list[dict[str, Any]]],
        tracking_uri: str | None = None,
    ) -> float | None:
        """Строгая holdout-оценка `@champion` (ONNX + `feature_store_state.json`), либо `None` если чемпиона ещё нет."""
        if len(y_te) < 4 or len(np.unique(y_te)) < 2:
            raise ValueError(
                f"Invalid holdout split for champion evaluation: n={len(y_te)}, classes={len(np.unique(y_te))}"
            )

        from project.models.postprocessors.pairwise import PairwisePostprocessor

        uri = tracking_uri or config.mlflow_tracking_uri
        info = cls.resolve_champion_info(tracking_uri=uri)
        if info is None:
            return None

        run_id = str(info["run_id"])
        X_eval = cls._build_champion_eval_matrix(run_id, X_te, dialogues_te, uri)
        p_meta = cls._infer_meta_onnx_probs(run_id, X_eval, uri)
        n_pairs = len(y_te) // 2
        if len(y_te) % 2 != 0:
            raise ValueError(f"Holdout size {len(y_te)} must be paired (even)")
        d_ids = np.repeat(np.arange(n_pairs), 2)
        p_idxs = np.arange(len(y_te)) % 2
        p_cal = PairwisePostprocessor().transform(p_meta, dialog_ids=d_ids, participant_indices=p_idxs)
        return float(log_loss(y_te, np.clip(p_cal, 1e-6, 1.0 - 1e-6)))


__all__ = ["ChampionEvaluator"]
