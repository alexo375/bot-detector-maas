"""Утилиты построения полного иерархического дерева признаков (`feature_tree.json`) для мета-стекера."""

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.metrics import log_loss

from project.core.config import config
from project.core.logging import ProjectLogger
from project.models.classical.stacking import TwoLevelStackingClassifier

logger = ProjectLogger.get_logger(__name__)

_EXCLUDED_METRIC_PREFIXES = ("optuna_", "fold_", "fi_")
_EXCLUDED_METRIC_NAMES = {"run_duration_sec", "execution_time_seconds", "brier_score"}


class EnsembleTreeBuilder:
    """Строитель иерархического JSON-дерева ансамбля (Level-2 мета-признаки + Level-1 модели)."""

    @staticmethod
    def compute_meta_oof_metrics(
        X_tr_mat: np.ndarray,
        y_tr_arr: np.ndarray,
        final_feature_names: list[str],
        metrics: dict[str, float],
    ) -> dict[str, dict[str, float]]:
        """Рассчитывает OOF LogLoss для каждого входного мета-признака и итоговые метрики мета-модели."""
        oof_metrics: dict[str, dict[str, float]] = {}
        for idx, name in enumerate(final_feature_names):
            col = X_tr_mat[:, idx]
            probs = (
                np.clip(col, 1e-6, 1.0 - 1e-6)
                if np.all((col >= 0.0) & (col <= 1.0))
                else 1.0 / (1.0 + np.exp(-np.clip(col, -30.0, 30.0)))
            )
            oof_metrics[name] = {"logloss": float(log_loss(y_tr_arr, probs))}

        oof_metrics["meta"] = {
            "logloss": metrics["val_logloss"],
            "logloss_raw": metrics["val_logloss_raw"],
            "roc_auc": metrics["val_roc_auc"],
            "accuracy": metrics["val_accuracy"],
            "f1": metrics["val_f1"],
        }
        return oof_metrics

    @staticmethod
    def _match_stacker_weight(model_key: str, weights: dict[str, float]) -> float | None:
        short_key = model_key.removesuffix("_detector")
        for candidate in (f"logit_{model_key}", f"logit_{short_key}", model_key, short_key):
            if candidate in weights:
                return float(weights[candidate])
        for w_name, w_val in weights.items():
            if short_key and short_key in w_name:
                return float(w_val)
        return None

    @staticmethod
    def _load_level1_subtree(
        client: MlflowClient,
        model_key: str,
        run_id: str,
        weight_in_stacker: float | None,
    ) -> dict[str, Any]:
        l1_metrics: dict[str, float] = {}
        l1_params: dict[str, Any] = {}
        try:
            run = client.get_run(run_id)
            l1_metrics = {
                k: round(float(v), 5)
                for k, v in run.data.metrics.items()
                if k not in _EXCLUDED_METRIC_NAMES and not k.startswith(_EXCLUDED_METRIC_PREFIXES)
            }
            l1_params = dict(run.data.params)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not fetch Level-1 run metadata for %s (%s): %s", model_key, run_id, exc)

        l1_tree: dict[str, Any] = {}
        try:
            with tempfile.TemporaryDirectory(prefix="l1-tree-") as tmp_dir:
                downloaded = Path(client.download_artifacts(run_id, "feature_tree.json", dst_path=tmp_dir))
                if downloaded.is_dir():
                    downloaded = downloaded / "feature_tree.json"
                if downloaded.is_file():
                    loaded = json.loads(downloaded.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        l1_tree = loaded
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not download feature_tree.json for %s (%s): %s", model_key, run_id, exc)

        l1_features = list(l1_tree.get("features") or [])
        model_family = str(l1_tree.get("model_family") or l1_params.get("model_family") or model_key)
        n_features = int(l1_tree.get("n_features") or len(l1_features))

        node: dict[str, Any] = {
            "model_family": model_family,
            "run_id": run_id,
            "weight_in_stacker": weight_in_stacker,
            "importance_in_stacker": round(abs(weight_in_stacker), 5) if weight_in_stacker is not None else None,
            "n_features": n_features,
            "oof_metrics": l1_metrics,
            "hyperparameters": l1_params,
            "features": l1_features,
        }
        nested_children = l1_tree.get("level1_base_models")
        if isinstance(nested_children, dict) and nested_children:
            node["level1_base_models"] = nested_children
        return node

    @classmethod
    def build(
        cls,
        stacker: TwoLevelStackingClassifier,
        level1_runs: dict[str, str] | None = None,
        tracking_uri: str | None = None,
    ) -> dict[str, Any]:
        """Строит полное дерево ансамбля: Level-2 мета-признаки и развернутые модели Level-1."""
        tree = stacker.get_ensemble_tree()
        tree.pop("level1_runs", None)
        if not level1_runs:
            return tree

        resolved_uri = tracking_uri if isinstance(tracking_uri, str) and tracking_uri else config.mlflow_tracking_uri
        client = MlflowClient(tracking_uri=resolved_uri)
        weights: dict[str, float] = tree.get("weights", {})
        base_models = dict(tree.get("level1_base_models") or {})

        for model_key, raw_run_id in level1_runs.items():
            if not raw_run_id:
                continue
            weight = cls._match_stacker_weight(model_key, weights)
            base_models[model_key] = cls._load_level1_subtree(
                client=client,
                model_key=model_key,
                run_id=str(raw_run_id),
                weight_in_stacker=weight,
            )

        tree["level1_base_models"] = base_models
        return tree


__all__ = ["EnsembleTreeBuilder"]
