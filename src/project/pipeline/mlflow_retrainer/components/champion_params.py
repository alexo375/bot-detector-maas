import json
import tempfile
from pathlib import Path
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from project.core.logging import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

_MODEL_PARAM_SPECS: dict[str, dict[str, tuple[type, float | None, float | None, int | None]]] = {
    "catboost": {
        "learning_rate": (float, 0.015, 0.12, None),
        "depth": (int, 4, 7, None),
        "l2_leaf_reg": (float, 1.0, 10.0, None),
        "random_strength": (float, 1e-3, 10.0, None),
    },
    "lightgbm": {
        "learning_rate": (float, 0.015, 0.12, None),
        "num_leaves": (int, 15, 63, None),
        "max_depth": (int, 3, 8, None),
        "min_child_samples": (int, 10, 50, 5),
        "subsample": (float, 0.6, 1.0, None),
        "colsample_bytree": (float, 0.4, 1.0, None),
        "reg_alpha": (float, 1e-3, 5.0, None),
        "reg_lambda": (float, 1e-3, 5.0, None),
    },
    "xgboost": {
        "max_depth": (int, 3, 7, None),
        "learning_rate": (float, 0.015, 0.12, None),
        "subsample": (float, 0.6, 1.0, None),
        "colsample_bytree": (float, 0.4, 1.0, None),
        "reg_alpha": (float, 1e-3, 5.0, None),
        "reg_lambda": (float, 1e-3, 5.0, None),
    },
    "stacker": {
        "C": (float, 0.001, 10.0, None),
        "fit_intercept": (bool, None, None, None),
    },
}


class ChampionParamResolver:
    """Извлекает и валидирует гиперпараметры `@champion` из MLflow для warm-start Optuna."""

    @staticmethod
    def _coerce_bool(raw: Any) -> bool | None:
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in ("true", "1"):
            return True
        if text in ("false", "0"):
            return False
        return None

    @classmethod
    def _coerce_param(cls, raw: Any, spec: tuple[type, float | None, float | None, int | None]) -> Any | None:
        ptype, lo, hi, step = spec
        try:
            if ptype is bool:
                return cls._coerce_bool(raw)
            val = float(raw)
            if lo is not None:
                val = max(lo, val)
            if hi is not None:
                val = min(hi, val)
            if ptype is int:
                ival = round(val)
                if step and lo is not None:
                    lo_int = int(lo)
                    ival = lo_int + round((ival - lo_int) / step) * step
                    if hi is not None:
                        ival = min(int(hi), max(lo_int, ival))
                return int(ival)
            return float(val)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _extract_optuna_params(cls, raw_params: dict[str, Any], model_key: str) -> dict[str, Any]:
        specs = _MODEL_PARAM_SPECS.get(model_key, {})
        parsed: dict[str, Any] = {}
        for name, spec in specs.items():
            if name in raw_params and (val := cls._coerce_param(raw_params[name], spec)) is not None:
                parsed[name] = val
        return parsed

    @staticmethod
    def _extract_tree_level1_run_ids(client: MlflowClient, champ_run: Any) -> dict[str, str]:
        champ_run_id = str(getattr(champ_run.info, "run_id", "") or "")
        if not champ_run_id:
            return {}
        with tempfile.TemporaryDirectory(prefix="champ-tree-") as tmp_dir:
            art = Path(client.download_artifacts(champ_run_id, "feature_tree.json", dst_path=tmp_dir))
            if art.is_dir():
                art = art / "feature_tree.json"
            if not art.is_file():
                raise FileNotFoundError(f"Champion run {champ_run_id} missing feature_tree.json")
            tree = json.loads(art.read_text(encoding="utf-8"))

        out: dict[str, str] = {}
        base_models = tree.get("level1_base_models") or {}
        if isinstance(base_models, dict):
            for m_name, node in base_models.items():
                if isinstance(node, dict) and node.get("run_id"):
                    out[str(m_name)] = str(node["run_id"])
        return out

    @classmethod
    def _resolve_tagged_runs(cls, client: MlflowClient, champ_run: Any) -> dict[str, Any]:
        found: dict[str, Any] = {}
        tags = champ_run.data.tags or {}
        for key in ("catboost", "lightgbm", "xgboost"):
            rid = tags.get(f"level1_{key}_detector_run_id") or tags.get(f"level1_{key}_run_id")
            if rid:
                found[key] = client.get_run(str(rid))
        return found

    @classmethod
    def _resolve_level1_runs(cls, client: MlflowClient, champ_run: Any) -> dict[str, Any]:
        found = cls._resolve_tagged_runs(client, champ_run)
        if len(found) == 3:
            return found

        tree_runs = cls._extract_tree_level1_run_ids(client, champ_run)
        for key in ("catboost", "lightgbm", "xgboost"):
            if key in found:
                continue
            for cand_name, rid in tree_runs.items():
                if cand_name.lower().startswith(key):
                    found[key] = client.get_run(rid)
                    break

        if len(found) != 3:
            missing = sorted({"catboost", "lightgbm", "xgboost"} - set(found))
            raise RuntimeError(f"Champion run {champ_run.info.run_id} is missing Level-1 run references for: {missing}")
        return found

    @classmethod
    def fetch(cls, tracking_uri: str) -> dict[str, dict[str, Any]]:
        """Загружает гиперпараметры Level-1 и Meta-Stacker текущего `@champion` (без фолбэков)."""
        client = MlflowClient(tracking_uri=tracking_uri)
        try:
            mv = client.get_model_version_by_alias("bot_detector", "champion")
        except MlflowException as exc:
            if "RESOURCE_DOES_NOT_EXIST" in str(getattr(exc, "error_code", "")) or "not found" in str(exc).lower():
                return {}
            raise

        run_id = str(mv.run_id or "")
        if not run_id:
            raise RuntimeError(f"Champion bot_detector v{mv.version} has no run_id")
        champ_run = client.get_run(run_id)

        result: dict[str, dict[str, Any]] = {}
        stacker_params = cls._extract_optuna_params(dict(champ_run.data.params or {}), "stacker")
        if stacker_params:
            result["stacker"] = stacker_params

        for key, run in cls._resolve_level1_runs(client, champ_run).items():
            model_params = cls._extract_optuna_params(dict(run.data.params or {}), key)
            if model_params:
                result[key] = model_params

        if result:
            logger.info("Loaded champion Optuna warm-start params for: %s", ", ".join(sorted(result)))
        return result


__all__ = ["ChampionParamResolver"]
