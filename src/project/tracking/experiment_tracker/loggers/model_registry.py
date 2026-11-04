from typing import Any

import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.onnx
import mlflow.pytorch
import mlflow.sklearn
import mlflow.transformers
import mlflow.xgboost

from project.core.config import config
from project.core.logging import ProjectLogger
from project.tracking.experiment_tracker.schemas.metric_mode import MetricOptimizationMode

logger = ProjectLogger.get_logger(__name__)


class ModelRegistryService:
    """Сервис регистрации моделей и управления алиасами `champion` / `challenger` в MLflow."""

    @staticmethod
    def _log_by_framework(model: Any, artifact_path: str) -> Any | None:
        logged: Any | None = None
        if hasattr(model, "graph") or "ModelProto" in str(type(model)):
            logged = mlflow.onnx.log_model(model, name=artifact_path)
        elif hasattr(model, "save_model") and "CatBoost" in str(type(model)):
            logged = mlflow.catboost.log_model(model, name=artifact_path)
        elif hasattr(model, "booster_") or "LGBM" in str(type(model)):
            logged = mlflow.lightgbm.log_model(model, name=artifact_path)
        elif hasattr(model, "get_booster") or "XGB" in str(type(model)):
            logged = mlflow.xgboost.log_model(model, name=artifact_path)
        elif hasattr(model, "save_pretrained") or (
            isinstance(model, dict) and "model" in model and hasattr(model["model"], "save_pretrained")
        ):
            logged = mlflow.transformers.log_model(model, name=artifact_path, task="text-classification")
        elif hasattr(model, "predict_proba"):
            logged = mlflow.sklearn.log_model(
                model,
                name=artifact_path,
                serialization_format="skops",
                skops_trusted_types=[
                    "project.models.classical.stacking.TwoLevelStackingClassifier",
                    "project.models.classical.TwoLevelStackingClassifier",
                ],
            )
        elif hasattr(model, "forward") or hasattr(model, "state_dict"):
            logged = mlflow.pytorch.log_model(model, name=artifact_path)
        return logged

    @staticmethod
    def _resolve_stage_tag(run_name: str) -> str:
        if "hmm" in run_name:
            return "champion_stack"
        if "meta" in run_name or "stack" in run_name:
            return "meta_stack"
        if "hybrid" in run_name or "rubert" in run_name:
            return "hybrid"
        if "pruned" in run_name or "top" in run_name:
            return "top_pruned"
        return "baseline"

    @classmethod
    def log_model(
        cls,
        model: Any,
        artifact_path: str = "model",
        register_as: str | None = "bot_detector",
        active_run: Any | None = None,
        tracking_uri: str | None = None,
    ) -> Any | None:
        """Логирование экземпляра модели в MLflow Model Registry (без преждевременного промоушена)."""
        model_info = cls._log_by_framework(model, artifact_path)
        if not (register_as and active_run):
            return None

        model_uri = getattr(model_info, "model_uri", None) or f"runs:/{active_run.info.run_id}/{artifact_path}"
        mv = mlflow.register_model(model_uri=model_uri, name=register_as)
        try:
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            run_data = client.get_run(active_run.info.run_id).data
            family = run_data.params.get("model_family") or run_data.tags.get("model_family") or "unknown"
            run_name = run_data.tags.get("mlflow.runName", "")
            client.set_model_version_tag(register_as, mv.version, "model_family", family)
            client.set_model_version_tag(register_as, mv.version, "stage", cls._resolve_stage_tag(run_name))
        except Exception as tag_err:  # noqa: BLE001  # Resilient fallback
            logger.debug("Failed to set model registry tags: %s", tag_err)
        return mv

    @staticmethod
    def _is_metric_better(candidate: float, baseline: float, mode: MetricOptimizationMode) -> bool:
        return candidate > baseline if mode == MetricOptimizationMode.MAX else candidate < baseline

    @staticmethod
    def _fetch_run_metric(client: Any, run_id: str | None, metric_name: str) -> float | None:
        if not run_id:
            return None
        try:
            val = client.get_run(str(run_id)).data.metrics.get(metric_name)
            return float(val) if val is not None else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to fetch run metric '%s' for %s: %s", metric_name, run_id, exc)
            return None

    @staticmethod
    def _safe_alias_version(client: Any, register_as: str, alias: str) -> Any | None:
        try:
            return client.get_model_version_by_alias(register_as, alias)
        except Exception:  # noqa: BLE001
            return None

    @classmethod
    def evaluate_promotion(
        cls,
        mv: Any,
        register_as: str,
        metric_for_promotion: str = "val_logloss",
        mode: MetricOptimizationMode = MetricOptimizationMode.MIN,
        active_run: Any | None = None,
        tracking_uri: str | None = None,
        logged_metrics: dict[str, float] | None = None,
    ) -> None:
        """Сравнивает метрику зарегистрированной модели с Champion и обновляет алиасы champion/challenger."""
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri or config.mlflow_tracking_uri)
        current_score = (logged_metrics or {}).get(metric_for_promotion)
        if current_score is None and active_run:
            current_score = cls._fetch_run_metric(client, active_run.info.run_id, metric_for_promotion)

        if current_score is None:
            logger.warning(
                "Промоушн модели %s (v%s) пропущен: метрика %s отсутствует.",
                register_as,
                mv.version,
                metric_for_promotion,
            )
            return

        current_champ_mv = cls._safe_alias_version(client, register_as, "champion")
        if current_champ_mv is None:
            client.set_registered_model_alias(register_as, "champion", mv.version)
            return
        if str(current_champ_mv.version) == str(mv.version):
            return

        champ_score = cls._fetch_run_metric(client, getattr(current_champ_mv, "run_id", None), metric_for_promotion)
        if champ_score is None or cls._is_metric_better(float(current_score), float(champ_score), mode):
            client.set_registered_model_alias(register_as, "challenger", current_champ_mv.version)
            client.set_registered_model_alias(register_as, "champion", mv.version)
            return

        current_chall_mv = cls._safe_alias_version(client, register_as, "challenger")
        if current_chall_mv is None:
            client.set_registered_model_alias(register_as, "challenger", mv.version)
            return
        if str(current_chall_mv.version) == str(mv.version):
            return

        chall_score = cls._fetch_run_metric(client, getattr(current_chall_mv, "run_id", None), metric_for_promotion)
        if chall_score is None or cls._is_metric_better(float(current_score), float(chall_score), mode):
            client.set_registered_model_alias(register_as, "challenger", mv.version)

    @classmethod
    def register_promoted_version(
        cls,
        *,
        model: Any,
        active_run: Any,
        register_as: str = "bot_detector",
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Any:
        """Регистрирует модель из уже валидированного активного рана и переводит её в `@champion`."""
        model_info = cls._log_by_framework(model, "model")
        run_id = str(active_run.info.run_id)
        model_uri = getattr(model_info, "model_uri", None) or f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=register_as)
        new_ver = str(mv.version)

        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri or config.mlflow_tracking_uri)
        prev_champ = cls._safe_alias_version(client, register_as, "champion")
        if prev_champ is not None and str(prev_champ.version) != new_ver:
            client.set_registered_model_alias(register_as, "challenger", str(prev_champ.version))
            client.set_model_version_tag(register_as, new_ver, "replaced_champion_version", str(prev_champ.version))

        client.set_registered_model_alias(register_as, "champion", new_ver)
        for tk, tv in (tags or {}).items():
            client.set_model_version_tag(register_as, new_ver, str(tk), str(tv))
        logger.info("Promoted %s v%s (run_id=%s) to @champion", register_as, new_ver, run_id)
        return mv


__all__ = ["ModelRegistryService"]
