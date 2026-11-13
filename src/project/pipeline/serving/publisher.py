"""Сервис загрузки моделей из MLflow в репозиторий Triton."""

import json
from pathlib import Path

from mlflow.tracking import MlflowClient

from project.core.config import config
from project.core.logging import ProjectLogger
from project.features.store.store import FEATURE_STORE_STATE_FILENAME
from project.infrastructure.triton.version_manager import TritonModelVersionManager
from project.tracking.experiment_tracker.loggers.onnx_resolver import OnnxArtifactResolver

logger = ProjectLogger.get_logger(__name__)

_TABULAR_TRITON_INPUTS: dict[str, str] = {
    "catboost_detector": "features",
    "lightgbm_detector": "float_input",
    "xgboost_detector": "float_input",
    "bot_ensemble": "features",
}


class ModelPublisher:
    """Сервис загрузки моделей, состояния `FeatureStore` и обновления `dims` из MLflow в Triton."""

    def __init__(
        self,
        version_manager: TritonModelVersionManager | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        self.version_manager = version_manager or TritonModelVersionManager()
        self.tracking_uri = tracking_uri or config.mlflow_tracking_uri

    @property
    def triton_dir(self) -> Path:
        """Путь к корневому репозиторию Triton."""
        return self.version_manager.repository_path

    def get_existing_versions(self, model_name: str) -> list[int]:
        """Получить список существующих числовых версий модели."""
        return self.version_manager.get_versions(model_name)

    def _find_composite_run_id(self, client: MlflowClient) -> str:
        """Строгий поиск `run_id` зарегистрированной модели `bot_detector@champion`."""
        mv = client.get_model_version_by_alias("bot_detector", "champion")
        run_id = getattr(mv, "run_id", None) if mv else None
        if not run_id:
            raise RuntimeError("MLflow model bot_detector@champion has no associated run_id")
        logger.info("Found composite model bot_detector (@champion) with run_id=%s", run_id)
        return str(run_id)

    def _sync_feature_artifacts(self, champ_run_id: str) -> None:
        """Строгая синхронизация `feature_store_state.json` и Triton `dims` без фолбэков."""
        ensemble_versions = self.get_existing_versions("bot_ensemble")
        target_ver = max(ensemble_versions) if ensemble_versions else 1
        ver_dir = self.version_manager.prepare_version_dir("bot_ensemble", version=target_ver)

        state_payload = OnnxArtifactResolver.download_json_artifact(
            run_id=champ_run_id,
            artifact_path=f"features/{FEATURE_STORE_STATE_FILENAME}",
            tracking_uri=self.tracking_uri,
        )
        active = state_payload.get("active_features")
        if not isinstance(active, list) or not active:
            raise ValueError(
                f"Artifact features/{FEATURE_STORE_STATE_FILENAME} in run {champ_run_id} has empty 'active_features'"
            )

        target_state = ver_dir / FEATURE_STORE_STATE_FILENAME
        target_state.write_text(
            json.dumps(state_payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        logger.info("Synced %s from run %s -> %s", FEATURE_STORE_STATE_FILENAME, champ_run_id, target_state)
        self._update_triton_input_dims(len(active))

    def _update_triton_input_dims(self, feature_dim: int) -> None:
        for model_name, input_name in _TABULAR_TRITON_INPUTS.items():
            self.version_manager.update_model_input_dim(model_name, feature_dim=feature_dim, input_name=input_name)

    def publish_models(self) -> dict[str, int]:
        """Строгая синхронизация 4 ONNX-моделей и состояния `FeatureStore` из `@champion` в Triton."""
        models = ("catboost_detector", "lightgbm_detector", "xgboost_detector", "meta_stacker")
        published_versions: dict[str, int] = {}

        client = MlflowClient(tracking_uri=self.tracking_uri)
        champ_run_id = self._find_composite_run_id(client)

        for model_name in models:
            src_onnx = OnnxArtifactResolver.resolve_model_onnx(
                model_name=model_name,
                run_id=champ_run_id,
                tracking_uri=self.tracking_uri,
            )
            new_version = self.version_manager.publish_model(model_name, src_onnx)
            published_versions[model_name] = new_version
            logger.info("Published %s version %d from MLflow via version manager", model_name, new_version)

        self._sync_feature_artifacts(champ_run_id)
        logger.info("ModelPublisher completed Triton synchronization for %s: %s", self.triton_dir, published_versions)
        return published_versions
