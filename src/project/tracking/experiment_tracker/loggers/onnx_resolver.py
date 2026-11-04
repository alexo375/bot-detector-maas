import json
from pathlib import Path
from typing import Any

import mlflow.artifacts
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from project.core.config import config
from project.core.logging import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class OnnxArtifactResolver:
    """Единый строгий резолвер ONNX-артефактов и состояния FeatureStore в MLflow 3 без фолбэков."""

    @staticmethod
    def _resolve_logged_model_uris(run: Any, client: MlflowClient, model_name: str) -> list[str]:
        uris: list[str] = []
        outputs = getattr(run, "outputs", None)
        for model_output in getattr(outputs, "model_outputs", []) or []:
            model_id = getattr(model_output, "model_id", None)
            if not model_id:
                continue
            logged_model = client.get_logged_model(str(model_id))
            if getattr(logged_model, "name", None) == model_name:
                uris.append(f"models:/{model_id}")
                art_loc = getattr(logged_model, "artifact_location", None)
                if art_loc:
                    uris.append(str(art_loc))
        return uris

    @staticmethod
    def _collect_run_logged_names(run: Any, client: MlflowClient) -> set[str]:
        names: set[str] = set()
        outputs = getattr(run, "outputs", None)
        for model_output in getattr(outputs, "model_outputs", []) or []:
            model_id = getattr(model_output, "model_id", None)
            if not model_id:
                continue
            logged_model = client.get_logged_model(str(model_id))
            if getattr(logged_model, "name", None):
                names.add(str(logged_model.name))
        return names

    @classmethod
    def has_required_models(
        cls,
        run_id: str,
        required_names: tuple[str, ...] | list[str],
        tracking_uri: str | None = None,
    ) -> bool:
        """Проверяет наличие требуемых ONNX-моделей в `LoggedModels` или артефактах рана."""
        uri = tracking_uri or config.mlflow_tracking_uri
        try:
            client = MlflowClient(tracking_uri=uri)
            run = client.get_run(run_id)
            logged_names = cls._collect_run_logged_names(run, client)
            if all(name in logged_names for name in required_names):
                return True

            top_paths = {a.path for a in client.list_artifacts(run_id)}
            return all(name in top_paths for name in required_names)
        except MlflowException as exc:
            logger.debug("MLflow lookup error checking ONNX models for %s: %s", run_id, exc)
            return False

    @classmethod
    def resolve_model_onnx(
        cls,
        model_name: str,
        run_id: str,
        tracking_uri: str | None = None,
    ) -> Path:
        """Строгий поиск и загрузка `model.onnx` из указанного рана MLflow (без фолбэков)."""
        if not run_id:
            raise ValueError(f"run_id is required to resolve ONNX model '{model_name}'")

        uri = tracking_uri or config.mlflow_tracking_uri
        client = MlflowClient(tracking_uri=uri)
        run = client.get_run(run_id)
        candidate_uris = cls._resolve_logged_model_uris(run, client, model_name)
        candidate_uris.append(f"runs:/{run_id}/{model_name}")

        last_err: Exception | None = None
        for cand_uri in candidate_uris:
            try:
                local_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=cand_uri, tracking_uri=uri))
                cand = local_dir / "model.onnx" if local_dir.is_dir() else local_dir
                if cand.is_file():
                    return cand
            except (MlflowException, OSError) as dl_err:
                last_err = dl_err

        raise FileNotFoundError(
            f"ONNX model '{model_name}' not found in MLflow run {run_id} (URIs tried: {candidate_uris})"
        ) from last_err

    @staticmethod
    def download_json_artifact(
        run_id: str,
        artifact_path: str,
        tracking_uri: str | None = None,
    ) -> dict[str, Any]:
        """Строгая загрузка обязательного JSON-артефакта из указанного рана MLflow."""
        if not run_id:
            raise ValueError(f"run_id is required to download artifact '{artifact_path}'")
        uri = tracking_uri or config.mlflow_tracking_uri
        local_file = Path(
            mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{artifact_path}", tracking_uri=uri)
        )
        if not local_file.is_file():
            raise FileNotFoundError(f"JSON artifact '{artifact_path}' missing in MLflow run {run_id}")
        payload = json.loads(local_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"JSON artifact '{artifact_path}' in run {run_id} must be an object")
        return payload


__all__ = ["OnnxArtifactResolver"]
