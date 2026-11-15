"""Сервис инспекции и валидации артефактов версий моделей в MLflow."""

import datetime as dt
import tempfile
from pathlib import Path
from typing import Any

import mlflow.artifacts
import onnx
from mlflow.tracking import MlflowClient

from project.core.config import config
from project.core.logging import ProjectLogger
from project.pipeline.model_rollback.schemas import ModelVersionCandidate
from project.tracking.experiment_tracker.loggers.onnx_resolver import OnnxArtifactResolver

logger = ProjectLogger.get_logger(__name__)

REQUIRED_ONNX_MODELS = (
    "catboost_detector",
    "lightgbm_detector",
    "xgboost_detector",
    "meta_stacker",
)


class ModelVersionInspector:
    """Инспектор версий `bot_detector` в MLflow 3 (`LoggedModels` и legacy-артефакты)."""

    @staticmethod
    def safe_alias_version(client: MlflowClient, model_name: str, alias: str) -> str | None:
        """Безопасное получение номера версии по алиасу MLflow."""
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            return str(mv.version) if mv is not None else None
        except Exception:  # noqa: BLE001
            return None

    @classmethod
    def run_has_required_onnx(cls, client: MlflowClient, run_id: str) -> bool:
        """Проверяет наличие всех 4 ONNX-моделей ансамбля в MLflow 3 `LoggedModels` или артефактах."""
        if not run_id:
            return False
        uri = getattr(client, "tracking_uri", None) or config.mlflow_tracking_uri
        return OnnxArtifactResolver.has_required_models(run_id, REQUIRED_ONNX_MODELS, tracking_uri=uri)

    @classmethod
    def _download_meta_onnx(cls, client: MlflowClient, run_id: str, tmp_dir: str) -> Path | None:
        uri = getattr(client, "tracking_uri", None) or config.mlflow_tracking_uri
        try:
            run = client.get_run(run_id)
            for cand_uri in OnnxArtifactResolver._resolve_logged_model_uris(run, client, "meta_stacker"):
                try:
                    local_dir = Path(
                        mlflow.artifacts.download_artifacts(artifact_uri=cand_uri, tracking_uri=uri, dst_path=tmp_dir)
                    )
                    cand = local_dir / "model.onnx" if local_dir.is_dir() else local_dir
                    if cand.is_file():
                        return cand
                except Exception as uri_err:  # noqa: BLE001
                    logger.debug("LoggedModel URI %s skipped for run %s: %s", cand_uri, run_id, uri_err)
        except Exception as run_err:  # noqa: BLE001
            logger.debug("LoggedModel lookup failed for run %s: %s", run_id, run_err)

        for rel in ("meta_stacker/model.onnx", "models/meta_stacker/model.onnx"):
            try:
                downloaded = Path(client.download_artifacts(run_id, rel, dst_path=tmp_dir))
                cand = downloaded / "model.onnx" if downloaded.is_dir() else downloaded
                if cand.is_file():
                    return cand
            except Exception as dl_err:  # noqa: BLE001
                logger.debug("Artifact path %s skipped for run %s: %s", rel, run_id, dl_err)
        return None

    @classmethod
    def validate_target_run_onnx(
        cls,
        client: MlflowClient,
        run_id: str,
        expected_meta_dim: int = 3,
    ) -> None:
        """Проверяет совместимость ONNX графа `meta_stacker` с текущей конфигурацией Triton."""
        if not cls.run_has_required_onnx(client, run_id):
            raise ValueError(f"MLflow run {run_id!r} не содержит полный набор ONNX-моделей {REQUIRED_ONNX_MODELS}.")

        with tempfile.TemporaryDirectory(prefix="rollback-val-") as tmp_dir:
            meta_onnx_path = cls._download_meta_onnx(client, run_id, tmp_dir)
            if meta_onnx_path is None:
                raise ValueError(f"В MLflow run {run_id!r} не найден файл ONNX для meta_stacker.")

            model_proto = onnx.load(str(meta_onnx_path))
            dims = model_proto.graph.input[0].type.tensor_type.shape.dim
            dim_val = int(dims[1].dim_value) if len(dims) > 1 and dims[1].dim_value else None
            if dim_val is not None and dim_val != expected_meta_dim:
                raise ValueError(
                    f"Несовместимая размерность meta_stacker ({dim_val} вместо {expected_meta_dim}) в run {run_id!r}."
                )

    @staticmethod
    def _build_candidate(
        client: MlflowClient,
        mv: Any,
        alias_map: dict[str, list[str]],
    ) -> ModelVersionCandidate:
        ver_str = str(mv.version)
        run_id = str(mv.run_id or "")
        val_loss: float | None = None
        try:
            metrics = client.get_run(run_id).data.metrics
            raw_loss = metrics.get("val_logloss")
            if raw_loss is not None:
                val_loss = float(raw_loss)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Metrics lookup skipped for v%s: %s", ver_str, exc)

        created_ts = float(mv.creation_timestamp or 0) / 1000.0
        dt_str = (
            dt.datetime.fromtimestamp(created_ts, tz=dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
            if created_ts > 0
            else "unknown-date"
        )
        aliases = sorted(set(alias_map.get(ver_str, []) + [f"@{a}" for a in (getattr(mv, "aliases", None) or [])]))
        loss_part = f"loss={val_loss:.4f}" if val_loss is not None else "loss=n/a"
        alias_part = f" | {', '.join(aliases)}" if aliases else ""
        return ModelVersionCandidate(
            version=ver_str,
            run_id=run_id,
            created_ts=created_ts,
            val_logloss=val_loss,
            aliases=aliases,
            label=f"v{ver_str} | {dt_str} | {loss_part}{alias_part}",
        )

    @classmethod
    def list_candidates(
        cls,
        model_name: str = "bot_detector",
        tracking_uri: str | None = None,
    ) -> list[ModelVersionCandidate]:
        """Возвращает отсортированный по убыванию список совместимых версий модели из MLflow."""
        client = MlflowClient(tracking_uri=tracking_uri or config.mlflow_tracking_uri)
        versions = list(client.search_model_versions(f"name='{model_name}'"))
        versions.sort(key=lambda mv: int(mv.version) if str(mv.version).isdigit() else -1, reverse=True)

        alias_map: dict[str, list[str]] = {}
        for alias in ("champion", "challenger", "rolled_back"):
            ver = cls.safe_alias_version(client, model_name, alias)
            if ver is not None:
                alias_map.setdefault(ver, []).append(f"@{alias}")

        return [
            cls._build_candidate(client, mv, alias_map)
            for mv in versions
            if cls.run_has_required_onnx(client, str(mv.run_id or ""))
        ]


__all__ = ["REQUIRED_ONNX_MODELS", "ModelVersionInspector"]
