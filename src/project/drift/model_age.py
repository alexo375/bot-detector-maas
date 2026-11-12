"""Аудит возраста активной модели Champion и извлечение метаданных из MLflow."""

import datetime as dt
import json
from typing import Any

import redis
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from project.core.config import config
from project.core.logging import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

_EFFECTIVE_SINCE_KEY_SUFFIX = "champion_effective_since_ts"


class ChampionMetadataReader:
    """Читает метаданные, параметры и возраст активного `@champion` из MLflow и Redis (строго без фолбэков)."""

    @staticmethod
    def _read_effective_since_override(model_name: str, champion_version: str) -> float | None:
        key = f"{model_name}:{_EFFECTIVE_SINCE_KEY_SUFFIX}"
        r_conn = redis.Redis.from_url(config.redis.url, socket_connect_timeout=0.5, socket_timeout=0.5)
        try:
            raw = r_conn.get(key)
        finally:
            r_conn.close()
        if not raw:
            return None
        payload = json.loads(raw)
        if isinstance(payload, dict) and str(payload.get("version")) == champion_version:
            ts_val = payload.get("ts")
            return float(ts_val) if ts_val is not None else None
        return None

    @classmethod
    def read(cls, model_name: str = "bot_detector") -> dict[str, Any]:
        """Получение возраста (в днях), версии и параметров `@champion` из MLflow."""
        now_ts = dt.datetime.now(dt.UTC).timestamp()
        client = MlflowClient(tracking_uri=config.mlflow_tracking_uri)
        try:
            champion_ver = client.get_model_version_by_alias(model_name, "champion")
        except MlflowException as exc:
            if "RESOURCE_DOES_NOT_EXIST" in str(getattr(exc, "error_code", "")) or "not found" in str(exc).lower():
                return {"age_days": 0.0, "version": "none", "best_params": {}, "run_id": "", "created_at_ts": None}
            raise

        creation_sec = float(champion_ver.creation_timestamp) / 1000.0
        override_ts = cls._read_effective_since_override(model_name, str(champion_ver.version))
        effective_sec = max(creation_sec, override_ts) if override_ts is not None else creation_sec
        age_days = max(0.0, (now_ts - effective_sec) / 86400.0)

        best_params: dict[str, Any] = {}
        if champion_ver.run_id:
            run = client.get_run(champion_ver.run_id)
            best_params = dict(run.data.params)

        return {
            "age_days": round(age_days, 2),
            "version": str(champion_ver.version),
            "best_params": best_params,
            "run_id": champion_ver.run_id,
            "created_at_ts": effective_sec,
        }


__all__ = ["ChampionMetadataReader"]
