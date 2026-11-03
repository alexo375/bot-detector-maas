"""Централизованная типизированная конфигурация проекта."""

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from project.core.paths import paths

load_dotenv(paths.env_file, override=False)


class RedisSettings(BaseSettings):
    """Настройки подключения и TTL для Redis."""

    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")

    url: str = Field(default="")
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    cache_default_ttl_sec: int = Field(default=86400, ge=0)
    task_default_ttl_sec: int = Field(default=3600, ge=0)

    @model_validator(mode="after")
    def _resolve_url(self) -> "RedisSettings":
        if not self.url:
            self.url = f"redis://{self.host}:{self.port}/0"
        return self


class RabbitMQSettings(BaseSettings):
    """Настройки подключения к RabbitMQ брокеру сообщений."""

    model_config = SettingsConfigDict(env_prefix="RABBITMQ_", extra="ignore")

    url: str = Field(default="amqp://guest:guest@localhost:5672/")
    prefetch_count: int = Field(default=5, ge=1, le=1000)
    queue_audit_events: str = Field(default="dialog_events")
    queue_inference_tasks: str = Field(default="dialogue_inference_tasks")


class TritonSettings(BaseSettings):
    """Настройки подключения к Triton Inference Server."""

    model_config = SettingsConfigDict(env_prefix="TRITON_", extra="ignore")

    url: str = Field(default="http://localhost:8000")
    grpc_url: str = Field(default="localhost:8001")
    protocol: str = Field(default="http")
    model_repository: str = Field(default=str(paths.triton_models))
    max_batch_size: int = Field(default=64, ge=1)


class ClickHouseSettings(BaseSettings):
    """Настройки подключения к ClickHouse DWH."""

    model_config = SettingsConfigDict(env_prefix="CLICKHOUSE_", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=8123, ge=1, le=65535)
    database: str = Field(default="bot_detection")
    user: str = Field(default="default")
    password: str = Field(default="")
    batch_size: int = Field(default=50, ge=1)
    flush_interval_sec: float = Field(default=2.0, ge=0.1)
    enable_audit: bool = Field(default=True)


class ScheduleSettings(BaseSettings):
    """Настройки расписаний регулярных пайплайнов Prefect."""

    model_config = SettingsConfigDict(env_prefix="SCHEDULE_", extra="ignore")

    drift_monitoring_cron: str = Field(default="*/2 * * * *")
    drift_window_days: int = Field(default=7, ge=1)
    drift_threshold: float = Field(default=0.20, gt=0.0, le=1.0)
    min_retrain_samples: int = Field(default=100, ge=10)
    max_model_staleness_days: int = Field(default=30, ge=1)
    auto_trigger_retrain: bool = Field(default=False)
    retrain_cooldown_sec: int = Field(default=600, ge=0)


class RetrainSettings(BaseSettings):
    """Параметры обучения и подбора гиперпараметров по эталону ноутбуков."""

    model_config = SettingsConfigDict(env_prefix="RETRAIN_", extra="ignore")

    sample_size: int = Field(default=3032, ge=20)
    n_trials: int = Field(default=300, ge=1)
    patience: int = Field(default=50, ge=1)
    n_splits: int = Field(default=5, ge=2)
    n_estimators: int = Field(default=750, ge=10)
    early_stopping_rounds: int = Field(default=40, ge=5)
    corr_threshold: float = Field(default=0.95, gt=0.0, le=1.0)


@dataclass
class AppConfig:
    """Глобальная конфигурация инфраструктурных параметров сервисов."""

    mlflow_tracking_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    ntfy_url: str = field(default_factory=lambda: os.getenv("NTFY_URL", "https://ntfy.sh/bot-detector-alerts"))

    redis: RedisSettings = field(default_factory=RedisSettings)
    rabbitmq: RabbitMQSettings = field(default_factory=RabbitMQSettings)
    triton: TritonSettings = field(default_factory=TritonSettings)
    clickhouse: ClickHouseSettings = field(default_factory=ClickHouseSettings)
    schedule: ScheduleSettings = field(default_factory=ScheduleSettings)
    retrain: RetrainSettings = field(default_factory=RetrainSettings)

    def to_dict(self) -> dict[str, Any]:
        """Возвращает словарное представление конфигурации."""
        return {
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
            "redis_url": self.redis.url,
            "rabbitmq_url": self.rabbitmq.url,
            "triton_url": self.triton.url,
            "clickhouse_host": self.clickhouse.host,
            "clickhouse_port": self.clickhouse.port,
        }


config = AppConfig()

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__all__ = [
    "AppConfig",
    "ClickHouseSettings",
    "RabbitMQSettings",
    "RedisSettings",
    "RetrainSettings",
    "ScheduleSettings",
    "TritonSettings",
    "config",
]
