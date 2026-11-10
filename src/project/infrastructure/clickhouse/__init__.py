"""Модуль инфраструктуры ClickHouse DWH."""

from project.infrastructure.clickhouse.audit import ChatLogRecord, ClickHouseAuditWriter
from project.infrastructure.clickhouse.connection import ClickHouseConnectionFactory
from project.infrastructure.clickhouse.dwh import FeatureDWH
from project.infrastructure.clickhouse.migrator import ClickHouseMigrator
from project.infrastructure.clickhouse.repositories import (
    ClickHouseFeatureRepository,
    ClickHousePredictionRepository,
    ClickHouseRawRepository,
)

__all__ = [
    "ChatLogRecord",
    "ClickHouseAuditWriter",
    "ClickHouseConnectionFactory",
    "ClickHouseFeatureRepository",
    "ClickHouseMigrator",
    "ClickHousePredictionRepository",
    "ClickHouseRawRepository",
    "FeatureDWH",
]
