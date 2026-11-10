from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from clickhouse_connect.driver.client import Client

from project.core.logging import ProjectLogger
from project.infrastructure.clickhouse.connection import ClickHouseConnectionFactory
from project.infrastructure.clickhouse.migrator import ClickHouseMigrator
from project.infrastructure.clickhouse.repositories import (
    ClickHouseFeatureRepository,
    ClickHousePredictionRepository,
    ClickHouseRawRepository,
)

logger = ProjectLogger.get_logger(__name__)


class FeatureDWH:
    """Клиент доступа к слоям данных в ClickHouse.

    Предоставляет лаконичный API для блокнотов и сервисов:
    - raw: сырые диалоги и сообщения
    - features: версионированные срезы признаков
    - predictions: OOF и тестовые предсказания для мета-стэкинга
    """

    def __init__(self, client: Client | None = None) -> None:
        self.client = client or ClickHouseConnectionFactory.create()
        self.raw = ClickHouseRawRepository(self.client)
        self.features = ClickHouseFeatureRepository(self.client)
        self.predictions = ClickHousePredictionRepository(self.client)

    def init_schema(self) -> list[str]:
        """Применяет миграции схемы DWH."""
        return ClickHouseMigrator.apply(self.client)

    def seed_raw_datasets(
        self,
        train_json_path: Path | str,
        test_json_path: Path | str,
        ytrain_csv_path: Path | str,
        ytest_csv_path: Path | str,
    ) -> dict[str, int]:
        """Загружает исходные JSON-диалоги и CSV-разметку в сырой слой ClickHouse (если еще не загружены)."""
        counts = {}
        if not self.raw.has_dialogues("train"):
            counts["train"] = self.raw.ingest_dialogues("train", train_json_path, ytrain_csv_path)
        else:
            counts["train"] = self.raw.count_dialogues("train")

        if not self.raw.has_dialogues("test"):
            counts["test"] = self.raw.ingest_dialogues("test", test_json_path, ytest_csv_path)
        else:
            counts["test"] = self.raw.count_dialogues("test")

        return counts

    def save_predictions(
        self,
        run_id: str,
        run_name: str,
        dataset_split: str,
        dialog_ids: np.ndarray,
        participant_indices: np.ndarray,
        prob_bot: np.ndarray,
        prob_bot_calibrated: np.ndarray | None = None,
        is_bot: np.ndarray | None = None,
        fold_indices: np.ndarray | None = None,
    ) -> int:
        """Сохранение предсказаний модели в ClickHouse DWH."""
        return self.predictions.save_predictions(
            run_id=run_id,
            run_name=run_name,
            dataset_split=dataset_split,
            dialog_ids=dialog_ids,
            participant_indices=participant_indices,
            prob_bot=prob_bot,
            prob_bot_calibrated=prob_bot_calibrated,
            is_bot=is_bot,
            fold_indices=fold_indices,
        )

    def load_meta_dataset(
        self,
        run_ids: list[str],
        dataset_split: str = "train_oof",
        use_calibrated: bool = True,
    ) -> pl.DataFrame:
        """Формирует готовый обучающий датасет для Level-2 Meta-Stacker через SQL JOIN."""
        return self.predictions.load_meta_dataset(
            run_ids=run_ids,
            dataset_split=dataset_split,
            use_calibrated=use_calibrated,
        )

    def status(self) -> dict[str, Any]:
        """Возвращает статус наполненности слоев DWH."""
        return {
            "raw_train_dialogues": self.raw.count_dialogues("train"),
            "raw_test_dialogues": self.raw.count_dialogues("test"),
            "feature_snapshots": len(self.features.list_snapshots()),
            "prediction_runs": len(self.predictions.list_prediction_runs()),
        }
