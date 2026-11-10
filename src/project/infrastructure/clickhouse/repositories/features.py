from typing import Any

import numpy as np
import polars as pl
from clickhouse_connect.driver.client import Client

from project.core.logging import ProjectLogger
from project.infrastructure.clickhouse.connection import ClickHouseConnectionFactory

logger = ProjectLogger.get_logger(__name__)


class ClickHouseFeatureRepository:
    """Репозиторий витрин признаков (Gold Layer / Feature Store)."""

    def __init__(self, client: Client | None = None) -> None:
        self.client = client or ClickHouseConnectionFactory.create()

    def _get_or_create_snapshot_id(
        self,
        snapshot_tag: str,
        feature_names: list[str],
        description: str = "",
    ) -> int:
        """Находит существующий snapshot_id по тегу или регистрирует новый инкрементный снапшот."""
        res = self.client.query(
            "SELECT snapshot_id FROM feature_snapshots WHERE snapshot_tag = {tag:String} LIMIT 1",
            parameters={"tag": snapshot_tag},
        ).result_rows

        if res:
            return int(res[0][0])

        max_res = self.client.query("SELECT max(snapshot_id) FROM feature_snapshots").result_rows
        max_id = max_res[0][0] if max_res and max_res[0][0] is not None else 0
        new_id = int(max_id) + 1

        self.client.insert(
            "feature_snapshots",
            [
                [
                    new_id,
                    snapshot_tag,
                    len(feature_names),
                    feature_names,
                    description,
                ]
            ],
            column_names=[
                "snapshot_id",
                "snapshot_tag",
                "features_count",
                "feature_names",
                "description",
            ],
        )
        logger.info(
            "Зарегистрирован новый срез признаков: ID=%d, Tag='%s' (%d фичей)", new_id, snapshot_tag, len(feature_names)
        )
        return new_id

    def has_snapshot(self, snapshot_tag: str, dataset_type: str = "train") -> bool:
        """Проверяет наличие записей для указанного тега снапшота и типа датасета."""
        cnt = self.client.command(
            "SELECT count() FROM dialogue_features FINAL WHERE snapshot_tag = {tag:String} AND dataset_type = {dtype:String}",
            parameters={"tag": snapshot_tag, "dtype": dataset_type},
        )
        return int(str(cnt)) > 0

    def save_snapshot(
        self,
        snapshot_tag: str,
        features_df: pl.DataFrame,
        dataset_type: str = "train",
        description: str = "",
        batch_size: int = 5000,
    ) -> int:
        """Сохранение набора признаков в таблицу dialogue_features.

        Args:
            snapshot_tag: Уникальный текстовый идентификатор версии (например, 'stage_20_candidates').
            features_df: Polars DataFrame, содержащий 'dialog_id', 'participant_index' и колонки фичей.
            dataset_type: 'train' или 'test'.
            description: Описание версии фичей.
            batch_size: Размер батча вставки.

        Returns:
            snapshot_id: Числовой идентификатор снапшота/партиции.

        """
        required_cols = {"dialog_id", "participant_index"}
        if not required_cols.issubset(set(features_df.columns)):
            raise ValueError(f"DataFrame должен содержать колонки {required_cols}")

        feature_cols = [
            c for c in features_df.columns if c not in ("dialog_id", "participant_index", "is_bot", "ID", "id")
        ]
        if not feature_cols:
            raise ValueError("В DataFrame отсутствуют колонки признаков")

        snapshot_id = self._get_or_create_snapshot_id(
            snapshot_tag=snapshot_tag,
            feature_names=feature_cols,
            description=description,
        )

        d_ids = [str(d) for d in features_df["dialog_id"].to_list()]
        p_idxs = features_df["participant_index"].to_numpy().astype(np.uint8)

        # Извлекаем значения признаков в словарь для ClickHouse Map(String, Float32)
        feats_matrix = features_df.select(feature_cols).to_numpy().astype(np.float32)

        rows_to_insert: list[list[Any]] = []
        for i in range(len(features_df)):
            feats_map = {col_name: float(feats_matrix[i, j]) for j, col_name in enumerate(feature_cols)}
            rows_to_insert.append(
                [
                    str(d_ids[i]),
                    int(p_idxs[i]),
                    dataset_type,
                    snapshot_id,
                    snapshot_tag,
                    feats_map,
                ]
            )

        cols = [
            "dialog_id",
            "participant_index",
            "dataset_type",
            "snapshot_id",
            "snapshot_tag",
            "features",
        ]

        for start in range(0, len(rows_to_insert), batch_size):
            chunk = rows_to_insert[start : start + batch_size]
            self.client.insert("dialogue_features", chunk, column_names=cols)

        logger.info(
            "Успешно сохранено %d строк признаков в срез '%s' (snapshot_id=%d, dataset_type='%s')",
            len(rows_to_insert),
            snapshot_tag,
            snapshot_id,
            dataset_type,
        )
        return snapshot_id

    def load_snapshot(
        self,
        snapshot_tag: str,
        selected_features: list[str] | None = None,
        dataset_type: str = "train",
    ) -> tuple[pl.DataFrame, list[str]]:
        """Загрузка матрицы признаков для указанного среза с проекцией колонок.

        Args:
            snapshot_tag: Тег снапшота признаков.
            selected_features: Опциональный список имен фичей (проекция). Если None - читаются все фичи снапшота.
            dataset_type: 'train' или 'test'.

        Returns:
            Tuple[pl.DataFrame, list[str]]:
                DataFrame с колонками [dialog_id, participant_index, <selected_features>] и список фичей.

        """
        if not self.has_snapshot(snapshot_tag, dataset_type):
            raise KeyError(f"Срез признаков '{snapshot_tag}' для dataset_type='{dataset_type}' не найден в ClickHouse.")

        # Если фичи не переданы, получаем полный список из каталога feature_snapshots
        if selected_features is None:
            res_meta = self.client.query(
                "SELECT feature_names FROM feature_snapshots WHERE snapshot_tag = {tag:String} LIMIT 1",
                parameters={"tag": snapshot_tag},
            ).result_rows
            if not res_meta or not res_meta[0][0]:
                raise ValueError(f"Метаданные списка признаков для '{snapshot_tag}' пусты")
            target_cols = list(res_meta[0][0])
        else:
            target_cols = list(selected_features)

        # Формируем проекцию колонок Map -> Named Columns в ClickHouse с хронологическим порядком
        map_selects = []
        for f in target_cols:
            safe_key = f.replace("\\", "\\\\").replace("'", "\\'")
            safe_alias = f.replace("`", "")
            map_selects.append(f"f.features['{safe_key}'] AS `{safe_alias}`")
        cols_query = ", ".join(map_selects)

        query = f"""
            SELECT
                f.dialog_id,
                f.participant_index,
                {cols_query}
            FROM dialogue_features AS f FINAL
            LEFT JOIN (
                SELECT dialog_id, participant_index, any(original_order_idx) AS orig_idx
                FROM raw_dialogues FINAL
                WHERE dataset_type = {{dtype:String}}
                GROUP BY dialog_id, participant_index
            ) AS r ON f.dialog_id = r.dialog_id AND f.participant_index = r.participant_index
            WHERE f.snapshot_tag = {{tag:String}} AND f.dataset_type = {{dtype:String}}
            ORDER BY r.orig_idx ASC, f.dialog_id ASC, f.participant_index ASC
        """  # noqa: S608  # Validated feature columns mapping
        result = self.client.query(query, parameters={"dtype": dataset_type, "tag": snapshot_tag})
        col_names = ["dialog_id", "participant_index", *target_cols]

        df = pl.DataFrame(result.result_rows, schema=col_names, orient="row")
        return df, target_cols

    def list_snapshots(self) -> pl.DataFrame:
        """Возвращает каталог всех зарегистрированных срезов признаков."""
        res = self.client.query(
            "SELECT snapshot_id, snapshot_tag, features_count, description, created_at FROM feature_snapshots ORDER BY snapshot_id"
        )
        return pl.DataFrame(
            res.result_rows,
            schema=["snapshot_id", "snapshot_tag", "features_count", "description", "created_at"],
            orient="row",
        )

    def drop_snapshot(self, snapshot_tag: str) -> bool:
        """Удаляет партицию снапшота из ClickHouse."""
        res = self.client.query(
            "SELECT snapshot_id FROM feature_snapshots WHERE snapshot_tag = {tag:String} LIMIT 1",
            parameters={"tag": snapshot_tag},
        ).result_rows
        if not res:
            return False

        snapshot_id = int(res[0][0])
        self.client.command(
            f"ALTER TABLE dialogue_features DROP PARTITION {snapshot_id}"
        )  # Validated partition integer
        self.client.command(
            "ALTER TABLE feature_snapshots DELETE WHERE snapshot_id = {sid:UInt32} SETTINGS mutations_sync = 1",
            parameters={"sid": snapshot_id},
        )
        logger.info("Срез '%s' (snapshot_id=%d) успешно удален из DWH", snapshot_tag, snapshot_id)
        return True
