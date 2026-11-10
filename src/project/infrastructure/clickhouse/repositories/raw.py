import json
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from clickhouse_connect.driver.client import Client

from project.core.logging import ProjectLogger
from project.infrastructure.clickhouse.connection import ClickHouseConnectionFactory

logger = ProjectLogger.get_logger(__name__)


class ClickHouseRawRepository:
    """Репозиторий сырого слоя диалогов и сообщений (Bronze Layer)."""

    def __init__(self, client: Client | None = None) -> None:
        self.client = client or ClickHouseConnectionFactory.create()

    def count_dialogues(self, dataset_type: str | None = None) -> int:
        """Возвращает число участников диалогов в raw_dialogues."""
        if dataset_type:
            safe_type = "train" if dataset_type == "train" else "test"
            return int(
                str(
                    self.client.command(
                        f"SELECT count() FROM raw_dialogues WHERE dataset_type = '{safe_type}'"  # noqa: S608
                    )
                )
            )
        return int(str(self.client.command("SELECT count() FROM raw_dialogues")))

    def has_dialogues(self, dataset_type: str) -> bool:
        """Проверяет наличие сырых данных для указанного типа датасета (train/test)."""
        return self.count_dialogues(dataset_type) > 0

    def ingest_dialogues(
        self,
        dataset_type: str,
        json_path: Path | str,
        labels_csv_path: Path | str,
        batch_size: int = 2000,
    ) -> int:
        """Загрузка исходных JSON-диалогов и CSV-разметки в raw_dialogues и raw_dialogue_messages."""
        j_path = Path(json_path)
        y_path = Path(labels_csv_path)

        if not j_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Файлы сырых данных не найдены: {j_path} или {y_path}")

        logger.info("Чтение сырых файлов для '%s': %s, %s", dataset_type, j_path, y_path)
        with open(j_path, encoding="utf-8") as f:
            raw_dialogs = json.load(f)

        df_labels = pd.read_csv(y_path)
        dialogues_rows: list[list[Any]] = []
        messages_rows: list[list[Any]] = []

        seen_dialog_ids: set[str] = set()

        # Создаем маппинг исходного порядка диалогов из JSON
        order_map: dict[str, int] = {str(k): idx for idx, k in enumerate(raw_dialogs.keys())}

        for _, row in df_labels.iterrows():
            d_id = str(row["dialog_id"])
            p_idx = int(row["participant_index"])
            is_bot = int(row["is_bot"]) if "is_bot" in row and pd.notna(row["is_bot"]) else None
            orig_order = order_map.get(d_id, 0)

            msgs = raw_dialogs.get(d_id, [])
            num_msgs = len(msgs)

            # Оценка длительности по секундам (если есть timestamp)
            duration_sec = 0
            if num_msgs > 1 and "timestamp" in msgs[0] and "timestamp" in msgs[-1]:
                try:
                    duration_sec = max(0, int(msgs[-1]["timestamp"] - msgs[0]["timestamp"]))
                except Exception:  # noqa: BLE001  # Resilient fallback
                    duration_sec = 0

            dialogues_rows.append(
                [
                    dataset_type,
                    d_id,
                    p_idx,
                    is_bot,
                    num_msgs,
                    duration_sec,
                    json.dumps(msgs, ensure_ascii=False),
                    orig_order,
                ]
            )

            # Нормализованные сообщения диалога (пишем один раз на dialog_id)
            if d_id not in seen_dialog_ids:
                seen_dialog_ids.add(d_id)
                for msg_idx, m in enumerate(msgs):
                    txt = m.get("text", "")
                    sender = int(m.get("participant_index", 0))
                    messages_rows.append(
                        [
                            dataset_type,
                            d_id,
                            msg_idx,
                            sender,
                            txt,
                            len(txt),
                            orig_order,
                        ]
                    )

        # Пакетная вставка в ClickHouse
        dlg_cols = [
            "dataset_type",
            "dialog_id",
            "participant_index",
            "is_bot",
            "num_messages",
            "duration_sec",
            "messages_json",
            "original_order_idx",
        ]
        self.client.insert("raw_dialogues", dialogues_rows, column_names=dlg_cols)

        msg_cols = [
            "dataset_type",
            "dialog_id",
            "msg_idx",
            "participant_index",
            "text",
            "char_count",
            "original_order_idx",
        ]
        for start in range(0, len(messages_rows), batch_size):
            chunk = messages_rows[start : start + batch_size]
            self.client.insert("raw_dialogue_messages", chunk, column_names=msg_cols)

        logger.info(
            "Успешно загружено в DWH: %d строк диалогов и %d сообщений для '%s'",
            len(dialogues_rows),
            len(messages_rows),
            dataset_type,
        )
        return len(dialogues_rows)

    def load_dialogues_dict(self, dataset_type: str) -> dict[str, list[dict[str, Any]]]:
        """Возвращает словарь {dialog_id: messages} для указанного датасета с сохранением хронологии."""
        res = self.client.query(
            "SELECT dialog_id, any(messages_json) FROM raw_dialogues FINAL WHERE dataset_type = {dtype:String} GROUP BY dialog_id ORDER BY any(original_order_idx) ASC",
            parameters={"dtype": dataset_type},
        ).result_rows
        return {str(row[0]): json.loads(row[1]) for row in res}

    def load_labels(self, dataset_type: str) -> pl.DataFrame:
        """Возвращает Polars DataFrame с разметкой (dialog_id, participant_index, is_bot) в хронологическом порядке."""
        res = self.client.query(
            "SELECT dialog_id, participant_index, is_bot FROM raw_dialogues FINAL WHERE dataset_type = {dtype:String} ORDER BY original_order_idx ASC, participant_index ASC",
            parameters={"dtype": dataset_type},
        )
        df = pl.DataFrame(
            res.result_rows,
            schema=["dialog_id", "participant_index", "is_bot"],
            orient="row",
        )
        if dataset_type == "test":
            df = df.with_columns((pl.col("dialog_id") + "_" + pl.col("participant_index").cast(pl.String)).alias("ID"))
        return df

    def load_participant_texts(self, dataset_type: str = "train", limit: int | None = None) -> pl.DataFrame:
        """Возвращает Polars DataFrame с агрегированными текстами реплик участников из DWH."""
        limit_clause = f"LIMIT {limit}" if limit is not None else ""
        query = f"""
            SELECT
                d.dialog_id,
                d.participant_index,
                d.is_bot,
                arrayStringConcat(groupArray(m.text), ' ') AS text
            FROM (
                SELECT dataset_type, dialog_id, participant_index, text
                FROM raw_dialogue_messages
                WHERE dataset_type = {{dtype:String}}
                ORDER BY dialog_id, participant_index, msg_idx
            ) m
            RIGHT JOIN raw_dialogues AS d FINAL
                ON d.dialog_id = m.dialog_id
                AND d.participant_index = m.participant_index
                AND d.dataset_type = m.dataset_type
            WHERE d.dataset_type = {{dtype:String}}
            GROUP BY d.dialog_id, d.participant_index, d.is_bot
            ORDER BY d.dialog_id, d.participant_index
            {limit_clause}
        """  # noqa: S608  # Validated limit and parameterized dataset_type
        res = self.client.query(query, parameters={"dtype": dataset_type})
        df = pl.DataFrame(
            res.result_rows,
            schema=["dialog_id", "participant_index", "is_bot", "text"],
            orient="row",
        )
        if dataset_type == "test":
            df = df.with_columns((pl.col("dialog_id") + "_" + pl.col("participant_index").cast(pl.String)).alias("ID"))
        return df
