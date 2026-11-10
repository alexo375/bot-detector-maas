from typing import Any

import numpy as np
import polars as pl
from clickhouse_connect.driver.client import Client

from project.core.logging import ProjectLogger
from project.infrastructure.clickhouse.connection import ClickHouseConnectionFactory

logger = ProjectLogger.get_logger(__name__)


class ClickHousePredictionRepository:
    """Репозиторий предсказаний моделей (OOF кросс-валидации и тестовых скоров)."""

    def __init__(self, client: Client | None = None) -> None:
        self.client = client or ClickHouseConnectionFactory.create()

    def has_predictions(self, run_id: str, dataset_split: str = "train_oof") -> bool:
        """Проверяет наличие предсказаний для конкретного run_id."""
        cnt = self.client.command(
            "SELECT count() FROM model_predictions WHERE run_id = {rid:String} AND dataset_split = {split:String}",
            parameters={"rid": run_id, "split": dataset_split},
        )
        return int(str(cnt)) > 0

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
        batch_size: int = 5000,
    ) -> int:
        """Сохранение OOF или Test предсказаний модели.

        Args:
            run_id: UUID рана из MLflow.
            run_name: Имя рана (например, 'catboost_top_pruned').
            dataset_split: 'train_oof' или 'test'.
            dialog_ids: Массив ID диалогов.
            participant_indices: Массив индексов участников (0 или 1).
            prob_bot: Сырые вероятности бота от модели.
            prob_bot_calibrated: Откалиброванные парные вероятности (PairwisePostprocessor).
            is_bot: Таргетные метки (для train_oof).
            fold_indices: Индексы фолдов кросс-валидации (для train_oof).
            batch_size: Размер батча вставки.

        Returns:
            Число сохраненных строк.

        """
        n_rows = len(dialog_ids)
        calibrated = prob_bot_calibrated if prob_bot_calibrated is not None else prob_bot
        folds = fold_indices if fold_indices is not None else np.full(n_rows, 255, dtype=np.uint8)

        rows_to_insert: list[list[Any]] = []
        for i in range(n_rows):
            raw_target = is_bot[i] if is_bot is not None else None
            target_val = (
                int(raw_target)
                if raw_target is not None
                and not (isinstance(raw_target, (float, np.floating)) and np.isnan(raw_target))
                else None
            )
            rows_to_insert.append(
                [
                    str(run_id),
                    str(run_name),
                    str(dataset_split),
                    str(dialog_ids[i]),
                    int(participant_indices[i]),
                    int(folds[i]),
                    float(prob_bot[i]),
                    float(calibrated[i]),
                    target_val,
                ]
            )

        cols = [
            "run_id",
            "run_name",
            "dataset_split",
            "dialog_id",
            "participant_index",
            "fold_idx",
            "prob_bot",
            "prob_bot_calibrated",
            "is_bot",
        ]

        for start in range(0, len(rows_to_insert), batch_size):
            chunk = rows_to_insert[start : start + batch_size]
            self.client.insert("model_predictions", chunk, column_names=cols)

        logger.info(
            "Сохранено %d предсказаний для run_id='%s' (%s, split='%s')",
            n_rows,
            run_id,
            run_name,
            dataset_split,
        )
        return n_rows

    def load_predictions(
        self,
        run_id: str,
        dataset_split: str = "train_oof",
    ) -> pl.DataFrame:
        """Загрузка предсказаний одной конкретной модели в хронологическом порядке."""
        query = """
            SELECT
                p.dialog_id,
                p.participant_index,
                p.fold_idx,
                p.prob_bot,
                p.prob_bot_calibrated,
                p.is_bot
            FROM model_predictions AS p FINAL
            LEFT JOIN (
                SELECT dialog_id, participant_index, any(original_order_idx) AS orig_idx
                FROM raw_dialogues
                GROUP BY dialog_id, participant_index
            ) AS r ON p.dialog_id = r.dialog_id AND p.participant_index = r.participant_index
            WHERE p.run_id = {rid:String} AND p.dataset_split = {split:String}
            ORDER BY r.orig_idx ASC, p.dialog_id ASC, p.participant_index ASC
        """
        res = self.client.query(query, parameters={"rid": run_id, "split": dataset_split})
        return pl.DataFrame(
            res.result_rows,
            schema=["dialog_id", "participant_index", "fold_idx", "prob_bot", "prob_bot_calibrated", "is_bot"],
            orient="row",
        )

    def load_meta_dataset(
        self,
        run_ids: list[str],
        dataset_split: str = "train_oof",
        use_calibrated: bool = True,
    ) -> pl.DataFrame:
        """Формирует объединенный датасет OOF-предсказаний для Level-2 Meta-Stacker.

        Выполняет многомодельный INNER JOIN в ClickHouse по (dialog_id, participant_index) в хронологическом порядке.
        """
        if not run_ids:
            raise ValueError("Список run_ids не может быть пустым")

        prob_col = "prob_bot_calibrated" if use_calibrated else "prob_bot"

        run_names_map: dict[str, str] = {}
        used_names: set[str] = set()
        for idx, rid in enumerate(run_ids):
            r = self.client.query(
                "SELECT run_name FROM model_predictions FINAL WHERE run_id = {rid:String} LIMIT 1",
                parameters={"rid": rid},
            ).result_rows
            base_candidate = (str(r[0][0]) if r else rid[:8]).replace("`", "")
            alias = base_candidate if base_candidate not in used_names else f"{base_candidate}_{idx}"
            used_names.add(alias)
            run_names_map[rid] = alias

        base_rid = run_ids[0]
        base_name = run_names_map[base_rid]

        select_clauses = [
            "m0.dialog_id AS dialog_id",
            "m0.participant_index AS participant_index",
            "m0.is_bot AS is_bot",
            f"m0.{prob_col} AS `prob_{base_name}`",
        ]

        sql_params: dict[str, str] = {
            "dataset_split": dataset_split,
            "rid_0": base_rid,
        }
        join_clauses = []
        for idx, rid in enumerate(run_ids[1:], start=1):
            rname = run_names_map[rid]
            p_key = f"rid_{idx}"
            sql_params[p_key] = rid
            select_clauses.append(f"m{idx}.{prob_col} AS `prob_{rname}`")
            join_clauses.append(f"""
                INNER JOIN (
                    SELECT dialog_id, participant_index, {prob_col}
                    FROM model_predictions FINAL
                    WHERE run_id = {{{p_key}:String}} AND dataset_split = {{dataset_split:String}}
                ) AS m{idx} ON m0.dialog_id = m{idx}.dialog_id AND m0.participant_index = m{idx}.participant_index
            """)  # noqa: S608  # Dynamically joined validated model predictions

        sql = f"""
            SELECT
                {", ".join(select_clauses)}
            FROM (
                SELECT dialog_id, participant_index, is_bot, {prob_col}
                FROM model_predictions FINAL
                WHERE run_id = {{rid_0:String}} AND dataset_split = {{dataset_split:String}}
            ) AS m0
            {" ".join(join_clauses)}
            LEFT JOIN (
                SELECT dialog_id, participant_index, any(original_order_idx) AS orig_idx
                FROM raw_dialogues FINAL
                GROUP BY dialog_id, participant_index
            ) AS r ON m0.dialog_id = r.dialog_id AND m0.participant_index = r.participant_index
            ORDER BY r.orig_idx ASC, m0.dialog_id ASC, m0.participant_index ASC
        """  # noqa: S608  # Validated SQL join construction

        res = self.client.query(sql, parameters=sql_params)
        cols = ["dialog_id", "participant_index", "is_bot"] + [f"prob_{run_names_map[rid]}" for rid in run_ids]
        return pl.DataFrame(res.result_rows, schema=cols, orient="row")

    def list_prediction_runs(self) -> pl.DataFrame:
        """Возвращает сводку по всем сохраненным OOF и тестовым прогонам моделей."""
        query = """
            SELECT
                run_id,
                run_name,
                dataset_split,
                count() AS rows_count,
                min(created_at) AS created_at
            FROM model_predictions FINAL
            GROUP BY run_id, run_name, dataset_split
            ORDER BY created_at DESC
        """
        res = self.client.query(query)
        return pl.DataFrame(
            res.result_rows,
            schema=["run_id", "run_name", "dataset_split", "rows_count", "created_at"],
            orient="row",
        )

    def delete_predictions(self, run_id: str) -> None:
        """Удаляет предсказания указанного рана из ClickHouse синхронно."""
        self.client.command(
            "ALTER TABLE model_predictions DELETE WHERE run_id = {rid:String} SETTINGS mutations_sync = 1",
            parameters={"rid": run_id},
        )
        logger.info("Предсказания для run_id='%s' удалены из ClickHouse", run_id)
