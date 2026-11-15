"""Сервис загрузки обучающих диалогов и подготовки TrainingBundle для ретрейна."""

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from project.core.config import config
from project.core.logging import ProjectLogger
from project.features.store import FeatureStore
from project.infrastructure.clickhouse.connection import ClickHouseConnectionFactory
from project.pipeline.mlflow_retrainer.components.pseudo_labeler import PseudoLabeler

logger = ProjectLogger.get_logger(__name__)


@dataclass
class TrainingBundle:
    """Полный пакет данных для адаптивного дообучения с поддержкой pseudo-labeling и диалогов."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    texts_df: pl.DataFrame
    dialogues: list[list[dict[str, Any]]]
    is_pseudo: np.ndarray


class TrainingDataLoader:
    """Сервис загрузки диалогов из ClickHouse DWH (с опциональным pseudo-labeling) и FeatureStore."""

    def __init__(
        self,
        feature_store: FeatureStore | None = None,
        client: Any = None,
    ) -> None:
        self.store = feature_store or FeatureStore(auto_load_from_repo=False)
        self.client = client

    @staticmethod
    def _format_participant_text(messages: list[dict[str, Any]], p_idx: int) -> str:
        p_text = " ".join(m.get("text", "") for m in messages if m.get("participant_index") == p_idx)
        if not p_text.strip():
            p_text = " ".join(m.get("text", "") for m in messages) or "Привет"
        return p_text

    @staticmethod
    def _enforce_paired_records(rows: Sequence[Sequence[Any]]) -> list[tuple[str, Any, int, int, bool]]:
        """Упорядочивает записи в полные пары `(p=0, p=1)`, сохраняя порядок диалогов."""
        grouped: dict[str, dict[int, tuple[Any, int, bool]]] = defaultdict(dict)
        order: list[str] = []
        raw_normalized: list[tuple[str, Any, int, int, bool]] = []

        for row in rows:
            d_id = str(row[0])
            msgs = json.loads(row[1]) if isinstance(row[1], str) else row[1]
            p_idx = int(row[2])
            is_bot = int(row[3])
            is_pseudo = bool(row[4]) if len(row) > 4 else False
            if d_id not in grouped:
                order.append(d_id)
            grouped[d_id][p_idx] = (msgs, is_bot, is_pseudo)
            raw_normalized.append((d_id, msgs, p_idx, is_bot, is_pseudo))

        paired_rows: list[tuple[str, Any, int, int, bool]] = []
        for d_id in order:
            p_map = grouped[d_id]
            if 0 in p_map and 1 in p_map:
                for p_idx in (0, 1):
                    msgs, is_bot, is_pseudo = p_map[p_idx]
                    paired_rows.append((d_id, msgs, p_idx, is_bot, is_pseudo))

        return paired_rows if paired_rows else raw_normalized

    def _build_bundle_from_records(
        self,
        rows: Sequence[Sequence[Any]],
        target_features: list[str],
    ) -> TrainingBundle:
        normalized = self._enforce_paired_records(rows)
        dialogues_map: dict[str, list[dict[str, Any]]] = {}
        ordered_dialog_ids: list[str] = []
        labels: list[int] = []
        pseudo_flags: list[bool] = []
        text_records: list[dict[str, Any]] = []

        for dialog_id, messages, p_idx, is_bot, is_pseudo in normalized:
            if dialog_id not in dialogues_map:
                dialogues_map[dialog_id] = messages
                ordered_dialog_ids.append(dialog_id)
            labels.append(is_bot)
            pseudo_flags.append(is_pseudo)
            text_records.append(
                {
                    "dialog_id": dialog_id,
                    "participant_index": p_idx,
                    "text": self._format_participant_text(messages, p_idx),
                    "is_bot": is_bot,
                    "is_pseudo": is_pseudo,
                }
            )

        dialogues_list = [dialogues_map[d_id] for d_id in ordered_dialog_ids]
        X = self.store.transform(dialogues_list, selected_features=target_features)
        y = np.array(labels[: len(X)], dtype=int)
        is_pseudo_arr = np.array(pseudo_flags[: len(X)], dtype=bool)
        texts_df = pl.DataFrame(text_records[: len(X)])
        return TrainingBundle(
            X=X,
            y=y,
            feature_names=target_features,
            texts_df=texts_df,
            dialogues=dialogues_list,
            is_pseudo=is_pseudo_arr,
        )

    def load_retraining_bundle(
        self,
        sample_size: int | None = None,
        include_pseudo_labels: bool = True,
        max_pseudo_ratio: float = 0.30,
    ) -> TrainingBundle:
        """Строгая загрузка обучающего `TrainingBundle` из ClickHouse `raw_dialogues` (без фолбэков)."""
        effective_size = sample_size if sample_size is not None else config.retrain.sample_size
        target_features = list(self.store.feature_names)
        aligned_limit = max(2, (int(effective_size) // 2) * 2)

        client = self.client or ClickHouseConnectionFactory.create(
            host=config.clickhouse.host,
            port=config.clickhouse.port,
            database=config.clickhouse.database,
            connect_timeout=3,
            send_receive_timeout=15,
        )
        query = (
            "SELECT dialog_id, messages_json, participant_index, is_bot "
            "FROM raw_dialogues FINAL "
            "WHERE dataset_type = 'train' AND is_bot IS NOT NULL "
            "ORDER BY original_order_idx ASC, dialog_id ASC, participant_index ASC "
            "LIMIT {lim:UInt32}"
        )
        rows = client.query(query, parameters={"lim": aligned_limit}).result_rows
        if len(rows) < 10:
            raise RuntimeError(f"ClickHouse raw_dialogues (train) has only {len(rows)} rows (minimum 10 required).")

        gold_rows = [(r[0], r[1], r[2], r[3], False) for r in rows]
        all_rows = list(gold_rows)
        if include_pseudo_labels:
            pseudo_rows = PseudoLabeler.fetch_rows(
                client=client,
                gold_rows_count=len(gold_rows),
                max_pseudo_ratio=max_pseudo_ratio,
            )
            all_rows.extend(pseudo_rows)
            if pseudo_rows:
                logger.info(
                    "Added %d high-confidence pseudo-labeled participant rows from chat_logs",
                    len(pseudo_rows),
                )

        bundle = self._build_bundle_from_records(all_rows, target_features)
        logger.info(
            "Loaded %d training samples (%d gold, %d pseudo) from ClickHouse",
            len(bundle.X),
            int(np.sum(~bundle.is_pseudo)),
            int(np.sum(bundle.is_pseudo)),
        )
        return bundle


__all__ = ["TrainingBundle", "TrainingDataLoader"]
