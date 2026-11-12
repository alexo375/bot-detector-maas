from typing import Any

import numpy as np
import polars as pl

from project.core.config import config
from project.core.logging import ProjectLogger
from project.features.store import FeatureStore
from project.infrastructure.clickhouse.connection import ClickHouseConnectionFactory

logger = ProjectLogger.get_logger(__name__)


class DWHFeatureFetcher:
    """Сервис выгрузки рабочих признаков и вероятностей предсказаний из ClickHouse DWH."""

    def __init__(
        self,
        feature_store: FeatureStore | None = None,
        client: Any = None,
    ) -> None:
        self._store = feature_store
        self.client = client

    @property
    def store(self) -> FeatureStore:
        """Ленивая инициализация FeatureStore (загружается из Triton-репозитория при первом обращении)."""
        if self._store is None:
            self._store = FeatureStore(auto_load_from_repo=True)
        return self._store

    @store.setter
    def store(self, value: FeatureStore) -> None:
        self._store = value

    def get_feature_names(self) -> list[str]:
        """Получение канонического списка активных признаков из FeatureStore."""
        if self.store.auto_load_from_repo:
            self.store.reload_if_changed()
        return list(self.store.feature_names)

    def _get_client(self) -> Any:
        return self.client or ClickHouseConnectionFactory.create(
            host=config.clickhouse.host,
            port=config.clickhouse.port,
            database=config.clickhouse.database,
            connect_timeout=3,
            send_receive_timeout=15,
        )

    def load_or_create_reference_dataset(
        self,
        sample_size: int | None = None,
    ) -> pl.DataFrame:
        """Строгая загрузка эталонного распределения признаков из ClickHouse `dialogue_features`."""
        feature_names = self.get_feature_names()

        try:
            client = self._get_client()
            if sample_size is not None:
                query = (
                    "SELECT features FROM dialogue_features FINAL "
                    "WHERE snapshot_tag = 'candidate_features' AND dataset_type = 'train' "
                    "ORDER BY dialog_id ASC, participant_index ASC "
                    "LIMIT {lim:UInt32}"
                )
                result = client.query(query, parameters={"lim": int(sample_size)})
            else:
                query = (
                    "SELECT features FROM dialogue_features FINAL "
                    "WHERE snapshot_tag = 'candidate_features' AND dataset_type = 'train'"
                )
                result = client.query(query)
            rows = result.result_rows

            if len(rows) < 10:
                raise RuntimeError(
                    f"ClickHouse dialogue_features contains only {len(rows)} rows (minimum 10 required). "
                    "Cannot perform data drift analysis without sufficient reference distribution."
                )

            matrix: list[list[float]] = []
            for row in rows:
                val = row[0]
                if not isinstance(val, dict):
                    raise TypeError(f"Expected Map/dict in dialogue_features.features, got {type(val).__name__}")
                matrix.append([float(val[fname]) for fname in feature_names])

            logger.info("Loaded %d reference feature vectors from ClickHouse dialogue_features", len(matrix))
            return pl.DataFrame(matrix, schema=feature_names, orient="row")
        except Exception as e:
            raise RuntimeError(f"ClickHouse reference_features fetch failed: {e}") from e

    def _build_time_filter(self, days: int, since_ts: float | None) -> tuple[str, dict[str, Any]]:
        """Формирует SQL-условие по окну дней и опциональной отсечке времени создания чемпиона."""
        clause = f"timestamp >= now() - INTERVAL {int(days)} DAY"
        params: dict[str, Any] = {}
        if since_ts is not None and since_ts > 0:
            clause += " AND timestamp >= toDateTime64({since_ts:Float64}, 3, 'UTC')"
            params["since_ts"] = float(since_ts)
        return clause, params

    def fetch_current_features(
        self,
        days: int = 7,
        limit: int | None = None,
        since_ts: float | None = None,
    ) -> pl.DataFrame:
        """Выгрузка текущих признаков из ClickHouse chat_logs за окно (или с момента промоушена чемпиона)."""
        feature_names = self.get_feature_names()

        try:
            client = self._get_client()
            time_clause, query_params = self._build_time_filter(days, since_ts)
            base_query = (
                f"SELECT features_p0, features_p1 FROM chat_logs "  # noqa: S608
                f"WHERE {time_clause} "
                f"AND length(features_p0) = {len(feature_names)} AND length(features_p1) = {len(feature_names)} "
                f"ORDER BY timestamp DESC"
            )
            if limit is not None:
                base_query += " LIMIT {lim:UInt32}"
                query_params["lim"] = int(limit)
            result = client.query(base_query, parameters=query_params if query_params else None)
            rows = result.result_rows

            if len(rows) < 10:
                raise RuntimeError(
                    f"ClickHouse chat_logs contains only {len(rows)} dialogues for the last {days} days "
                    "(minimum 10 required). Cannot perform data drift analysis without sufficient production data."
                )

            extracted = []
            for r in rows:
                if len(r[0]) == len(feature_names):
                    extracted.append(r[0])
                if len(r[1]) == len(feature_names):
                    extracted.append(r[1])
            logger.info("Fetched %d participant vectors from ClickHouse DWH for last %d days", len(extracted), days)
            return pl.DataFrame(extracted, schema=feature_names, orient="row")
        except Exception as e:
            raise RuntimeError(f"ClickHouse chat_logs fetch failed: {e}") from e

    def fetch_prediction_data(
        self,
        days: int = 7,
        limit: int | None = None,
        since_ts: float | None = None,
        reference_run_id: str | None = None,
    ) -> tuple[Any, Any, list[str]]:
        """Строгая выгрузка эталонных (`model_predictions`) и текущих (`chat_logs`) предсказаний из DWH."""
        client = self._get_client()
        time_clause, query_params = self._build_time_filter(days, since_ts)
        base_query = (
            f"SELECT prob_p0, prob_p1, verdict FROM chat_logs "  # noqa: S608
            f"WHERE {time_clause} "
            f"ORDER BY timestamp DESC"
        )
        if limit is not None:
            base_query += " LIMIT {lim:UInt32}"
            query_params["lim"] = int(limit)
        rows = client.query(base_query, parameters=query_params if query_params else None).result_rows

        cur_probs: list[float] = []
        cur_verdicts: list[str] = []
        for r in rows:
            cur_probs.extend([float(r[0]), float(r[1])])
            cur_verdicts.append(str(r[2]))

        if reference_run_id:
            target_rid = str(reference_run_id)
        else:
            rid_rows = client.query(
                "SELECT run_id FROM model_predictions FINAL "
                "WHERE dataset_split = 'train_oof' AND run_name = 'logistic_meta_stacker' "
                "ORDER BY created_at DESC LIMIT 1"
            ).result_rows
            if not rid_rows:
                return np.array([], dtype=np.float64), np.array([], dtype=np.float64), []
            target_rid = str(rid_rows[0][0])

        ref_rows = client.query(
            "SELECT prob_bot_calibrated FROM model_predictions FINAL "
            "WHERE dataset_split = 'train_oof' AND run_id = {rid:String} "
            "ORDER BY dialog_id, participant_index LIMIT 500",
            parameters={"rid": target_rid},
        ).result_rows
        ref_list = [float(r[0]) for r in ref_rows if r[0] is not None]

        if len(ref_list) < 10 or len(cur_probs) < 10:
            return np.array(ref_list, dtype=np.float64), np.array([], dtype=np.float64), []

        return np.array(ref_list, dtype=np.float64), np.array(cur_probs, dtype=np.float64), cur_verdicts

    def fetch_concept_drift_probs(
        self,
        days: int = 7,
        since_ts: float | None = None,
        limit: int = 200,
    ) -> dict[str, np.ndarray]:
        """Строгая выгрузка сохраненных вероятностей базовых моделей из `chat_logs` для Concept Drift."""
        client = self._get_client()
        time_clause, query_params = self._build_time_filter(days, since_ts)
        query_params["lim"] = int(limit)
        query = (
            f"SELECT base_model_probs FROM chat_logs "  # noqa: S608
            f"WHERE {time_clause} AND notEmpty(base_model_probs) "
            f"ORDER BY timestamp DESC LIMIT {{lim:UInt32}}"
        )
        rows = client.query(query, parameters=query_params).result_rows
        if not rows:
            return {}

        valid_maps = [raw_map for (raw_map,) in rows if isinstance(raw_map, dict) and raw_map]
        if not valid_maps:
            return {}

        expected_keys = sorted(str(k) for k in valid_maps[0])
        model_lists: dict[str, list[float]] = {k: [] for k in expected_keys}
        for raw_map in valid_maps:
            if not all(k in raw_map and isinstance(raw_map[k], (list, tuple)) for k in expected_keys):
                continue
            lengths = {len(raw_map[k]) for k in expected_keys}
            if len(lengths) != 1 or next(iter(lengths)) == 0:
                continue
            for k in expected_keys:
                model_lists[k].extend(float(p) for p in raw_map[k])

        if not model_lists or len(next(iter(model_lists.values()))) < 2:
            return {}
        return {name: np.asarray(vals, dtype=np.float64) for name, vals in model_lists.items()}

    def count_recent_dialogues(self, days: int = 7, since_ts: float | None = None) -> int:
        """Строгий подсчет количества валидных продакшн-диалогов в `chat_logs` за окно."""
        feature_names = self.get_feature_names()
        client = self._get_client()
        time_clause, query_params = self._build_time_filter(days, since_ts)
        query = (
            f"SELECT count() FROM chat_logs "  # noqa: S608
            f"WHERE {time_clause} "
            f"AND length(features_p0) = {len(feature_names)} AND length(features_p1) = {len(feature_names)}"
        )
        return int(client.query(query, parameters=query_params if query_params else None).result_rows[0][0])
