"""Модуль безопасного отбора высокоуверенных диалогов для pseudo-labeling из chat_logs."""

import json
from typing import Any

from project.core.logging import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class PseudoLabeler:
    """Безопасный отбор высокоуверенных продакшн-диалогов из `chat_logs` для pseudo-labeling."""

    @staticmethod
    def _build_balanced_pairs(
        chat_rows: list[Any],
        max_pairs: int,
        high_conf: float,
    ) -> list[tuple[str, Any, int, int, bool]]:
        bot_pairs: list[tuple[tuple[str, Any, int, int, bool], tuple[str, Any, int, int, bool]]] = []
        human_pairs: list[tuple[tuple[str, Any, int, int, bool], tuple[str, Any, int, int, bool]]] = []
        seen_dialog_ids: set[str] = set()

        for d_id_raw, messages_raw, p0_raw, p1_raw in chat_rows:
            raw_key = str(d_id_raw)
            if raw_key in seen_dialog_ids:
                continue
            msgs = json.loads(messages_raw) if isinstance(messages_raw, str) else messages_raw
            if not isinstance(msgs, list) or len(msgs) < 2:
                continue
            seen_dialog_ids.add(raw_key)
            d_id = f"pseudo_{raw_key}"
            y0 = 1 if float(p0_raw) >= high_conf else 0
            y1 = 1 if float(p1_raw) >= high_conf else 0
            pair = ((d_id, msgs, 0, y0, True), (d_id, msgs, 1, y1, True))
            (bot_pairs if (y0 == 1 or y1 == 1) else human_pairs).append(pair)

        half_pairs = min(max_pairs // 2, len(bot_pairs), len(human_pairs)) if bot_pairs and human_pairs else 0
        selected = (
            bot_pairs[:half_pairs] + human_pairs[:half_pairs]
            if half_pairs > 0
            else (bot_pairs + human_pairs)[:max_pairs]
        )
        rows: list[tuple[str, Any, int, int, bool]] = []
        for r0, r1 in selected:
            rows.extend((r0, r1))
        return rows

    @classmethod
    def fetch_rows(
        cls,
        client: Any,
        gold_rows_count: int,
        max_pseudo_ratio: float = 0.30,
        high_conf: float = 0.90,
        low_conf: float = 0.10,
    ) -> list[tuple[str, Any, int, int, bool]]:
        """Запрос и сбалансированный отбор уверенных пар диалогов из ClickHouse `chat_logs`."""
        max_pseudo_rows = max(0, int(gold_rows_count * max_pseudo_ratio))
        max_pairs = max_pseudo_rows // 2
        if max_pairs < 1:
            return []

        query = (
            "SELECT dialog_id, messages_json, prob_p0, prob_p1 "
            "FROM chat_logs "
            "WHERE NOT startsWith(dialog_id, 'drift_inject_') "
            "AND length(messages_json) > 2 "
            "AND (prob_p0 >= {hi:Float64} OR prob_p0 <= {lo:Float64}) "
            "AND (prob_p1 >= {hi:Float64} OR prob_p1 <= {lo:Float64}) "
            "AND (prob_p0 + prob_p1 <= 1.05) "
            "ORDER BY timestamp DESC "
            "LIMIT {lim:UInt32}"
        )
        try:
            chat_rows = client.query(
                query,
                parameters={"hi": float(high_conf), "lo": float(low_conf), "lim": int(max_pairs * 4)},
            ).result_rows
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pseudo-labeling query from chat_logs skipped: %s", exc)
            return []

        return cls._build_balanced_pairs(chat_rows, max_pairs=max_pairs, high_conf=high_conf)


__all__ = ["PseudoLabeler"]
