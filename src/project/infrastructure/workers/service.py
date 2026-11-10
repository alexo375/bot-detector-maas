"""Сервис конвейера инференса диалогов.

Инкапсулирует извлечение фичей, вызов Triton, калибровку вероятностей,
кэширование в Redis и подготовку событий для ClickHouse DWH.
"""

import time
from typing import Any

import httpx
import numpy as np

from project.core.config import config
from project.core.logging import ProjectLogger
from project.core.schemas import AuditMode, DialogueResultItem, ParticipantVerdict, PredictResponse
from project.core.utils import DialogueHasher
from project.features.store import FeatureStore
from project.infrastructure.clickhouse import ClickHouseAuditWriter
from project.infrastructure.rabbitmq import AuditLogPublisher, DialogueEvent
from project.infrastructure.rabbitmq.schemas.task import InferenceTask
from project.infrastructure.redis import DialogueCache
from project.infrastructure.triton.client import TritonEnsembleClient
from project.models.postprocessors.pairwise import PairwisePostprocessor

logger = ProjectLogger.get_logger(__name__)


class WebhookNotifier:
    """Асинхронный отправитель HTTP POST webhook уведомлений о результатах инференса."""

    @staticmethod
    async def send(url: str, payload: dict[str, Any]) -> None:
        """HTTP POST уведомление на webhook клиента."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                logger.info("Webhook successfully delivered to %s (HTTP %d)", url, resp.status_code)
        except Exception as exc:  # noqa: BLE001  # Resilient fallback
            logger.error("Failed to deliver webhook to %s: %s", url, exc)


class InferencePipelineService:
    """Сервис выполнения сквозного пайплайна инференса для пачки диалогов."""

    def __init__(
        self,
        triton_client: TritonEnsembleClient | None = None,
        feature_store: FeatureStore | None = None,
        postprocessor: PairwisePostprocessor | None = None,
        cache: DialogueCache | None = None,
        publisher: AuditLogPublisher | None = None,
        audit_writer: ClickHouseAuditWriter | None = None,
    ) -> None:
        self.triton_client = triton_client or TritonEnsembleClient()
        self.feature_store = feature_store or FeatureStore(auto_load_from_repo=True)
        self.postprocessor = postprocessor or PairwisePostprocessor()
        self.cache = cache or DialogueCache()
        self.audit_writer = audit_writer or ClickHouseAuditWriter(publisher=publisher)

    def _infer_cache_misses(
        self,
        dialogue_msgs: list[list[dict[str, Any]]],
        miss_indices: list[int],
    ) -> tuple[
        dict[int, tuple[float, float]], dict[int, tuple[list[float], list[float]]], dict[int, dict[str, list[float]]]
    ]:
        p_calibrated_map: dict[int, tuple[float, float]] = {}
        features_map: dict[int, tuple[list[float], list[float]]] = {}
        base_probs_map: dict[int, dict[str, list[float]]] = {}
        if not miss_indices:
            return p_calibrated_map, features_map, base_probs_map

        miss_msgs = [dialogue_msgs[idx] for idx in miss_indices]
        features_miss = self.feature_store.transform(miss_msgs)
        p_meta, p_cb, p_lgb, p_xgb = self.triton_client.infer(features_miss)

        n_miss = len(miss_indices)
        p_calibrated_miss = self.postprocessor.transform(
            p_meta,
            dialog_ids=np.repeat([f"d_{i}" for i in range(n_miss)], 2),
            participant_indices=np.tile([0, 1], n_miss),
        )
        for m_pos, orig_idx in enumerate(miss_indices):
            idx0, idx1 = 2 * m_pos, 2 * m_pos + 1
            p_calibrated_map[orig_idx] = (
                round(float(p_calibrated_miss[idx0]), 4),
                round(float(p_calibrated_miss[idx1]), 4),
            )
            features_map[orig_idx] = (
                list(map(float, features_miss[idx0])) if idx0 < len(features_miss) else [],
                list(map(float, features_miss[idx1])) if idx1 < len(features_miss) else [],
            )
            base_probs_map[orig_idx] = {
                "catboost": [round(float(p_cb[idx0]), 6), round(float(p_cb[idx1]), 6)],
                "lightgbm": [round(float(p_lgb[idx0]), 6), round(float(p_lgb[idx1]), 6)],
                "xgboost": [round(float(p_xgb[idx0]), 6), round(float(p_xgb[idx1]), 6)],
            }
        return p_calibrated_map, features_map, base_probs_map

    def process_task(self, task: InferenceTask) -> tuple[PredictResponse, list[DialogueEvent]]:
        """Прямой пайплайн обработки задачи со склейкой частичного кэша."""
        t0 = time.perf_counter()
        if not task.dialogues:
            return PredictResponse(results=[], total=0, batch_latency_ms=0.0), []

        dialogue_msgs = [d.get("messages", []) for d in task.dialogues]
        hashes = [DialogueHasher.compute(msgs) for msgs in dialogue_msgs]
        cached_entries = self.cache.mget(hashes)
        miss_indices = [idx for idx, entry in enumerate(cached_entries) if entry is None]
        p_calibrated_map, features_map, base_probs_map = self._infer_cache_misses(dialogue_msgs, miss_indices)

        batch_latency_ms = (time.perf_counter() - t0) * 1000.0
        item_latency_ms = batch_latency_ms / max(len(task.dialogues), 1)
        results: list[DialogueResultItem] = []
        events_to_publish: list[DialogueEvent] = []
        cache_items_to_set: list[tuple[str, Any]] = []
        should_audit = config.clickhouse.enable_audit and task.audit_mode != AuditMode.OFF

        for i, d in enumerate(task.dialogues):
            d_id = str(d.get("dialog_id", f"dlg_{i}"))
            raw_msgs = dialogue_msgs[i]
            cached_val = cached_entries[i]
            if cached_val is not None:
                verdicts_dicts = cached_val["verdicts"]
                results.append(
                    DialogueResultItem.from_inference(
                        dialog_id=d_id,
                        dialogue_length=cached_val.get("dialogue_length", len(raw_msgs)),
                        verdicts=verdicts_dicts,
                        cached=True,
                        latency_ms=round(item_latency_ms, 3),
                    )
                )
                if should_audit:
                    events_to_publish.append(
                        self.audit_writer.create_event(
                            dialog_id=d_id,
                            messages=raw_msgs,
                            verdicts=verdicts_dicts,
                            features=[[], []],
                            base_model_probs=cached_val.get("base_probs") or {},
                            cached=True,
                            latency_ms=round(item_latency_ms, 3),
                            audit_mode=task.audit_mode,
                        )
                    )
                continue

            p0, p1 = p_calibrated_map[i]
            verdicts = [
                ParticipantVerdict(participant_index=0, bot_probability=p0, is_bot=bool(p0 >= 0.5)),
                ParticipantVerdict(participant_index=1, bot_probability=p1, is_bot=bool(p1 >= 0.5)),
            ]
            verdicts_dicts = [v.model_dump() for v in verdicts]
            results.append(
                DialogueResultItem.from_inference(
                    dialog_id=d_id,
                    dialogue_length=len(raw_msgs),
                    verdicts=verdicts_dicts,
                    cached=False,
                    latency_ms=round(item_latency_ms, 2),
                )
            )
            cache_items_to_set.append(
                (
                    hashes[i],
                    {"dialogue_length": len(raw_msgs), "verdicts": verdicts_dicts, "base_probs": base_probs_map[i]},
                )
            )
            if should_audit:
                events_to_publish.append(
                    self.audit_writer.create_event(
                        dialog_id=d_id,
                        messages=raw_msgs,
                        verdicts=verdicts_dicts,
                        features=[features_map[i][0], features_map[i][1]],
                        base_model_probs=base_probs_map[i],
                        cached=False,
                        latency_ms=item_latency_ms,
                        audit_mode=task.audit_mode,
                    )
                )

        if cache_items_to_set:
            self.cache.mset(cache_items_to_set)
        return PredictResponse(
            results=results, total=len(results), batch_latency_ms=round(batch_latency_ms, 2)
        ), events_to_publish


__all__ = [
    "InferencePipelineService",
    "WebhookNotifier",
]
