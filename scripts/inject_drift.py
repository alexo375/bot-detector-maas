"""Утилита управляемой инъекции дрейфа на реальных коротких/сложных диалогах train и сброса baseline."""

import argparse
import json
import logging
import socket
import time
import urllib.request
import uuid
from typing import Any

import clickhouse_connect
import httpx
import redis

from project.core.config import config
from project.core.logging import LoggingConfigurator
from project.core.urls import service_urls
from project.core.utils import DialogueHasher
from project.features.store.store import FeatureStore

logger = logging.getLogger("drift_injector")

_DRIFT_PREFIX = "drift_inject_"


def _resolve_host(host: str) -> str:
    """Откат на localhost при запуске вне Docker-сети."""
    try:
        socket.gethostbyname(host)
        return host
    except socket.gaierror:
        return "localhost"


def _get_ch_client() -> Any:
    return clickhouse_connect.get_client(
        host=_resolve_host(config.clickhouse.host),
        port=config.clickhouse.port,
        database=config.clickhouse.database,
        connect_timeout=3,
    )


def _get_redis_client() -> redis.Redis:
    for url in (config.redis.url, "redis://localhost:6379/0"):
        try:
            r_conn = redis.Redis.from_url(url, socket_timeout=2.0)
            r_conn.ping()
            return r_conn
        except Exception as exc:
            logger.debug("Redis connection failed (%s): %s", url, exc)
    raise ConnectionError("Could not connect to Redis via config or localhost")


def _sync_api_gateway_metrics(api_url: str) -> None:
    for url in (f"{api_url.rstrip('/')}/metrics", "http://localhost:8080/metrics"):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    logger.info("Synced API Gateway metrics via %s", url)
                    return
        except Exception as exc:
            logger.debug("Metrics sync failed (%s): %s", url, exc)


def _select_hard_short_train_dialogues(count: int) -> list[dict[str, Any]]:
    """Строго выбирает самые короткие/сложные диалоги из ClickHouse `raw_dialogues` (без локального фолбэка)."""
    ts = int(time.time())
    selected_msgs: list[list[dict[str, Any]]] = []

    client = _get_ch_client()
    query = (
        "SELECT dialog_id, any(messages_json) AS msgs "
        "FROM raw_dialogues "
        "WHERE dataset_type = 'train' AND is_bot IS NOT NULL "
        "GROUP BY dialog_id "
        "ORDER BY min(num_messages) ASC, length(any(messages_json)) ASC "
        "LIMIT {lim:UInt32}"
    )
    rows = client.query(query, parameters={"lim": int(count * 2)}).result_rows
    for _d_id, raw_msgs in rows:
        msgs = json.loads(raw_msgs) if isinstance(raw_msgs, str) else raw_msgs
        if isinstance(msgs, list) and msgs:
            selected_msgs.append(
                [{"participant_index": int(m["participant_index"]), "text": str(m.get("text", ""))} for m in msgs]
            )
        if len(selected_msgs) >= count:
            break

    if len(selected_msgs) < count:
        raise RuntimeError(f"ClickHouse raw_dialogues returned only {len(selected_msgs)} dialogues (< {count})")

    return [
        {"dialog_id": f"{_DRIFT_PREFIX}{ts}_{idx}_{uuid.uuid4().hex[:6]}", "messages": msgs}
        for idx, msgs in enumerate(selected_msgs[:count])
    ]


def _send_predict_batch(
    client: httpx.Client,
    api_url: str,
    batch: list[dict[str, Any]],
) -> None:
    resp = client.post(f"{api_url.rstrip('/')}/predict", json={"dialogues": batch, "audit_mode": "production"})
    resp.raise_for_status()
    if resp.status_code == 202:
        task_id = resp.json()["task_id"]
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            poll = client.get(f"{api_url.rstrip('/')}/predict/{task_id}")
            if poll.status_code == 200:
                return
            if poll.status_code != 202:
                poll.raise_for_status()
            time.sleep(0.15)
        raise TimeoutError(f"Timed out waiting for async /predict/{task_id}")


def _invalidate_dialogue_cache(r_conn: redis.Redis, hashes: list[str] | None = None) -> int:
    """Удаляет закэшированные скоры диалогов (и при полном сбросе async_task:*) из Redis."""
    if hashes is not None:
        keys = [f"dialog_score:{h}" for h in hashes]
        return int(r_conn.delete(*keys)) if keys else 0

    keys_to_delete: list[Any] = []
    for pattern in ("dialog_score:*", "async_task:*"):
        keys_to_delete.extend(r_conn.scan_iter(match=pattern, count=500))
    return int(r_conn.delete(*keys_to_delete)) if keys_to_delete else 0


def inject_drift(count: int = 100, batch_size: int = 20, api_url: str = "http://localhost:8080") -> int:
    """Отправляет выбранные короткие/сложные train-диалоги через живой инференс (API -> Triton -> ClickHouse)."""
    dialogues = _select_hard_short_train_dialogues(count=count)
    if not dialogues:
        raise RuntimeError("No train dialogues available in ClickHouse raw_dialogues or data/train.json")

    try:
        r_conn = _get_redis_client()
        try:
            hashes = [DialogueHasher.compute(d["messages"]) for d in dialogues]
            cleared = _invalidate_dialogue_cache(r_conn, hashes=hashes)
            if cleared:
                logger.info("Invalidated %d cached dialog_score entries in Redis before drift inject", cleared)
        finally:
            r_conn.close()
    except Exception as exc:
        logger.debug("Pre-inject Redis cache invalidation skipped: %s", exc)

    with httpx.Client(timeout=30.0) as http_client:
        for i in range(0, len(dialogues), batch_size):
            _send_predict_batch(http_client, api_url, dialogues[i : i + batch_size])

    ch_client = _get_ch_client()
    deadline = time.monotonic() + 15.0
    flushed = 0
    while time.monotonic() < deadline:
        res = ch_client.query(
            "SELECT count() FROM chat_logs WHERE startsWith(dialog_id, {pref:String})",
            parameters={"pref": _DRIFT_PREFIX},
        ).result_rows
        flushed = int(res[0][0]) if res else 0
        if flushed >= len(dialogues):
            break
        time.sleep(0.5)

    logger.info(
        "Injected %d hard/short train dialogues via /predict (%d flushed to chat_logs)", len(dialogues), flushed
    )
    logger.info(
        "Prefect 'drift-monitor' will evaluate metrics on the next 2-min tick and raise alerts for manual retraining."
    )
    logger.info(" - Grafana Data Drift:  %s", service_urls.grafana_drift_dashboard)
    logger.info(" - Prefect Deployments: %s", service_urls.prefect_deployments)
    logger.info(" - MLflow Models:       %s", service_urls.mlflow_models)
    return flushed


def reset_baseline_state(api_url: str = "http://localhost:8080") -> None:
    """Удаляет синтетические/инжектированные строки из chat_logs и сбрасывает метрики Redis в HEALTHY."""
    try:
        ch_client = _get_ch_client()
        params = {"pref": _DRIFT_PREFIX}
        cnt = int(
            ch_client.query(
                "SELECT count() FROM chat_logs WHERE startsWith(dialog_id, {pref:String})",
                parameters=params,
            ).result_rows[0][0]
        )
        if cnt > 0:
            ch_client.command(
                "ALTER TABLE chat_logs DELETE WHERE startsWith(dialog_id, {pref:String}) SETTINGS mutations_sync = 1",
                parameters=params,
            )
        logger.info("Cleaned up %d injected records from chat_logs", cnt)
    except Exception as exc:
        logger.warning("Failed to clean chat_logs records: %s", exc)

    feature_names = FeatureStore().feature_names
    r_conn = _get_redis_client()
    try:
        cleared_cache = _invalidate_dialogue_cache(r_conn, hashes=None)
        if cleared_cache:
            logger.info("Cleared %d dialogue/task cache keys in Redis", cleared_cache)
        r_conn.delete(
            "bot_detector:drift_metrics",
            "bot_detector:prediction_drift_metrics",
            "bot_detector:concept_drift_metrics",
            "bot_detector:feature_importance_metrics",
            "bot_detector:adaptive_decision",
            "bot_detector:retrain_in_progress",
            "bot_detector:retrain_cooldown_until",
        )
        r_conn.set(
            "bot_detector:drift_metrics",
            json.dumps(
                {
                    "drift_share": 0.0,
                    "dataset_drift": False,
                    "number_of_drifted_columns": 0,
                    "number_of_columns": len(feature_names),
                    "drifted_features": [],
                    "column_p_values": dict.fromkeys(feature_names, 1.0),
                    "computed_at_ts": time.time(),
                }
            ),
        )
        r_conn.set(
            "bot_detector:prediction_drift_metrics",
            json.dumps(
                {
                    "psi": 0.0,
                    "mean_entropy": 0.15,
                    "is_drifted": False,
                    "verdict_shares": {"both_humans": 0.65, "both_bots": 0.01, "p0_bot": 0.17, "p1_bot": 0.17},
                }
            ),
        )
        r_conn.set(
            "bot_detector:concept_drift_metrics",
            json.dumps({"disagreement_rate": 0.0, "feature_rank_correlation": 1.0, "concept_drift_detected": False}),
        )
        r_conn.set(
            "bot_detector:adaptive_decision",
            json.dumps({"dialogues_count": 0, "model_age_days": 0.0, "retrain_status": 0, "reason": "system_healthy"}),
        )
    finally:
        r_conn.close()

    _sync_api_gateway_metrics(api_url)
    logger.info("Reset Redis drift metrics and retrain locks to baseline HEALTHY state")


def main() -> None:
    """Точка входа CLI для инъекции дрифта и сброса baseline."""
    parser = argparse.ArgumentParser(description="Inject hard/short train dialogues into /predict or reset baseline")
    parser.add_argument("--count", type=int, default=100, help="Number of short/hard train dialogues to inject")
    parser.add_argument("--batch-size", type=int, default=20, help="Predict batch size per HTTP request")
    parser.add_argument("--api-url", default="http://localhost:8080", help="API Gateway base URL")
    parser.add_argument("--reset", action="store_true", help="Clean injected rows and reset Redis baseline state")
    args = parser.parse_args()

    LoggingConfigurator.configure()
    if args.reset:
        reset_baseline_state(api_url=args.api_url)
    else:
        inject_drift(count=args.count, batch_size=args.batch_size, api_url=args.api_url)


if __name__ == "__main__":
    main()
