import argparse
import asyncio
import contextlib
import json
import re
import signal
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aio_pika

from project.core.config import config
from project.core.logging import LoggingConfigurator, ProjectLogger
from project.core.schemas import AuditMode
from project.infrastructure.clickhouse.audit.schema import ChatLogRecord
from project.infrastructure.clickhouse.connection import ClickHouseConnectionFactory
from project.infrastructure.clickhouse.migrator import ClickHouseMigrator
from project.infrastructure.rabbitmq import QUEUE_AUDIT_EVENTS

logger = ProjectLogger.get_logger(__name__)

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


class ClickHouseLogConsumer:
    """Асинхронный воркер для вычитки событий из RabbitMQ и батчевой записи в ClickHouse."""

    def __init__(
        self,
        rabbitmq_url: str | None = None,
        ch_host: str | None = None,
        ch_port: int | None = None,
        ch_user: str | None = None,
        ch_password: str | None = None,
        ch_database: str | None = None,
        batch_size: int | None = None,
        flush_interval_sec: float | None = None,
    ) -> None:
        """Инициализация параметров подключения и буфера сообщений."""
        self.rabbitmq_url = rabbitmq_url or config.rabbitmq.url
        self.ch_host = ch_host or config.clickhouse.host
        self.ch_port = ch_port if ch_port is not None else config.clickhouse.port
        self.ch_user = ch_user or config.clickhouse.user
        self.ch_password = ch_password if ch_password is not None else config.clickhouse.password
        self.ch_database = ch_database or config.clickhouse.database
        self.batch_size = batch_size if batch_size is not None else config.clickhouse.batch_size
        self.flush_interval_sec = (
            flush_interval_sec if flush_interval_sec is not None else config.clickhouse.flush_interval_sec
        )

        self.ch_client: Any = None
        self.connection: aio_pika.abc.AbstractRobustConnection | None = None
        self.channel: aio_pika.abc.AbstractChannel | None = None
        self.queue: aio_pika.abc.AbstractQueue | None = None

        self._running = True
        self._buffer_lock = asyncio.Lock()
        self._message_buffer: list[tuple[aio_pika.abc.AbstractIncomingMessage, list[Any], AuditMode]] = []
        self._last_flush_time = time.monotonic()
        self._initialized_test_dbs: set[str] = set()

    def connect_clickhouse(self) -> bool:
        """Установка соединения с сервером ClickHouse."""
        try:
            self.ch_client = ClickHouseConnectionFactory.create(
                host=self.ch_host,
                port=self.ch_port,
                database=self.ch_database,
                username=self.ch_user,
                password=self.ch_password,
                connect_timeout=3,
                send_receive_timeout=10,
            )
            self.ch_client.command("SELECT 1")
            try:
                ClickHouseMigrator.apply(self.ch_client)
            except Exception as mig_err:  # noqa: BLE001
                logger.warning("Не удалось применить миграции ClickHouse: %s", mig_err)
            return True
        except Exception as e:  # noqa: BLE001  # Resilient fallback
            self.ch_client = None
            logger.warning("Ошибка подключения к ClickHouse: %s", e)
            return False

    async def connect_rabbitmq(self) -> bool:
        """Установка асинхронного соединения с брокером сообщений RabbitMQ."""
        try:
            self.connection = await aio_pika.connect_robust(self.rabbitmq_url, timeout=5.0)
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=self.batch_size * 2)
            self.queue = await self.channel.declare_queue(QUEUE_AUDIT_EVENTS, durable=True)
            return True
        except Exception as e:  # noqa: BLE001  # Resilient fallback
            logger.error("Ошибка подключения к RabbitMQ: %s", e)
            return False

    def parse_event_to_row(self, payload: dict[str, Any]) -> list[Any]:
        """Преобразование JSON события в форматированную строку таблицы ClickHouse."""
        record = ChatLogRecord.from_payload(payload)
        return record.to_row()

    def _ensure_test_chat_logs_table(self, test_db: str) -> None:
        """Создает изолированную тестовую БД и таблицу chat_logs при поступлении test-событий."""
        if self.ch_client is None or test_db in self._initialized_test_dbs:
            return
        if not re.fullmatch(r"[A-Za-z0-9_]+", test_db):
            raise ValueError(f"Недопустимое имя тестовой БД ClickHouse: {test_db!r}")
        self.ch_client.command(f"CREATE DATABASE IF NOT EXISTS {test_db}")
        for mig_name in ("004_create_chat_logs.sql", "005_add_base_model_probs_to_chat_logs.sql"):
            mig_path = _MIGRATIONS_DIR / mig_name
            if not mig_path.is_file():
                continue
            sql = mig_path.read_text(encoding="utf-8").replace("chat_logs", f"{test_db}.chat_logs")
            for stmt in (s.strip() for s in sql.split(";") if s.strip()):
                self.ch_client.command(stmt)
        self._initialized_test_dbs.add(test_db)

    async def _flush_target_batch(
        self,
        items: list[tuple[aio_pika.abc.AbstractIncomingMessage, list[Any]]],
        database: str | None = None,
    ) -> int:
        """Сброс подбатча в целевую БД ClickHouse и ack/nack соответствующих сообщений."""
        if not items or self.ch_client is None:
            return 0
        rows = [row for _, row in items]
        raw_msgs = [msg for msg, _ in items]
        insert_kwargs: dict[str, Any] = {
            "table": "chat_logs",
            "data": rows,
            "column_names": ChatLogRecord.column_names(),
        }
        try:
            if database is not None:
                await asyncio.to_thread(self._ensure_test_chat_logs_table, database)
                insert_kwargs["database"] = database
            await asyncio.to_thread(self.ch_client.insert, **insert_kwargs)
            for msg in raw_msgs:
                with contextlib.suppress(Exception):
                    await msg.ack()
            return len(rows)
        except Exception as e:  # noqa: BLE001  # Resilient fallback
            logger.error("Ошибка вставки в ClickHouse: %s", e)
            for msg in raw_msgs:
                with contextlib.suppress(Exception):
                    await msg.nack(requeue=True)
            return 0

    async def flush_buffer(self) -> int:
        """Сброс накопленного микробатча в ClickHouse с маршрутизацией production/test и ack."""
        async with self._buffer_lock:
            if not self._message_buffer:
                self._last_flush_time = time.monotonic()
                return 0
            batch_to_insert = list(self._message_buffer)
            self._message_buffer.clear()
            self._last_flush_time = time.monotonic()

        off_msgs = [m for m, _, mode in batch_to_insert if mode == AuditMode.OFF]
        for msg in off_msgs:
            with contextlib.suppress(Exception):
                await msg.ack()

        persist_items = [(m, r, mode) for m, r, mode in batch_to_insert if mode != AuditMode.OFF]
        if not persist_items:
            return 0

        if self.ch_client is None:
            await asyncio.to_thread(self.connect_clickhouse)

        if self.ch_client is None:
            for msg, _, _ in persist_items:
                with contextlib.suppress(Exception):
                    await msg.nack(requeue=True)
            return 0

        prod_items = [(m, r) for m, r, mode in persist_items if mode == AuditMode.PRODUCTION]
        test_items = [(m, r) for m, r, mode in persist_items if mode == AuditMode.TEST]
        inserted = await self._flush_target_batch(prod_items)
        if test_items:
            inserted += await self._flush_target_batch(test_items, database=f"{self.ch_database}_test")
        return inserted

    async def _periodic_flush(self) -> None:
        """Периодический сброс буфера по таймеру."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval_sec)
                if self._message_buffer and (time.monotonic() - self._last_flush_time >= self.flush_interval_sec):
                    await self.flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001  # Resilient fallback
                logger.warning("Error in periodic flush: %s", e)

    async def _iter_messages(self, queue_iter: Any, run_once: bool) -> AsyncIterator[Any]:
        """Итерация сообщений очереди (бесконечно или дренирование батча для run_once)."""
        if not run_once:
            async for msg in queue_iter:
                yield msg
            return

        timeout = max(0.05, min(self.flush_interval_sec, 1.0))
        for _ in range(self.batch_size):
            try:
                msg = await asyncio.wait_for(queue_iter.__anext__(), timeout=timeout)
            except (StopAsyncIteration, TimeoutError):
                break
            yield msg

    async def _handle_message(self, msg: Any) -> int:
        try:
            payload = json.loads(msg.body.decode("utf-8"))
            audit_mode = AuditMode(payload.get("audit_mode", AuditMode.PRODUCTION))
            row = self.parse_event_to_row(payload)
            async with self._buffer_lock:
                self._message_buffer.append((msg, row, audit_mode))
                buf_len = len(self._message_buffer)
        except Exception as e:  # noqa: BLE001  # Resilient fallback
            logger.error("Error decoding event: %s", e)
            with contextlib.suppress(Exception):
                await msg.reject(requeue=False)
            return 0

        if buf_len >= self.batch_size:
            await self.flush_buffer()
        return 1

    async def run(self, run_once: bool = False, max_messages: int | None = None) -> int:
        """Запуск цикла прослушивания очереди и сброса батчей."""
        self.connect_clickhouse()
        while self._running:
            rmq_ok = await self.connect_rabbitmq()
            if rmq_ok and self.queue is not None:
                break
            if run_once:
                return 0
            logger.warning("RabbitMQ connection failed, retrying in 2 seconds...")
            await asyncio.sleep(2.0)

        if self.queue is None:
            return 0

        flush_task = asyncio.create_task(self._periodic_flush())
        total_processed = 0
        try:
            async with self.queue.iterator() as queue_iter:
                async for msg in self._iter_messages(queue_iter, run_once=run_once):
                    if not self._running:
                        break
                    total_processed += await self._handle_message(msg)
                    if max_messages and total_processed >= max_messages:
                        break
        except asyncio.CancelledError:
            logger.info("ClickHouseLogConsumer cancelled, flushing remaining buffer...")
        finally:
            flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await flush_task
            if self._message_buffer:
                await self.flush_buffer()
            await self.close()
        return total_processed

    async def close(self) -> None:
        """Корректное закрытие сетевых соединений RabbitMQ и ClickHouse."""
        self._running = False
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
        if self.ch_client is not None:
            self.ch_client.close()


def main(argv: list[str] | None = None) -> None:
    """Точка входа воркера ClickHouse консумера."""
    parser = argparse.ArgumentParser(description="ClickHouse Log Stream Consumer Worker")
    parser.add_argument("--once", action="store_true", help="Однократный прогон")
    parser.add_argument("--batch-size", type=int, default=None, help="Размер батча для вставки")
    parser.add_argument("--flush-interval", type=float, default=None, help="Интервал сброса в секундах")
    args = parser.parse_args(argv)

    LoggingConfigurator.configure()
    consumer = ClickHouseLogConsumer(batch_size=args.batch_size, flush_interval_sec=args.flush_interval)

    async def _serve() -> None:
        loop = asyncio.get_running_loop()
        run_task = asyncio.create_task(consumer.run(run_once=args.once))

        def _request_stop() -> None:
            consumer._running = False
            run_task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _request_stop)

        await run_task

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
