import logging
from pathlib import Path

from clickhouse_connect.driver.client import Client

from project.infrastructure.clickhouse.connection import ClickHouseConnectionFactory

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class ClickHouseMigrator:
    """Менеджер применения SQL-миграций схемы ClickHouse DWH."""

    @staticmethod
    def init_migrations_table(client: Client) -> None:
        """Создает системную таблицу учета примененных миграций."""
        client.command(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version String,
                applied_at DateTime DEFAULT now()
            )
            ENGINE = ReplacingMergeTree()
            ORDER BY version;
            """
        )

    @classmethod
    def apply(cls, client: Client | None = None) -> list[str]:
        """Применяет все новые SQL-миграции из директории `migrations/`."""
        ch_client = client or ClickHouseConnectionFactory.create()
        cls.init_migrations_table(ch_client)

        applied_versions = {row[0] for row in ch_client.query("SELECT version FROM schema_migrations").result_rows}
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        newly_applied: list[str] = []

        for mig_file in migration_files:
            version = mig_file.name
            if version in applied_versions:
                continue

            logger.info("Применяем миграцию ClickHouse: %s", version)
            sql_content = mig_file.read_text(encoding="utf-8")
            statements = [stmt.strip() for stmt in sql_content.split(";") if stmt.strip()]
            for stmt in statements:
                ch_client.command(stmt)

            ch_client.insert("schema_migrations", [[version]], column_names=["version"])
            newly_applied.append(version)
            logger.info("Миграция %s успешно применена", version)

        return newly_applied


__all__ = ["MIGRATIONS_DIR", "ClickHouseMigrator"]
