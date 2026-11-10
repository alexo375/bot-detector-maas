"""Модульные тесты для репозиториев ClickHouse и мигратора."""

from unittest.mock import MagicMock, patch

import numpy as np

from project.infrastructure.clickhouse.connection import ClickHouseConnectionFactory
from project.infrastructure.clickhouse.migrator import ClickHouseMigrator
from project.infrastructure.clickhouse.repositories import (
    ClickHousePredictionRepository,
    ClickHouseRawRepository,
)


def test_resolve_clickhouse_host_local() -> None:
    """Проверка возврата localhost и 127.0.0.1 без DNS lookup."""
    assert ClickHouseConnectionFactory.resolve_host("localhost") == "localhost"
    assert ClickHouseConnectionFactory.resolve_host("127.0.0.1") == "127.0.0.1"


def test_resolve_clickhouse_host_fallback() -> None:
    """Проверка фолбэка на 127.0.0.1 при сбое резолвинга хоста."""
    with patch("socket.gethostbyname", side_effect=OSError("Host not found")):
        assert ClickHouseConnectionFactory.resolve_host("unresolved.internal.host") == "127.0.0.1"


def test_init_migrations_table() -> None:
    """Проверка выполнения DDL для создания таблицы миграций."""
    mock_client = MagicMock()
    ClickHouseMigrator.init_migrations_table(mock_client)
    mock_client.command.assert_called_once()
    assert "CREATE TABLE IF NOT EXISTS schema_migrations" in mock_client.command.call_args[0][0]


def test_apply_migrations_skips_already_applied() -> None:
    """Проверка пропуска ранее примененных миграций."""
    mock_client = MagicMock()
    # Имитируем, что все миграции уже записаны в базе
    mock_client.query.return_value.result_rows = [
        ("001_create_raw_tables.sql",),
        ("002_create_feature_store_tables.sql",),
        ("003_create_model_predictions.sql",),
        ("004_create_chat_logs.sql",),
        ("005_add_base_model_probs_to_chat_logs.sql",),
    ]

    applied = ClickHouseMigrator.apply(mock_client)
    # Ни одной новой миграции применяться не должно
    assert len(applied) == 0


def test_clickhouse_raw_repository_queries() -> None:
    """Проверка методов count_dialogues и has_dialogues в ClickHouseRawRepository."""
    mock_client = MagicMock()
    mock_client.command.return_value = "42"

    repo = ClickHouseRawRepository(client=mock_client)
    assert repo.count_dialogues("train") == 42
    assert repo.has_dialogues("train") is True
    mock_client.command.assert_called_with("SELECT count() FROM raw_dialogues WHERE dataset_type = 'train'")


def test_clickhouse_prediction_repository_has_and_save() -> None:
    """Проверка методов проверки и сохранения предсказаний в ClickHousePredictionRepository."""
    mock_client = MagicMock()
    mock_client.command.return_value = "10"

    repo = ClickHousePredictionRepository(client=mock_client)
    assert repo.has_predictions("run_123", dataset_split="train_oof") is True

    # Сохранение предсказаний
    d_ids = np.array(["d1", "d2"])
    p_idx = np.array([0, 1])
    probs = np.array([0.2, 0.8])

    saved_count = repo.save_predictions(
        run_id="run_123",
        run_name="catboost_run",
        dataset_split="train_oof",
        dialog_ids=d_ids,
        participant_indices=p_idx,
        prob_bot=probs,
    )
    assert saved_count == 2
    mock_client.insert.assert_called_once()


def test_clickhouse_feature_repository_snapshot_lifecycle() -> None:
    """Проверка создания и проверки наличия снапшота признаков."""
    from project.infrastructure.clickhouse.repositories import ClickHouseFeatureRepository

    mock_client = MagicMock()
    mock_client.command.return_value = "1"
    # Первый запрос: снапшот не найден
    mock_client.query.side_effect = [
        MagicMock(result_rows=[]),  # SELECT snapshot_id -> empty
        MagicMock(result_rows=[(5,)]),  # SELECT max(snapshot_id) -> 5
    ]

    repo = ClickHouseFeatureRepository(client=mock_client)
    snapshot_id = repo._get_or_create_snapshot_id(
        snapshot_tag="v1_tag",
        feature_names=["f1", "f2"],
        description="test",
    )
    assert snapshot_id == 6
    assert repo.has_snapshot("v1_tag", dataset_type="train") is True


def test_feature_dwh_delegation() -> None:
    """Проверка делегирования вызовов в FeatureDWH."""
    from project.infrastructure.clickhouse.dwh import FeatureDWH

    mock_client = MagicMock()
    dwh = FeatureDWH(client=mock_client)

    assert dwh.raw.client is mock_client
    assert dwh.features.client is mock_client
    assert dwh.predictions.client is mock_client


def test_seeder_fails_fast_when_mlflow_runs_missing() -> None:
    """Без валидных MLflow ранов ClickHouseDataSeeder обязан падать (без стартового ONNX-фолбэка)."""
    from unittest.mock import patch

    import pytest

    from project.infrastructure.clickhouse.seeding.data_seeder import ClickHouseDataSeeder

    mock_dwh = MagicMock()
    mock_mlflow_client = MagicMock()
    mock_mlflow_client.search_experiments.return_value = []

    seeder = ClickHouseDataSeeder(dwh=mock_dwh)
    with (
        patch("mlflow.tracking.MlflowClient", return_value=mock_mlflow_client),
        pytest.raises(RuntimeError, match="seed failed without starter ONNX fallback"),
    ):
        seeder.seed_model_predictions()


def test_prediction_repository_save_handles_none_and_duplicate_run_names() -> None:
    """save_predictions безопасно обрабатывает None/NaN в is_bot, а load_meta_dataset различает дубликаты run_name."""
    mock_client = MagicMock()
    repo = ClickHousePredictionRepository(client=mock_client)

    saved = repo.save_predictions(
        run_id="r_test",
        run_name="test_run",
        dataset_split="test",
        dialog_ids=np.array(["d0", "d1"]),
        participant_indices=np.array([0, 1]),
        prob_bot=np.array([0.1, 0.9]),
        is_bot=np.array([None, float("nan")], dtype=object),
    )
    assert saved == 2
    inserted_rows = mock_client.insert.call_args.args[1]
    assert inserted_rows[0][-1] is None
    assert inserted_rows[1][-1] is None

    mock_client.query.side_effect = [
        MagicMock(result_rows=[("same_name",)]),
        MagicMock(result_rows=[("same_name",)]),
        MagicMock(result_rows=[("d0", 0, 1, 0.2, 0.8)]),
    ]
    df = repo.load_meta_dataset(["run_1", "run_2"])
    assert df.columns == ["dialog_id", "participant_index", "is_bot", "prob_same_name", "prob_same_name_1"]
