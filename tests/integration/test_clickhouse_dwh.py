"""Интеграционные тесты слоя ClickHouse DWH: Raw, Feature Store и Predictions."""

from collections.abc import Generator

import numpy as np
import polars as pl
import pytest

from project.infrastructure.clickhouse import (
    ClickHouseConnectionFactory,
    FeatureDWH,
)

TEST_DB = "bot_detection_test"

pytestmark = [pytest.mark.live, pytest.mark.xdist_group("clickhouse_test_db")]


@pytest.fixture(scope="module")
def dwh() -> Generator[FeatureDWH]:
    """Инициализация изолированной тестовой БД ClickHouse и применение миграций."""
    admin = ClickHouseConnectionFactory.create()
    admin.command(f"CREATE DATABASE IF NOT EXISTS {TEST_DB}")
    client = ClickHouseConnectionFactory.create(database=TEST_DB)
    dwh_client = FeatureDWH(client)
    dwh_client.init_schema()
    try:
        yield dwh_client
    finally:
        client.close()
        admin.command(f"DROP DATABASE IF EXISTS {TEST_DB}")
        admin.close()


def test_dwh_feature_snapshot_lifecycle(dwh):
    """Тест полного жизненного цикла среза признаков: сохранение, загрузка, проекция и удаление."""
    tag = "_test_snapshot_tag"
    n_samples = 20

    # Создаем тестовый DataFrame
    test_df = pl.DataFrame(
        {
            "dialog_id": [f"dlg_{i}" for i in range(100, 100 + n_samples)],
            "participant_index": [0, 1] * (n_samples // 2),
            "feat_a": np.random.randn(n_samples).astype(np.float32),
            "feat_b": np.random.randn(n_samples).astype(np.float32),
            "feat_c": np.random.randn(n_samples).astype(np.float32),
        }
    )

    # Сохраняем снапшот
    snapshot_id = dwh.features.save_snapshot(
        snapshot_tag=tag,
        features_df=test_df,
        dataset_type="train",
        description="Тестовый снапшот",
    )
    assert snapshot_id > 0
    assert dwh.features.has_snapshot(tag, "train") is True

    # Загружаем полный снапшот
    loaded_df, feat_names = dwh.features.load_snapshot(tag, dataset_type="train")
    assert len(loaded_df) == n_samples
    assert set(feat_names) == {"feat_a", "feat_b", "feat_c"}
    assert "dialog_id" in loaded_df.columns
    assert "participant_index" in loaded_df.columns

    # Загружаем проекцию Top-K признаков (только feat_a и feat_c)
    proj_df, proj_feats = dwh.features.load_snapshot(
        tag,
        selected_features=["feat_a", "feat_c"],
        dataset_type="train",
    )
    assert proj_df.columns == ["dialog_id", "participant_index", "feat_a", "feat_c"]
    assert proj_feats == ["feat_a", "feat_c"]
    assert len(proj_df) == n_samples

    # Удаляем срез
    deleted = dwh.features.drop_snapshot(tag)
    assert deleted is True
    assert dwh.features.has_snapshot(tag, "train") is False


def test_dwh_prediction_repository_and_meta_join(dwh):
    """Тест сохранения OOF-предсказаний и сборки мета-датасета через SQL JOIN."""
    run_1 = "test_run_cb_123"
    run_2 = "test_run_lgb_456"

    d_ids = np.array(["dlg_10", "dlg_10", "dlg_20", "dlg_20"])
    p_idxs = np.array([0, 1, 0, 1], dtype=np.uint8)
    y_true = np.array([1, 0, 0, 1], dtype=np.uint8)

    # Предсказания модели 1 (CatBoost)
    cb_raw = np.array([0.9, 0.1, 0.2, 0.8], dtype=np.float32)
    cb_calib = np.array([0.88, 0.12, 0.22, 0.78], dtype=np.float32)
    dwh.predictions.save_predictions(
        run_id=run_1,
        run_name="catboost_top_pruned",
        dataset_split="train_oof",
        dialog_ids=d_ids,
        participant_indices=p_idxs,
        prob_bot=cb_raw,
        prob_bot_calibrated=cb_calib,
        is_bot=y_true,
    )

    # Предсказания модели 2 (LightGBM)
    lgb_raw = np.array([0.85, 0.15, 0.25, 0.75], dtype=np.float32)
    lgb_calib = np.array([0.84, 0.16, 0.26, 0.74], dtype=np.float32)
    dwh.predictions.save_predictions(
        run_id=run_2,
        run_name="lightgbm_top_pruned",
        dataset_split="train_oof",
        dialog_ids=d_ids,
        participant_indices=p_idxs,
        prob_bot=lgb_raw,
        prob_bot_calibrated=lgb_calib,
        is_bot=y_true,
    )

    # Проверяем наличие
    assert dwh.predictions.has_predictions(run_1, "train_oof") is True
    assert dwh.predictions.has_predictions(run_2, "train_oof") is True

    # Загружаем мета-датасет со состыкованными предсказаниями
    meta_df = dwh.predictions.load_meta_dataset(
        run_ids=[run_1, run_2],
        dataset_split="train_oof",
        use_calibrated=True,
    )
    assert len(meta_df) == 4
    assert "prob_catboost_top_pruned" in meta_df.columns
    assert "prob_lightgbm_top_pruned" in meta_df.columns
    assert "is_bot" in meta_df.columns

    # Проверяем значения первой строки
    row0 = meta_df.row(0, named=True)
    assert row0["dialog_id"] == "dlg_10"
    assert row0["participant_index"] == 0
    assert row0["is_bot"] == 1
    assert abs(row0["prob_catboost_top_pruned"] - 0.88) < 1e-4
    assert abs(row0["prob_lightgbm_top_pruned"] - 0.84) < 1e-4

    # Чистим тестовые раны
    dwh.predictions.delete_predictions(run_1)
    dwh.predictions.delete_predictions(run_2)
    assert dwh.predictions.has_predictions(run_1, "train_oof") is False
    assert dwh.predictions.has_predictions(run_2, "train_oof") is False


def test_dwh_raw_dataset_ingestion(dwh):
    """Тест загрузки и проверки сырых датасетов (train/test)."""
    from project import paths

    # Проверяем сидирование сырых данных
    counts = dwh.seed_raw_datasets(
        train_json_path=paths.raw.train_json,
        test_json_path=paths.raw.test_json,
        ytrain_csv_path=paths.raw.ytrain_csv,
        ytest_csv_path=paths.raw.ytest_csv,
    )
    assert counts["train"] == 3032
    assert counts["test"] == 758

    # Проверяем чтение разметки
    df_labels = dwh.raw.load_labels("train")
    assert len(df_labels) == 3032
    assert "dialog_id" in df_labels.columns
    assert "is_bot" in df_labels.columns

    # Проверяем чтение словаря диалогов
    dlgs = dwh.raw.load_dialogues_dict("train")
    assert len(dlgs) == 1516  # 3032 / 2 = 1516 уникальных диалогов


def test_dwh_raw_load_participant_texts(dwh):
    """Тест агрегированной загрузки текстов участников из DWH для RuBERT и NLP."""
    df_train = dwh.raw.load_participant_texts("train", limit=10)
    assert len(df_train) == 10
    assert set(df_train.columns) == {"dialog_id", "participant_index", "is_bot", "text"}
    assert isinstance(df_train["text"][0], str)

    df_test = dwh.raw.load_participant_texts("test", limit=10)
    assert len(df_test) == 10
    assert "ID" in df_test.columns
    assert "text" in df_test.columns
