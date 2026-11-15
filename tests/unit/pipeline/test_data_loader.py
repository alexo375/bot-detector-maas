"""Модульные тесты для компонента TrainingDataLoader."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from project.features.store import FeatureStore
from project.pipeline.mlflow_retrainer.components.data_loader import TrainingDataLoader


def test_training_data_loader_clickhouse_success() -> None:
    """Проверка успешной загрузки обучающей выборки из ClickHouse через load_retraining_bundle."""
    mock_client = MagicMock()
    mock_store = MagicMock(spec=FeatureStore)
    mock_store.feature_names = ["f1", "f2"]
    mock_store.transform.return_value = np.zeros((10, 2), dtype=np.float32)

    rows = []
    for i in range(10):
        rows.append(
            (
                f"dlg_{i}",
                json.dumps([{"participant_index": 0, "text": "hello"}]),
                0,
                1 if i % 2 == 0 else 0,
            )
        )

    mock_client.query.return_value.result_rows = rows
    loader = TrainingDataLoader(feature_store=mock_store, client=mock_client)

    bundle = loader.load_retraining_bundle(sample_size=10, include_pseudo_labels=False)

    assert bundle.X.shape == (10, 2)
    assert len(bundle.y) == 10
    assert len(bundle.texts_df) == 10
    assert bundle.feature_names == ["f1", "f2"]
    assert not bool(np.any(bundle.is_pseudo))
    mock_store.transform.assert_called_once()


def test_training_data_loader_fail_fast_on_clickhouse_error() -> None:
    """При сбое ClickHouse TrainingDataLoader падает без фолбэка на локальные файлы."""
    mock_client = MagicMock()
    mock_client.query.side_effect = RuntimeError("ClickHouse connection reset")
    mock_store = MagicMock(spec=FeatureStore)
    mock_store.feature_names = ["f1", "f2", "f3"]

    loader = TrainingDataLoader(feature_store=mock_store, client=mock_client)
    with pytest.raises(RuntimeError, match="ClickHouse connection reset"):
        loader.load_retraining_bundle(sample_size=20)


def test_training_data_loader_retraining_bundle_with_pseudo_labels() -> None:
    """Проверка отбора pseudo-labeled строк из chat_logs и разметки is_pseudo."""
    mock_client = MagicMock()
    mock_store = MagicMock(spec=FeatureStore)
    mock_store.feature_names = ["f1", "f2"]
    mock_store.transform.side_effect = lambda dlgs, **_kwargs: np.zeros((len(dlgs) * 2, 2), dtype=np.float32)

    gold_rows = []
    for i in range(10):
        msgs = json.dumps(
            [
                {"participant_index": 0, "text": "Привет!"},
                {"participant_index": 1, "text": "Привет, как дела?"},
            ]
        )
        gold_rows.append((f"gold_{i}", msgs, 0, i % 2))
        gold_rows.append((f"gold_{i}", msgs, 1, (i + 1) % 2))

    pseudo_msgs = json.dumps(
        [
            {"participant_index": 0, "text": "Опишите вашу проблему подробно."},
            {"participant_index": 1, "text": "Ок, сейчас напишу"},
        ]
    )
    pseudo_chat_rows = [
        ("prod_bot_1", pseudo_msgs, 0.96, 0.03),
        ("prod_human_1", pseudo_msgs, 0.04, 0.05),
    ]

    mock_client.query.side_effect = [
        MagicMock(result_rows=gold_rows),
        MagicMock(result_rows=pseudo_chat_rows),
    ]
    loader = TrainingDataLoader(feature_store=mock_store, client=mock_client)
    bundle = loader.load_retraining_bundle(sample_size=20, include_pseudo_labels=True, max_pseudo_ratio=0.30)

    assert len(bundle.dialogues) == 12
    assert len(bundle.X) == 24
    assert int(np.sum(~bundle.is_pseudo)) == 20
    assert int(np.sum(bundle.is_pseudo)) == 4
    second_sql = mock_client.query.call_args_list[1][0][0]
    assert "drift_inject_" in second_sql


def test_training_data_loader_fails_fast_on_insufficient_rows() -> None:
    """При недостатке строк (< 10) в ClickHouse raw_dialogues выбрасывается RuntimeError."""
    mock_client = MagicMock()
    mock_client.query.return_value.result_rows = []
    loader = TrainingDataLoader(client=mock_client)

    with pytest.raises(RuntimeError, match="minimum 10 required"):
        loader.load_retraining_bundle(sample_size=21, include_pseudo_labels=True)
