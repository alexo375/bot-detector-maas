"""Модульные фикстуры генерации синтетических датасетов и лоадеров обучения."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from project.pipeline.mlflow_retrainer.components.data_loader import TrainingBundle, TrainingDataLoader


@pytest.fixture
def synthetic_data() -> dict[str, Any]:
    """Синтетический сбалансированный датасет диалогов для быстрых тестов моделей и постобработчиков."""
    rng = np.random.default_rng(42)
    n_dialogs = 30
    n_participants = 2
    n_samples = n_dialogs * n_participants
    n_features = 8

    X = rng.normal(0, 1, size=(n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples).astype(int)

    dialog_ids = np.repeat([f"dlg_{i}" for i in range(n_dialogs)], n_participants)
    participant_indices = np.tile([0, 1], n_dialogs)
    test_ids = np.array([f"{d}_{p}" for d, p in zip(dialog_ids[:10], participant_indices[:10], strict=False)])

    return {
        "X_train": X,
        "y_train": y,
        "groups": dialog_ids,
        "X_test": X[:10],
        "dialog_ids_train": dialog_ids,
        "participant_indices_train": participant_indices,
        "dialog_ids_test": dialog_ids[:10],
        "participant_indices_test": participant_indices[:10],
        "test_ids": test_ids,
        "feature_names": [f"feat_{i}" for i in range(n_features)],
    }


@pytest.fixture
def synthetic_training_loader() -> TrainingDataLoader:
    """Генератор детерминированного синтетического датасета с информативными и шумовыми признаками."""
    rng = np.random.default_rng(42)
    n_samples = 60
    n_features = 16

    y = rng.integers(0, 2, size=n_samples).astype(np.int32)
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    for i in range(4):
        X[:, i] += (y * 2.5 - 1.25).astype(np.float32)

    feature_names = [f"feat_{i}" for i in range(n_features)]
    n_pairs = n_samples // 2
    dialog_ids = np.repeat([f"syn_dlg_{i}" for i in range(n_pairs)], 2)
    participant_indices = np.tile([0, 1], n_pairs)
    texts_df = pl.DataFrame(
        {
            "dialog_id": dialog_ids.tolist(),
            "participant_index": participant_indices.tolist(),
            "text": [f"msg_{i}" for i in range(n_samples)],
            "is_bot": y.tolist(),
            "is_pseudo": [False] * n_samples,
        }
    )

    loader = MagicMock(spec=TrainingDataLoader)
    loader.load_retraining_bundle.return_value = TrainingBundle(
        X=X,
        y=y,
        feature_names=feature_names,
        texts_df=texts_df,
        dialogues=[],
        is_pseudo=np.zeros(n_samples, dtype=bool),
    )
    return loader
