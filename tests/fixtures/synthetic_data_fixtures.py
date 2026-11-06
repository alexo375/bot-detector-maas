"""Модульные фикстуры генерации синтетических датасетов и лоадеров обучения."""

from typing import Any

import numpy as np
import pytest


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
