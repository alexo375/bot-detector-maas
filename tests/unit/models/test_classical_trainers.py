"""Модульные тесты для классических тренеров (CatBoost, LightGBM, XGBoost)."""

from typing import Any

import numpy as np
import pytest

from project.models.classical import (
    CatBoostSearchConfig,
    CatBoostTrainer,
    ClassicalTrainerConfig,
    LightGBMTrainer,
    XGBoostTrainer,
)


@pytest.mark.parametrize(
    "trainer_cls",
    [
        CatBoostTrainer,
        LightGBMTrainer,
        XGBoostTrainer,
    ],
)
def test_classical_trainer_tune_and_fit(
    isolated_mlflow_env: str,
    synthetic_data: dict[str, Any],
    trainer_cls: type[Any],
) -> None:
    """Проверяет независимую работу tune() и fit() для каждого семейства моделей."""
    del isolated_mlflow_env
    cfg = ClassicalTrainerConfig(
        experiment_name="unit-tests",
        run_name=f"test_{trainer_cls.__name__}",
        n_trials=2,
        patience=2,
        n_splits=2,
        early_stopping_rounds=5,
        register_as=None,
    )

    trainer = trainer_cls(config=cfg)

    # 1. Проверяем процедуру поиска параметров
    tune_res = trainer.tune(
        X=synthetic_data["X_train"],
        y=synthetic_data["y_train"],
        groups=synthetic_data["groups"],
    )
    assert tune_res.best_params is not None
    assert tune_res.best_score > 0
    assert trainer.best_params == tune_res.best_params

    # 2. Проверяем обучение ансамбля с найденными параметрами
    fit_res = trainer.fit(
        X_train=synthetic_data["X_train"],
        y_train=synthetic_data["y_train"],
        groups=synthetic_data["groups"],
        X_test=synthetic_data["X_test"],
        dialog_ids_train=synthetic_data["dialog_ids_train"],
        participant_indices_train=synthetic_data["participant_indices_train"],
        dialog_ids_test=synthetic_data["dialog_ids_test"],
        participant_indices_test=synthetic_data["participant_indices_test"],
        test_ids=synthetic_data["test_ids"],
        feature_names=synthetic_data["feature_names"],
    )

    assert len(fit_res.oof_calibrated) == len(synthetic_data["y_train"])
    assert len(fit_res.test_calibrated) == len(synthetic_data["X_test"])
    assert "val_logloss" in fit_res.metrics
    assert "val_roc_auc" in fit_res.metrics
    assert len(fit_res.models) == 2
    assert len(fit_res.selected_features) == 8


def test_feature_pruning_tuning(isolated_mlflow_env: str, synthetic_data: dict[str, Any]) -> None:
    """Проверяет совместную оптимизацию гиперпараметров и числа признаков."""
    del isolated_mlflow_env
    search_cfg = CatBoostSearchConfig(
        tune_feature_count=True,
        min_features=3,
        max_features=6,
    )
    cfg = ClassicalTrainerConfig(
        experiment_name="unit-tests",
        run_name="test_pruning",
        n_trials=2,
        patience=2,
        n_splits=2,
        register_as=None,
    )

    trainer = CatBoostTrainer(config=cfg, search_config=search_cfg)
    ranking = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    tune_res = trainer.tune(
        X=synthetic_data["X_train"],
        y=synthetic_data["y_train"],
        groups=synthetic_data["groups"],
        feature_ranking_indices=ranking,
    )

    assert tune_res.optimal_features_count is not None
    assert 3 <= tune_res.optimal_features_count <= 6

    fit_res = trainer.fit(
        X_train=synthetic_data["X_train"],
        y_train=synthetic_data["y_train"],
        groups=synthetic_data["groups"],
        X_test=synthetic_data["X_test"],
        dialog_ids_train=synthetic_data["dialog_ids_train"],
        participant_indices_train=synthetic_data["participant_indices_train"],
        dialog_ids_test=synthetic_data["dialog_ids_test"],
        participant_indices_test=synthetic_data["participant_indices_test"],
        test_ids=synthetic_data["test_ids"],
        feature_names=synthetic_data["feature_names"],
        feature_ranking_indices=ranking,
    )

    assert len(fit_res.selected_features) == tune_res.optimal_features_count


def test_optuna_early_stopping_callback_stops_study() -> None:
    """Проверка срабатывания early stopping в Optuna при превышении patience."""
    from unittest.mock import MagicMock

    from project.models.callbacks.early_stopping import OptunaEarlyStoppingCallback

    cb = OptunaEarlyStoppingCallback(patience=2)
    study = MagicMock()

    t1 = MagicMock()
    t1.value = 0.5
    cb(study, t1)
    study.stop.assert_not_called()

    t2 = MagicMock()
    t2.value = 0.6  # ухудшение 1
    cb(study, t2)
    study.stop.assert_not_called()

    t3 = MagicMock()
    t3.value = 0.7  # ухудшение 2 >= patience
    cb(study, t3)
    study.stop.assert_called_once()
