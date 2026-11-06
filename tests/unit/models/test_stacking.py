"""Модульные тесты универсального стекера моделей TwoLevelStackingClassifier."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from project.models.classical import TwoLevelStackingClassifier


def test_two_level_stacking_classifier_lifecycle() -> None:
    """Проверка полного цикла обучения, OOF-матрицы, предсказания и генерации дерева ансамбля."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.choice([0, 1], size=100)
    groups = np.repeat(np.arange(20), 5)

    base_models = [
        ("dt1", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ("dt2", DecisionTreeClassifier(max_depth=4, random_state=43)),
    ]
    meta_model = LogisticRegression(C=1.0, random_state=42)

    stacker = TwoLevelStackingClassifier(
        base_models=base_models,
        meta_model=meta_model,
        n_splits=3,
        random_state=42,
    )
    feature_names = [f"col_{i}" for i in range(10)]
    stacker.fit(X, y, groups=groups, feature_names=feature_names)

    assert stacker.oof_matrix_ is not None
    assert stacker.oof_matrix_.shape == (100, 2)
    assert stacker.oof_predictions_ is not None
    assert len(stacker.oof_predictions_) == 100

    probs = stacker.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    preds = stacker.predict(X)
    assert len(preds) == 100

    tree = stacker.get_ensemble_tree()
    assert tree["ensemble_architecture"] == "TwoLevelStackingClassifier"
    assert "level2_meta_model" in tree
    assert "level1_base_models" in tree
    assert "dt1" in tree["level1_base_models"]
    assert "dt2" in tree["level1_base_models"]
    assert "weights" in tree["level2_meta_model"]
    assert "dt1" in tree["level2_meta_model"]["weights"]
    assert "dt2" in tree["level2_meta_model"]["weights"]
    assert "features" in tree
    assert len(tree["features"]) == 2
    assert len(tree["level1_base_models"]["dt1"]["features"]) == 10
    assert len(tree["level1_base_models"]["dt2"]["features"]) == 10


def test_two_level_stacking_direct_meta_features() -> None:
    """Проверка работы TwoLevelStackingClassifier напрямую с мета-признаками (логитами)."""
    np.random.seed(42)
    Z = np.random.randn(100, 4)
    y = np.random.choice([0, 1], size=100)
    groups = np.repeat(np.arange(20), 5)
    feature_names = ["logit_cb", "logit_lgb", "logit_xgb", "logit_rubert"]

    stacker = TwoLevelStackingClassifier(
        base_models=None,
        meta_model=LogisticRegression(C=1.0, random_state=42),
        n_splits=3,
        random_state=42,
    )
    stacker.fit(Z, y, groups=groups, feature_names=feature_names)

    assert stacker.meta_model_ is not None
    assert stacker.coef_.shape == (1, 4)
    assert isinstance(stacker.intercept_, np.ndarray)
    assert len(stacker.feature_importances_) == 4

    probs = stacker.predict_proba(Z)
    assert probs.shape == (100, 2)

    tree = stacker.get_ensemble_tree()
    assert tree["architecture"] == "TwoLevelStackingClassifier"
    assert len(tree["features"]) == 4
    assert tree["features"][0]["name"] in feature_names


def test_stacking_trainer_and_search_config() -> None:
    """Проверка StackingSearchConfig и TwoLevelStackingTrainer."""
    from unittest.mock import MagicMock

    import optuna

    from project.models.classical import ClassicalTrainerConfig, StackingSearchConfig, TwoLevelStackingTrainer

    search_cfg = StackingSearchConfig()
    fixed = search_cfg.get_fixed_parameters()
    assert fixed["solver"] == "lbfgs"

    trial = MagicMock(spec=optuna.Trial)
    trial.suggest_float.return_value = 0.5
    trial.suggest_categorical.return_value = True
    sampled = search_cfg.sample_parameters(trial)
    assert sampled["C"] == 0.5
    assert sampled["fit_intercept"] is True

    cfg = ClassicalTrainerConfig(experiment_name="test-exp", run_name="test-stacker", n_splits=3)
    trainer = TwoLevelStackingTrainer(config=cfg, search_config=search_cfg)
    assert trainer.model_family == "TwoLevelStackingClassifier"

    model = trainer.build_model({"C": 0.5, "fit_intercept": True})
    assert isinstance(model, LogisticRegression)
    assert model.C == 0.5

    # Проверка fit_fold
    X_tr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y_tr = np.array([0, 0, 1, 1])
    trainer.fit_fold(model, X_tr, y_tr, X_tr[:2], y_tr[:2])
    assert hasattr(model, "coef_")


def test_two_level_stacking_default_meta_model_and_kfold() -> None:
    """Проверка создания мета-модели по умолчанию и разбиения KFold без групп."""
    Z = np.random.randn(30, 2)
    y = np.random.choice([0, 1], size=30)

    stacker = TwoLevelStackingClassifier(base_models=None, meta_model=None, n_splits=2)
    stacker.fit(Z, y, groups=None)

    assert stacker.meta_model_ is not None
    assert isinstance(stacker.meta_model_, LogisticRegression)
    preds = stacker.predict_proba(Z)
    assert preds.shape == (30, 2)


def test_stacking_trainer_fit_lifecycle() -> None:
    """Проверка полного цикла обучения и логирования в TwoLevelStackingTrainer.fit."""
    from unittest.mock import MagicMock

    from project.models.classical import ClassicalTrainerConfig, TwoLevelStackingTrainer

    cfg = ClassicalTrainerConfig(
        experiment_name="test_stacking_exp",
        run_name="test_stacking_run",
        n_splits=2,
        pairwise_eps=0.01,
    )
    trainer = TwoLevelStackingTrainer(config=cfg)

    mock_tracker = MagicMock()
    mock_tracker._has_logged_dataset = True
    mock_tracker.active_run.info.run_id = "test_run_xyz"
    trainer._tracker = mock_tracker

    n_samples = 20
    X_tr = np.random.randn(n_samples, 4)
    y_tr = np.array([0, 1] * 10)
    groups = np.repeat(np.arange(10), 2)
    d_ids_tr = np.array([f"dlg_{i // 2}" for i in range(n_samples)])
    p_idx_tr = np.array([i % 2 for i in range(n_samples)])

    X_te = np.random.randn(4, 4)
    d_ids_te = np.array(["d_te_0", "d_te_0", "d_te_1", "d_te_1"])
    p_idx_te = np.array([0, 1, 0, 1])
    test_ids = np.array([1, 2, 3, 4])

    res = trainer.fit(
        X_train=X_tr,
        y_train=y_tr,
        groups=groups,
        X_test=X_te,
        dialog_ids_train=d_ids_tr,
        participant_indices_train=p_idx_tr,
        dialog_ids_test=d_ids_te,
        participant_indices_test=p_idx_te,
        test_ids=test_ids,
    )

    assert res.run_id == "test_run_xyz"
    assert len(res.metrics) > 0
    assert "val_logloss" in res.metrics
    assert len(res.models) == 2
    mock_tracker.log_metrics.assert_called_once()
    mock_tracker.log_feature_tree.assert_called_once()


def test_evaluate_model_promotion_honors_max_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModelRegistryService.evaluate_promotion корректно учитывает MetricOptimizationMode.MAX."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from project.tracking.experiment_tracker.loggers.model_registry import ModelRegistryService
    from project.tracking.experiment_tracker.schemas.metric_mode import MetricOptimizationMode

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = SimpleNamespace(version="1", run_id="champ_run")
    mock_client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(metrics={"val_roc_auc": 0.90}))

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda **_kwargs: mock_client)
    ModelRegistryService.evaluate_promotion(
        mv=SimpleNamespace(version="2"),
        register_as="bot_detector",
        metric_for_promotion="val_roc_auc",
        mode=MetricOptimizationMode.MAX,
        logged_metrics={"val_roc_auc": 0.95},
    )
    mock_client.set_registered_model_alias.assert_any_call("bot_detector", "champion", "2")


def test_build_hierarchical_ensemble_tree_expands_level1_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EnsembleTreeBuilder.build разворачивает признаки моделей Level-1 без голых ссылок level1_runs."""
    import json
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from project.models.classical.trainers.stacking_tree import EnsembleTreeBuilder

    base_dir = tmp_path / "l1_trees"
    base_dir.mkdir()
    cb_tree = {
        "architecture": "SingleModel",
        "model_family": "CatBoostClassifier",
        "n_features": 2,
        "features": [{"name": "echo_all", "importance": 8.5}, {"name": "cyr_ratio", "importance": 4.2}],
    }
    cb_file = base_dir / "cb_feature_tree.json"
    cb_file.write_text(json.dumps(cb_tree), encoding="utf-8")

    lgb_tree = {
        "architecture": "SingleModel",
        "model_family": "LGBMClassifier",
        "n_features": 1,
        "features": [{"name": "phrase_mean", "importance": 120.0}],
    }
    lgb_file = base_dir / "lgb_feature_tree.json"
    lgb_file.write_text(json.dumps(lgb_tree), encoding="utf-8")

    Z = np.random.randn(40, 2)
    y = np.array([0, 1] * 20)
    stacker = TwoLevelStackingClassifier(base_models=None, n_splits=2, random_state=42)
    stacker.fit(Z, y, feature_names=["logit_catboost", "logit_lightgbm"])

    mock_client = MagicMock()
    mock_client.get_run.side_effect = lambda rid: SimpleNamespace(
        data=SimpleNamespace(
            metrics={"val_logloss": 0.17 if rid == "cb_run" else 0.18, "optuna_trial_val_logloss": 0.25},
            params={"model_family": "CatBoostClassifier" if rid == "cb_run" else "LGBMClassifier", "depth": "6"},
        )
    )
    mock_client.download_artifacts.side_effect = lambda rid, _path, _dst_path=None, **_kw: str(
        cb_file if rid == "cb_run" else lgb_file
    )

    monkeypatch.setattr("project.models.classical.trainers.stacking_tree.MlflowClient", lambda **_kw: mock_client)
    tree = EnsembleTreeBuilder.build(
        stacker,
        level1_runs={"catboost_detector": "cb_run", "lightgbm_detector": "lgb_run"},
    )

    assert "level1_runs" not in tree
    assert "catboost_detector" in tree["level1_base_models"]
    assert "lightgbm_detector" in tree["level1_base_models"]
    cb_node = tree["level1_base_models"]["catboost_detector"]
    assert cb_node["run_id"] == "cb_run"
    assert cb_node["weight_in_stacker"] == tree["weights"]["logit_catboost"]
    assert cb_node["oof_metrics"] == {"val_logloss": 0.17}
    assert cb_node["features"] == cb_tree["features"]
    assert tree["features"][0]["importance"] >= tree["features"][1]["importance"]
