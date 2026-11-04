import numpy as np
import pandas as pd
import polars as pl
import pytest
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from project.tracking import ExperimentTracker, ExperimentTrackerBuilder


def test_builder_validation_fails_without_dataset():
    builder = (
        ExperimentTrackerBuilder(run_name="test_fail_run")
        .set_description("Тестовый эксперимент")
        .set_params(model_family="CatBoost", params={"depth": 5})
        .set_metrics(val_logloss=0.15)
    )
    with pytest.raises(ValueError, match="Не задан датасет"):
        builder.build()


def test_builder_validation_fails_without_params():
    builder = (
        ExperimentTrackerBuilder(run_name="test_fail_run")
        .set_description("Тестовый эксперимент")
        .set_dataset(pl.DataFrame(np.zeros((10, 5))), name="test_data")
        .set_metrics(val_logloss=0.15)
    )
    with pytest.raises(ValueError, match="Не заданы параметры"):
        builder.build()


def test_builder_validation_fails_without_metrics():
    builder = (
        ExperimentTrackerBuilder(run_name="test_fail_run")
        .set_description("Тестовый эксперимент")
        .set_dataset(pl.DataFrame(np.zeros((10, 5))), name="test_data")
        .set_params(model_family="CatBoost")
    )
    with pytest.raises(ValueError, match="Не заданы метрики"):
        builder.build()


def test_builder_validation_fails_without_description():
    builder = (
        ExperimentTrackerBuilder(run_name="test_fail_run")
        .set_dataset(pl.DataFrame(np.zeros((10, 5))), name="test_data")
        .set_params(model_family="CatBoost")
        .set_metrics(val_logloss=0.15)
    )
    with pytest.raises(ValueError, match="Не задано обязательное понятное описание эксперимента на русском языке"):
        builder.build()


def test_builder_passes_when_description_disabled():
    tracker = (
        ExperimentTrackerBuilder(run_name="test_pass_no_desc")
        .require_description(False)
        .require_graphics(False)
        .require_feature_tree(False)
        .require_submission(False)
        .set_dataset(pl.DataFrame(np.zeros((10, 5))), name="test_data")
        .set_params(model_family="CatBoost")
        .set_metrics(val_logloss=0.15)
        .build()
    )
    assert tracker is not None


def test_tracker_strict_validation_on_exit(mlflow_sqlite_uri: str) -> None:
    test_uri = mlflow_sqlite_uri
    tracker = ExperimentTracker(
        run_name="test_unvalidated_exit",
        experiment_name="test_experiment",
        tracking_uri=test_uri,
        require_description=False,
        require_graphics=False,
        require_feature_tree=False,
    )
    with pytest.raises(ValueError, match="Ошибка MLOps-валидации"), tracker:
        # Missing dataset, params, metrics!
        pass


def test_tracker_successful_lifecycle(mlflow_sqlite_uri: str) -> None:
    test_uri = mlflow_sqlite_uri
    data = pl.DataFrame(np.random.randn(20, 4))
    y = np.random.randint(0, 2, size=20)
    tracker = (
        ExperimentTrackerBuilder(
            run_name="test_successful_builder_run",
            experiment_name="test_experiment",
            tracking_uri=test_uri,
            description="Успешный запуск модели LightGBM",
        )
        .require_graphics(False)
        .require_feature_tree(False)
        .require_submission(False)
        .set_dataset(data=data, name="test_builder_dataset", targets=y)
        .set_params(model_family="LightGBM", params={"n_estimators": 50})
        .set_metrics(val_logloss=0.22, val_roc_auc=0.95, val_accuracy=0.90)
        .build()
    )

    with tracker:
        pass


def test_tracker_model_registry_auto_promotion(mlflow_sqlite_uri: str) -> None:
    test_uri = mlflow_sqlite_uri
    reg_name = "test_model_registry"

    x_dummy = pl.DataFrame(np.random.randn(20, 4))
    y_dummy = np.random.randint(0, 2, size=20)
    model1 = LogisticRegression().fit(x_dummy, y_dummy)
    model2 = LogisticRegression().fit(x_dummy, y_dummy)
    model3 = LogisticRegression().fit(x_dummy, y_dummy)

    # Run 1: initial model with logloss 0.25 -> must become champion
    t1 = (
        ExperimentTrackerBuilder(
            "run_1",
            experiment_name="test_exp",
            tracking_uri=test_uri,
            description="Базовая модель 1",
        )
        .require_graphics(False)
        .require_feature_tree(False)
        .require_submission(False)
        .set_dataset(x_dummy, name="d1", targets=y_dummy)
        .set_params(model_family="LogReg")
        .set_metrics(val_logloss=0.25)
        .build()
    )
    with t1:
        t1.log_model_instance(model1, register_as=reg_name, metric_for_promotion="val_logloss")

    client = MlflowClient(tracking_uri=test_uri)
    champ_v1 = client.get_model_version_by_alias(reg_name, "champion")
    assert str(champ_v1.version) == "1"

    # Run 2: worse model with logloss 0.35 -> must become challenger, champion remains v1
    t2 = (
        ExperimentTrackerBuilder(
            "run_2",
            experiment_name="test_exp",
            tracking_uri=test_uri,
            description="Худшая модель 2",
        )
        .require_graphics(False)
        .require_feature_tree(False)
        .require_submission(False)
        .set_dataset(x_dummy, name="d2", targets=y_dummy)
        .set_params(model_family="LogReg")
        .set_metrics(val_logloss=0.35)
        .build()
    )
    with t2:
        t2.log_model_instance(model2, register_as=reg_name, metric_for_promotion="val_logloss")

    champ_v2 = client.get_model_version_by_alias(reg_name, "champion")
    chall_v2 = client.get_model_version_by_alias(reg_name, "challenger")
    assert str(champ_v2.version) == "1"
    assert str(chall_v2.version) == "2"

    # Run 3: better model with logloss 0.12 -> becomes new champion, v1 demoted to challenger!
    t3 = (
        ExperimentTrackerBuilder(
            "run_3",
            experiment_name="test_exp",
            tracking_uri=test_uri,
            description="Лучшая чемпионская модель 3",
        )
        .require_graphics(False)
        .require_feature_tree(False)
        .require_submission(False)
        .set_dataset(x_dummy, name="d3", targets=y_dummy)
        .set_params(model_family="LogReg")
        .set_metrics(val_logloss=0.12)
        .build()
    )
    with t3:
        t3.log_model_instance(model3, register_as=reg_name, metric_for_promotion="val_logloss")

    champ_v3 = client.get_model_version_by_alias(reg_name, "champion")
    chall_v3 = client.get_model_version_by_alias(reg_name, "challenger")
    assert str(champ_v3.version) == "3"
    assert str(chall_v3.version) == "1"


def test_builder_validation_fails_without_val_logloss():
    builder = (
        ExperimentTrackerBuilder(run_name="test_missing_logloss")
        .set_description("Тестовый эксперимент")
        .set_dataset(pl.DataFrame(np.zeros((10, 5))), name="test_data")
        .set_params(model_family="CatBoost")
        .set_metrics(val_roc_auc=0.95)
    )
    with pytest.raises(ValueError, match="Не задана обязательная целевая метрика val_logloss"):
        builder.build()


def test_builder_validation_fails_with_empty_model_family():
    builder = (
        ExperimentTrackerBuilder(run_name="test_empty_family")
        .set_description("Тестовый эксперимент")
        .set_dataset(pl.DataFrame(np.zeros((10, 5))), name="test_data")
        .set_params(model_family="")
        .set_metrics(val_logloss=0.15)
    )
    with pytest.raises(ValueError, match="Не указано обязательное семейство модели model_family"):
        builder.build()


def test_tracker_fails_if_graphics_required_but_missing(mlflow_sqlite_uri: str) -> None:
    test_uri = mlflow_sqlite_uri
    tracker = (
        ExperimentTrackerBuilder(run_name="test_missing_graphics", tracking_uri=test_uri)
        .set_description("Тест отсутствия графиков")
        .require_graphics(True)
        .require_feature_tree(False)
        .require_submission(False)
        .set_dataset(pl.DataFrame(np.zeros((10, 5))), name="test_data")
        .set_params(model_family="CatBoost")
        .set_metrics(val_logloss=0.15)
        .build()
    )
    with pytest.raises(ValueError, match="Обязательные графики не залогированы"), tracker:
        pass


def test_tracker_fails_if_submission_required_but_missing(mlflow_sqlite_uri: str) -> None:
    test_uri = mlflow_sqlite_uri
    tracker = (
        ExperimentTrackerBuilder(run_name="test_missing_sub", tracking_uri=test_uri)
        .set_description("Тест отсутствия сабмишена")
        .require_graphics(False)
        .require_feature_tree(False)
        .require_submission(True)
        .set_dataset(pl.DataFrame(np.zeros((10, 5))), name="test_data")
        .set_params(model_family="CatBoost")
        .set_metrics(val_logloss=0.15)
        .build()
    )
    with pytest.raises(ValueError, match="Обязательный файл сабмишена не залогирован"), tracker:
        pass


def test_tracker_fails_if_feature_importance_required_but_missing(mlflow_sqlite_uri: str) -> None:
    test_uri = mlflow_sqlite_uri
    tracker = (
        ExperimentTrackerBuilder(run_name="test_missing_feat_imp", tracking_uri=test_uri)
        .set_description("Тест отсутствия важности признаков")
        .require_graphics(False)
        .require_feature_tree(True)
        .require_submission(False)
        .set_dataset(pl.DataFrame(np.zeros((10, 5))), name="test_data")
        .set_params(model_family="CatBoost")
        .set_metrics(val_logloss=0.15)
        .build()
    )
    with pytest.raises(ValueError, match="Обязательное дерево признаков не залогировано"), tracker:
        pass


def test_tracker_logs_graphics_and_features_and_submission(mlflow_sqlite_uri: str) -> None:
    test_uri = mlflow_sqlite_uri
    x = np.random.randn(50, 4)
    y = np.random.randint(0, 2, size=50)
    features = ["feat_0", "feat_1", "feat_2", "feat_3"]
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(x, y)
    probs = model.predict_proba(x)[:, 1]

    sub_df = pd.DataFrame({"ID": np.arange(50), "is_bot": probs})

    tracker = (
        ExperimentTrackerBuilder(run_name="test_full_artifacts", tracking_uri=test_uri)
        .set_description("Полный тест логирования графиков, признаков и сабмишена")
        .require_graphics(True)
        .require_feature_tree(True)
        .require_submission(True)
        .set_dataset(pl.DataFrame(x), name="dataset_full", targets=y)
        .set_params(model_family="RandomForest", n_features=4, feature_names=features)
        .set_metrics(
            val_logloss=0.25,
            val_logloss_raw=0.26,
            val_roc_auc=0.90,
            val_accuracy=0.85,
            val_f1=0.82,
        )
        .build()
    )

    with tracker as t:
        run_id = t.active_run.info.run_id
        t.log_feature_tree(
            {"architecture": "SingleModel", "features": [{"name": f, "importance": 0.1} for f in features]}
        )
        t.log_all_graphics(y_true=y, y_pred_proba=probs)
        t.log_submission(sub_df, filename="submission.csv")

    client = MlflowClient(tracking_uri=test_uri)
    target_run = client.get_run(run_id)
    metrics = target_run.data.metrics
    for key in ("val_logloss", "val_logloss_raw", "val_roc_auc", "val_accuracy", "val_f1"):
        assert key in metrics

    root_artifacts = [a.path for a in client.list_artifacts(target_run.info.run_id)]
    graphics_artifacts = [a.path for a in client.list_artifacts(target_run.info.run_id, "graphics")]

    assert sorted(graphics_artifacts) == [
        "graphics/calibration_curve.png",
        "graphics/confusion_matrix.png",
        "graphics/prediction_distribution.png",
        "graphics/roc_curve.png",
    ]
    assert "features" not in root_artifacts
    assert "feature_tree.json" in root_artifacts
    assert "submission.csv" in root_artifacts


def test_builder_defer_metrics(mlflow_sqlite_uri: str) -> None:
    test_uri = mlflow_sqlite_uri
    tracker = (
        ExperimentTrackerBuilder(run_name="test_defer_metrics", tracking_uri=test_uri)
        .set_description("Тест отложенных метрик")
        .set_dataset(pl.DataFrame(np.zeros((10, 5))), name="dataset_deferred")
        .set_params(model_family="LightGBM")
        .require_graphics(False)
        .require_feature_tree(False)
        .require_submission(False)
        .defer_metrics()
        .build()
    )
    assert tracker is not None
    with tracker as t:
        t.log_metrics(val_logloss=0.20)
