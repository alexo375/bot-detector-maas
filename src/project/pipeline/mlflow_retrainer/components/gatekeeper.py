"""Оркестратор адаптивного дообучения моделей с паритетом офлайн-архитектуры."""

import os
from typing import Any

import numpy as np

from project.core.config import config
from project.core.logging import ProjectLogger
from project.infrastructure.clickhouse import FeatureDWH
from project.pipeline.mlflow_retrainer.components.champion_evaluator import ChampionEvaluator
from project.pipeline.mlflow_retrainer.components.champion_params import ChampionParamResolver
from project.pipeline.mlflow_retrainer.components.data_loader import TrainingBundle, TrainingDataLoader
from project.pipeline.mlflow_retrainer.components.dataset_splitter import DatasetSplit, DatasetSplitter
from project.pipeline.mlflow_retrainer.schemas import RetrainEvaluationResult
from project.pipeline.mlflow_retrainer.stages import (
    FeatureSelectionStage,
    Level1Stage,
    StackingAndPromotionStage,
)
from project.pipeline.mlflow_retrainer.stages.feature_selection import FeaturePruner

logger = ProjectLogger.get_logger(__name__)


class RetrainingGatekeeper:
    """Сервис переобучения моделей детекции ботов с паритетом офлайн-архитектуры и изоляцией валидации."""

    def __init__(
        self,
        n_trials: int | None = None,
        patience: int | None = None,
        data_loader: TrainingDataLoader | None = None,
        n_estimators: int | None = None,
        fast_mode: bool = False,
        feature_keep_ratio: float = 0.80,
        experiment_name: str = "bot_detection_retraining",
        corr_threshold: float | None = None,
        sample_size: int | None = None,
        n_splits: int | None = None,
        early_stopping_rounds: int | None = None,
        dwh: FeatureDWH | None = None,
        drift_context: dict[str, Any] | None = None,
    ) -> None:
        self.fast_mode = fast_mode
        base_trials = n_trials if n_trials is not None else config.retrain.n_trials
        base_patience = patience if patience is not None else config.retrain.patience
        base_estimators = n_estimators if n_estimators is not None else config.retrain.n_estimators
        base_samples = sample_size if sample_size is not None else config.retrain.sample_size
        base_early_stop = (
            early_stopping_rounds if early_stopping_rounds is not None else config.retrain.early_stopping_rounds
        )

        self.n_trials = 3 if fast_mode else int(base_trials)
        self.patience = 1 if fast_mode else int(base_patience)
        self.n_estimators = min(int(base_estimators), 25) if fast_mode else int(base_estimators)
        self.sample_size = 300 if fast_mode else int(base_samples)
        self.early_stopping_rounds = 5 if fast_mode else int(base_early_stop)
        self.n_splits = int(n_splits if n_splits is not None else config.retrain.n_splits)
        self.corr_threshold = float(corr_threshold if corr_threshold is not None else config.retrain.corr_threshold)
        self.feature_keep_ratio = max(0.2, min(1.0, feature_keep_ratio))
        self.experiment_name = experiment_name
        self.data_loader = data_loader or TrainingDataLoader()
        self.drift_context = drift_context or {}

        if dwh is not None:
            self.dwh: FeatureDWH | None = dwh
        elif os.getenv("PYTEST_CURRENT_TEST"):
            self.dwh = None
        else:
            self.dwh = FeatureDWH()

    def fetch_champion_loss(self) -> float | None:
        """Получение валидационного LogLoss `@champion` или `None` до первого промоушена."""
        return ChampionEvaluator.fetch_champion_loss()

    def fetch_champion_importance(self) -> dict[str, float]:
        """Получение карты важности признаков из продакшн-модели Champion."""
        return ChampionEvaluator.fetch_champion_importance()

    def _load_bundle(self) -> TrainingBundle:
        """Строгая загрузка `TrainingBundle` из ClickHouse DWH (без фолбэков)."""
        bundle = self.data_loader.load_retraining_bundle(
            sample_size=self.sample_size,
            include_pseudo_labels=True,
        )
        if not isinstance(bundle, TrainingBundle):
            raise TypeError(f"Expected TrainingBundle from data_loader, got {type(bundle).__name__}")
        return bundle

    def _prepare_pruned_features(
        self,
        split: DatasetSplit,
        bundle: TrainingBundle,
        n_cv_splits: int,
    ) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
        """Обучает 8-view `FeatureStore` на train, удаляет `|corr| > 0.95` и пишет `pruned_features` в DWH."""
        if split.tr_dlgs and len(split.tr_dlgs) * 2 == len(split.y_tr):
            store, X_tr_raw, X_te_raw, raw_names = DatasetSplitter.fit_full_feature_store(
                split.tr_dlgs, split.te_dlgs, split.y_tr, split.groups_tr, n_splits=n_cv_splits
            )
        else:
            store = None
            X_tr_raw, X_te_raw, raw_names = split.X_tr, split.X_te, list(bundle.feature_names)

        pruner = FeaturePruner(corr_threshold=self.corr_threshold)
        X_tr = pruner.fit_transform(X_tr_raw, feature_names=raw_names)
        X_te = pruner.transform(X_te_raw) if len(X_te_raw) > 0 else np.empty((0, X_tr.shape[1]), dtype=np.float32)
        active_features = pruner.get_feature_names_out()

        if store is not None:
            store.set_active_features(active_features)
            feature_store_state = store.to_state_dict()
            FeaturePruner.save_pruned_snapshot(
                dwh=self.dwh,
                X_tr=X_tr,
                X_te=X_te,
                dialog_ids_tr=split.dialog_ids_tr,
                participant_indices_tr=split.participant_indices_tr,
                dialog_ids_te=split.dialog_ids_te,
                participant_indices_te=split.participant_indices_te,
                feature_names=active_features,
            )
        else:
            feature_store_state = {
                "version": 1,
                "active_features": list(active_features),
                "bayesian": None,
                "tfidf": None,
                "char_tfidf": None,
            }
        return X_tr, X_te, active_features, feature_store_state

    def retrain_and_evaluate(self) -> RetrainEvaluationResult:
        """Обучение Level-1 -> Level-2 Stacker (из ClickHouse OOF), строгий ONNX holdout и промоушн."""
        logger.info(
            "Starting offline-parity adaptive retraining pipeline in '%s' (fast_mode=%s, n_trials=%d, n_estimators=%d)",
            self.experiment_name,
            self.fast_mode,
            self.n_trials,
            self.n_estimators,
        )
        champion_info = ChampionEvaluator.resolve_champion_info()
        bundle = self._load_bundle()
        split = DatasetSplitter.split_dialogue_pairs(bundle)

        n_cv_splits = min(self.n_splits, max(2, len(np.unique(split.groups_tr))))
        X_tr, X_te, active_features, feature_store_state = self._prepare_pruned_features(split, bundle, n_cv_splits)
        total_features = len(active_features)

        if champion_info is not None:
            holdout_champ_loss = ChampionEvaluator.evaluate_champion_on_holdout(X_te, split.y_te, split.te_dlgs)
            if holdout_champ_loss is None:
                raise RuntimeError("Champion exists in MLflow Registry but holdout ONNX evaluation returned None")
            champion_loss: float | None = float(holdout_champ_loss)
            champion_eval_mode = "holdout_onnx"
        else:
            benchmark_loss = self.fetch_champion_loss()
            champion_loss = float(benchmark_loss) if benchmark_loss is not None else None
            champion_eval_mode = "benchmark_override" if champion_loss is not None else "initial_bootstrap"

        logger.info(
            "Champion benchmark: version=%s mode=%s loss=%s",
            champion_info.get("version") if champion_info is not None else "none",
            champion_eval_mode,
            f"{champion_loss:.5f}" if champion_loss is not None else "none",
        )

        rankings = FeatureSelectionStage.compute_rankings(
            X_tr,
            split.y_tr,
            self.n_estimators,
            groups=split.groups_tr,
            feature_names=active_features,
            n_splits=n_cv_splits,
        )
        selected = FeatureSelectionStage.select_features(
            all_feature_names=active_features,
            rankings=rankings,
            total_features=total_features,
            feature_keep_ratio=self.feature_keep_ratio,
        )

        champion_optuna = ChampionParamResolver.fetch(config.mlflow_tracking_uri)
        # Важно: читаем важность признаков текущего @champion ДО train_and_evaluate_stacker,
        # так как при победе Challenger алиас @champion перезаписывается внутри него.
        reference_importance = self.fetch_champion_importance()
        l1 = Level1Stage(
            experiment_name=self.experiment_name,
            n_trials=self.n_trials,
            patience=self.patience,
            n_splits=n_cv_splits,
            early_stopping_rounds=self.early_stopping_rounds,
            n_estimators=self.n_estimators,
            champion_params=champion_optuna,
            dwh=self.dwh,
        ).run(
            X_tr=X_tr,
            X_te=X_te,
            y_tr=split.y_tr,
            groups_tr=split.groups_tr,
            dialog_ids_tr=split.dialog_ids_tr,
            participant_indices_tr=split.participant_indices_tr,
            dialog_ids_te=split.dialog_ids_te,
            participant_indices_te=split.participant_indices_te,
            test_ids=np.arange(len(split.y_te)),
            rankings=rankings,
            selected=selected,
        )

        stacking_stage = StackingAndPromotionStage(
            experiment_name=self.experiment_name,
            n_trials=self.n_trials,
            patience=self.patience,
            n_splits=n_cv_splits,
            n_estimators=self.n_estimators,
            dwh=self.dwh,
        )
        Z_tr, Z_te, meta_y_tr, meta_groups_tr, meta_d_ids_tr, meta_p_idx_tr, meta_is_pseudo = (
            stacking_stage.load_meta_matrices(
                l1=l1,
                y_tr=split.y_tr,
                dialog_ids_tr=split.dialog_ids_tr,
                participant_indices_tr=split.participant_indices_tr,
                groups_tr=split.groups_tr,
                is_pseudo_tr=split.is_pseudo_tr,
            )
        )

        stacker_res, metrics, trained_models = stacking_stage.train_and_evaluate_stacker(
            Z_tr=Z_tr,
            Z_te=Z_te,
            y_tr=meta_y_tr,
            groups_tr=meta_groups_tr,
            dialog_ids_tr=meta_d_ids_tr,
            participant_indices_tr=meta_p_idx_tr,
            dialog_ids_te=split.dialog_ids_te,
            participant_indices_te=split.participant_indices_te,
            y_te=split.y_te,
            is_pseudo_tr=meta_is_pseudo,
            test_ids=np.arange(len(split.y_te)),
            meta_feature_names=["logit_catboost", "logit_lightgbm", "logit_xgboost"],
            l1=l1,
            selected=selected,
            champion_loss=champion_loss,
            champion_version=champion_info["version"] if champion_info is not None else None,
            champion_run_id=champion_info["run_id"] if champion_info is not None else None,
            champion_eval_mode=champion_eval_mode,
            initial_params=champion_optuna.get("stacker"),
            feature_store_state=feature_store_state,
        )

        diff_data, _ = stacking_stage.analyze_drift(
            l1=l1,
            reference_importance=reference_importance,
            model_probs={"catboost": l1.cb_test, "lightgbm": l1.lgb_test, "xgboost": l1.xgb_test},
        )

        return RetrainEvaluationResult(
            metrics=metrics,
            diff_data=diff_data,
            selected_features=list(selected.selected_by_model.get("catboost", [])),
            selected_features_by_model=selected.selected_by_model,
            selected_feature_indices_by_model={
                "catboost": list(selected.cb_feature_indices),
                "lightgbm": list(selected.lgb_feature_indices),
                "xgboost": list(selected.xgb_feature_indices),
            },
            total_features=total_features,
            y_true=meta_y_tr.tolist() if hasattr(meta_y_tr, "tolist") else list(meta_y_tr),
            y_pred_proba=(
                stacker_res.oof_calibrated.tolist()
                if hasattr(stacker_res.oof_calibrated, "tolist")
                else list(stacker_res.oof_calibrated)
            ),
            models=trained_models,
            run_id=stacker_res.run_id,
            level1_runs=l1.level1_runs,
        )


__all__ = ["RetrainingGatekeeper"]
