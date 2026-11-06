import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from project.core.logging import ProjectLogger
from project.models.classical.configs.base_search_config import BaseSearchConfig
from project.models.classical.configs.stacking_search_config import StackingSearchConfig
from project.models.classical.schemas.classical_trainer_config import ClassicalTrainerConfig
from project.models.classical.schemas.training_result import TrainingResult
from project.models.classical.schemas.tuning_result import TuningResult
from project.models.classical.stacking import TwoLevelStackingClassifier
from project.models.classical.trainers.base_classical_trainer import BaseClassicalTrainer
from project.models.classical.trainers.stacking_tree import EnsembleTreeBuilder
from project.models.postprocessors.pairwise import PairwisePostprocessor
from project.tracking.experiment_tracker.builder import ExperimentTrackerBuilder
from project.tracking.experiment_tracker.tracker import ExperimentTracker

logger = ProjectLogger.get_logger(__name__)


class TwoLevelStackingTrainer(BaseClassicalTrainer):
    """Специализированный тренер для ансамбля TwoLevelStackingClassifier."""

    def __init__(
        self,
        config: ClassicalTrainerConfig | None = None,
        search_config: BaseSearchConfig | None = None,
        base_models: list[tuple[str, Any]] | None = None,
        level1_runs: dict[str, str] | None = None,
        use_logits: bool = True,
    ) -> None:
        super().__init__(config=config, search_config=search_config or StackingSearchConfig())
        self.base_models = base_models
        self.level1_runs = level1_runs or {}
        self.use_logits = use_logits

    @property
    def model_family(self) -> str:
        """Идентификатор семейства модели."""
        return "TwoLevelStackingClassifier"

    def default_search_config(self) -> StackingSearchConfig:
        """Конфигурация поиска гиперпараметров по умолчанию."""
        return StackingSearchConfig()

    def build_model(self, params: dict[str, Any]) -> LogisticRegression:
        """Инстанцирование мета-классификатора логистической регрессии."""
        clean_params = {k: v for k, v in params.items() if not k.startswith("top_")}
        for k, v in self.search_config.get_fixed_parameters().items():
            clean_params.setdefault(k, v)
        clean_params.setdefault("solver", "lbfgs")
        clean_params.setdefault("max_iter", 1000)
        return LogisticRegression(
            **clean_params,
            random_state=self.config.random_state,
        )

    def _prepare_meta_matrix(
        self,
        X_tr_mat: np.ndarray,
        X_te_mat: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Согласованно переводит обучающую и тестовую матрицы в пространство логитов или вероятностей."""
        tr_arr = np.asarray(X_tr_mat, dtype=np.float64)
        te_arr = np.asarray(X_te_mat, dtype=np.float64) if X_te_mat is not None else None
        if not self.use_logits:
            return tr_arr, te_arr

        tr_is_logits = bool(np.any(tr_arr < 0.0) or np.any(tr_arr > 1.0))
        te_is_logits = te_arr is not None and len(te_arr) > 0 and bool(np.any(te_arr < 0.0) or np.any(te_arr > 1.0))
        if tr_is_logits or te_is_logits:
            return tr_arr, te_arr

        eps = 1e-5
        tr_clip = np.clip(tr_arr, eps, 1.0 - eps)
        tr_out = np.log(tr_clip / (1.0 - tr_clip))
        if te_arr is None:
            return tr_out, None
        te_clip = np.clip(te_arr, eps, 1.0 - eps)
        return tr_out, np.log(te_clip / (1.0 - te_clip))

    def tune(
        self,
        X: pd.DataFrame | pl.DataFrame | np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        search_config: BaseSearchConfig | None = None,
        feature_ranking_indices: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> TuningResult:
        """Подбор гиперпараметров мета-модели в том же пространстве логитов/вероятностей, что и fit()."""
        if feature_names is None and isinstance(X, (pd.DataFrame, pl.DataFrame)):
            feature_names = [str(c) for c in X.columns]
        X_meta, _ = self._prepare_meta_matrix(self._to_numpy(X, dtype=np.float64))
        return super().tune(
            X=X_meta,
            y=y,
            groups=groups,
            search_config=search_config,
            feature_ranking_indices=feature_ranking_indices,
            feature_names=feature_names,
        )

    def fit_fold(
        self,
        model: LogisticRegression,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
    ) -> None:
        """Обучение мета-модели на одном фолде."""
        del X_va, y_va
        model.fit(X_tr, y_tr)

    def _ensure_stacker_tracker(
        self,
        full_fit_df: pl.DataFrame,
        y_tr_arr: np.ndarray,
        final_feature_names: list[str],
        use_params: dict[str, Any],
    ) -> tuple[ExperimentTracker, bool]:
        had_existing = self._tracker is not None
        if self._tracker is None:
            self._tracker = (
                ExperimentTrackerBuilder(run_name=self.config.run_name, experiment_name=self.config.experiment_name)
                .set_environment(self.config.environment)
                .set_description(self.config.description or f"Обучение ансамбля {self.model_family}")
                .set_dataset(
                    data=full_fit_df, name=self.config.dataset_name, targets=y_tr_arr, feature_names=final_feature_names
                )
                .set_params(
                    model_family=self.model_family,
                    optimizer="Optuna" if self.best_params is not None else "Manual",
                    n_trials=self.config.n_trials,
                    patience=self.config.patience,
                    **use_params,
                )
                .require_feature_tree(True)
                .defer_metrics()
                .build()
            )
            self._tracker.__enter__()
        elif not self._tracker._has_logged_dataset:
            self._tracker.log_dataset(
                data=full_fit_df, name=self.config.dataset_name, targets=y_tr_arr, feature_names=final_feature_names
            )
        return self._tracker, had_existing

    def _run_meta_cv(
        self,
        X_tr_meta: np.ndarray,
        X_te_meta: np.ndarray,
        y_tr_arr: np.ndarray,
        groups_arr: np.ndarray,
        use_params: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, list[LogisticRegression], np.ndarray]:
        cv = GroupKFold(n_splits=self.config.n_splits)
        oof_raw = np.zeros(len(y_tr_arr), dtype=np.float64)
        test_preds_folds = np.zeros(len(X_te_meta), dtype=np.float64)
        fold_indices = np.zeros(len(y_tr_arr), dtype=np.int32)
        trained_models: list[LogisticRegression] = []

        for fold_idx, (trn_idx, val_idx) in enumerate(cv.split(X_tr_meta, y_tr_arr, groups_arr)):
            clf = self.build_model(use_params)
            self.fit_fold(clf, X_tr_meta[trn_idx], y_tr_arr[trn_idx], X_tr_meta[val_idx], y_tr_arr[val_idx])
            oof_raw[val_idx] = clf.predict_proba(X_tr_meta[val_idx])[:, 1]
            fold_indices[val_idx] = fold_idx
            test_preds_folds += clf.predict_proba(X_te_meta)[:, 1] / float(self.config.n_splits)
            trained_models.append(clf)
        return oof_raw, test_preds_folds, trained_models, fold_indices

    def _build_final_stacker(
        self,
        X_tr_mat: np.ndarray,
        X_tr_meta: np.ndarray,
        y_tr_arr: np.ndarray,
        oof_calibrated: np.ndarray,
        final_feature_names: list[str],
        use_params: dict[str, Any],
        metrics: dict[str, float],
    ) -> tuple[TwoLevelStackingClassifier, dict[str, float]]:
        full_meta_clf = self.build_model(use_params)
        full_meta_clf.fit(X_tr_meta, y_tr_arr)
        final_stacker = TwoLevelStackingClassifier(
            base_models=self.base_models,
            meta_model=full_meta_clf,
            n_splits=self.config.n_splits,
            random_state=self.config.random_state,
            use_logits=self.use_logits,
        )
        final_stacker.meta_model_ = full_meta_clf
        final_stacker.classes_ = np.array([0, 1])
        final_stacker.feature_names_in_ = list(final_feature_names)
        final_stacker._base_names = list(final_feature_names)
        final_stacker.oof_predictions_ = oof_calibrated
        final_stacker.oof_metrics_ = EnsembleTreeBuilder.compute_meta_oof_metrics(
            X_tr_mat, y_tr_arr, final_feature_names, metrics
        )
        coef = full_meta_clf.coef_[0] if getattr(full_meta_clf.coef_, "ndim", 1) > 1 else full_meta_clf.coef_
        meta_imp = {
            str(final_feature_names[idx]): round(float(abs(val)), 5)
            for idx, val in enumerate(coef)
            if idx < len(final_feature_names)
        }
        return final_stacker, meta_imp

    def _finalize_and_log(
        self,
        tracker: ExperimentTracker,
        run_id: str,
        X_tr_mat: np.ndarray,
        X_tr_meta: np.ndarray,
        y_tr_arr: np.ndarray,
        oof_raw: np.ndarray,
        test_preds_folds: np.ndarray,
        trained_models: list[LogisticRegression],
        fold_indices: np.ndarray,
        feature_names: list[str],
        use_params: dict[str, Any],
        dialog_ids_train: np.ndarray,
        participant_indices_train: np.ndarray,
        dialog_ids_test: np.ndarray,
        participant_indices_test: np.ndarray,
        test_ids: np.ndarray,
        post_fit_callback: Callable[[ExperimentTracker, TrainingResult], None] | None,
    ) -> TrainingResult:
        pairwise = PairwisePostprocessor(max_pair_sum=1.0, eps=self.config.pairwise_eps)
        oof_calibrated = pairwise.transform(
            X=oof_raw, dialog_ids=dialog_ids_train, participant_indices=participant_indices_train
        )
        test_calibrated = pairwise.transform(
            X=test_preds_folds, dialog_ids=dialog_ids_test, participant_indices=participant_indices_test
        )
        metrics = {
            "val_logloss": float(log_loss(y_tr_arr, oof_calibrated)),
            "val_logloss_raw": float(log_loss(y_tr_arr, oof_raw)),
            "val_roc_auc": float(roc_auc_score(y_tr_arr, oof_calibrated)),
            "val_accuracy": float(accuracy_score(y_tr_arr, (oof_calibrated >= 0.5).astype(int))),
            "val_f1": float(f1_score(y_tr_arr, (oof_calibrated >= 0.5).astype(int))),
        }
        tracker.log_metrics(extra_metrics=metrics)
        final_stacker, meta_imp = self._build_final_stacker(
            X_tr_mat, X_tr_meta, y_tr_arr, oof_calibrated, feature_names, use_params, metrics
        )
        ensemble_tree = EnsembleTreeBuilder.build(
            final_stacker, level1_runs=self.level1_runs, tracking_uri=getattr(tracker, "tracking_uri", None)
        )
        tracker.log_feature_tree(ensemble_tree)
        tracker.log_all_graphics(y_true=y_tr_arr, y_pred_proba=oof_calibrated)
        submission_df = pl.DataFrame({"ID": test_ids, "is_bot": test_calibrated})
        tracker.log_submission(submission_df, filename="submission.csv")
        result = TrainingResult(
            oof_raw=oof_raw,
            oof_calibrated=oof_calibrated,
            test_raw=test_preds_folds,
            test_calibrated=test_calibrated,
            metrics=metrics,
            best_params=use_params,
            models=trained_models,
            selected_features=feature_names,
            submission_df=submission_df,
            run_id=run_id,
            fold_indices=fold_indices,
            full_model=final_stacker,
            feature_importances=meta_imp,
        )
        if post_fit_callback is not None:
            post_fit_callback(tracker, result)
        elif self.config.register_as:
            logger.info("Регистрация модели ансамбля в MLflow Model Registry: %s", self.config.register_as)
            tracker.log_model_instance(
                final_stacker,
                artifact_path="model",
                register_as=self.config.register_as,
                metric_for_promotion="val_logloss",
            )
        return result

    def fit(
        self,
        X_train: pd.DataFrame | pl.DataFrame | np.ndarray,
        y_train: np.ndarray,
        groups: np.ndarray,
        X_test: pd.DataFrame | pl.DataFrame | np.ndarray,
        dialog_ids_train: np.ndarray,
        participant_indices_train: np.ndarray,
        dialog_ids_test: np.ndarray,
        participant_indices_test: np.ndarray,
        test_ids: np.ndarray,
        feature_names: list[str] | None = None,
        feature_ranking_indices: np.ndarray | None = None,
        params: dict[str, Any] | None = None,
        refit_full: bool = True,
        selected_feature_indices: list[int] | None = None,
        post_fit_callback: Callable[[ExperimentTracker, TrainingResult], None] | None = None,
    ) -> TrainingResult:
        """Обучает ансамбль стекинга по фолдам, калибрует предсказания и логирует артефакты."""
        del feature_ranking_indices, refit_full, selected_feature_indices
        X_tr_mat, X_te_mat = self._to_numpy(X_train, dtype=np.float64), self._to_numpy(X_test, dtype=np.float64)
        y_tr_arr, groups_arr = np.asarray(y_train, dtype=int), np.asarray(groups)
        X_tr_meta, X_te_meta = self._prepare_meta_matrix(X_tr_mat, X_te_mat)
        if X_te_meta is None:
            raise ValueError("X_test must not be None")
        use_params = (params or self.best_params or {}).copy()
        for k, v in self.search_config.get_fixed_parameters().items():
            use_params.setdefault(k, v)
        if feature_names is None:
            feature_names = (
                [str(c) for c in X_train.columns]
                if isinstance(X_train, (pd.DataFrame, pl.DataFrame))
                else [f"meta_f_{i}" for i in range(X_tr_mat.shape[1])]
            )
        full_fit_df = pl.DataFrame({col: X_tr_meta[:, i] for i, col in enumerate(feature_names)})
        tracker, had_existing = self._ensure_stacker_tracker(full_fit_df, y_tr_arr, feature_names, use_params)
        try:
            run_id = tracker.active_run.info.run_id if tracker.active_run else ""
            if had_existing and use_params:
                tracker.log_params(**use_params)
            oof_raw, test_preds_folds, trained_models, fold_indices = self._run_meta_cv(
                X_tr_meta, X_te_meta, y_tr_arr, groups_arr, use_params
            )
            result = self._finalize_and_log(
                tracker,
                run_id,
                X_tr_mat,
                X_tr_meta,
                y_tr_arr,
                oof_raw,
                test_preds_folds,
                trained_models,
                fold_indices,
                feature_names,
                use_params,
                dialog_ids_train,
                participant_indices_train,
                dialog_ids_test,
                participant_indices_test,
                test_ids,
                post_fit_callback,
            )
            self.close_tracker()
        except Exception:
            self.close_tracker(exc_info=sys.exc_info())
            raise
        return result
