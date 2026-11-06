import sys
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from project.core.logging import ProjectLogger
from project.models.classical.configs.base_search_config import BaseSearchConfig
from project.models.classical.schemas.classical_trainer_config import ClassicalTrainerConfig
from project.models.classical.schemas.training_result import TrainingResult
from project.models.classical.schemas.tuning_result import TuningResult
from project.models.classical.trainers.cross_validation import CrossValidationRunner
from project.models.classical.trainers.evaluator import TrainerArtifactManager
from project.tracking.experiment_tracker.tracker import ExperimentTracker

logger = ProjectLogger.get_logger(__name__)


class BaseClassicalTrainer(ABC):
    """Базовый класс тренера для CatBoost, LightGBM и XGBoost."""

    def __init__(
        self,
        config: ClassicalTrainerConfig | None = None,
        search_config: BaseSearchConfig | None = None,
    ) -> None:
        self.config = config or ClassicalTrainerConfig()
        self.search_config = search_config or self.default_search_config()
        self.best_params: dict[str, Any] | None = None
        self.optimal_features_count: int | None = None
        self.optimal_features_param_name: str | None = None
        self._feature_names: list[str] | None = None
        self._tracker: ExperimentTracker | None = None

    def __del__(self) -> None:
        self.close_tracker()

    def close_tracker(self, exc_info: tuple[Any, Any, Any] | None = None) -> None:
        """Безопасно завершает активный ран трекера в MLflow, если он существует."""
        if self._tracker is None:
            return
        tracker = self._tracker
        self._tracker = None
        try:
            if exc_info is not None:
                tracker.__exit__(*exc_info)
            else:
                tracker.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001  # Resilient fallback
            logger.warning("Ошибка при закрытии ExperimentTracker: %s", exc)

    @property
    @abstractmethod
    def model_family(self) -> str:
        """Название семейства модели для трекера и реестра."""

    @abstractmethod
    def default_search_config(self) -> BaseSearchConfig:
        """Возвращает конфигурацию поиска гиперпараметров по умолчанию."""

    @abstractmethod
    def build_model(self, params: dict[str, Any]) -> Any:
        """Создает экземпляр модели со специфичными флагами выполнения."""

    @abstractmethod
    def fit_fold(
        self,
        model: Any,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
    ) -> None:
        """Обучает модель на одном фолде с ранней остановкой."""

    @staticmethod
    def _to_numpy(X: Any, dtype: Any = None) -> np.ndarray:
        """Безопасная конвертация Pandas/Polars DataFrame или array-like в NumPy массив."""
        arr = X.to_numpy() if isinstance(X, (pd.DataFrame, pl.DataFrame)) else np.asarray(X)
        return arr.astype(dtype, copy=False) if dtype is not None else arr

    def tune(
        self,
        X: pd.DataFrame | pl.DataFrame | np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        search_config: BaseSearchConfig | None = None,
        feature_ranking_indices: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> TuningResult:
        """Выполняет поиск оптимальных гиперпараметров (и числа признаков) через Optuna."""
        search_cfg = search_config or self.search_config
        X_mat = self._to_numpy(X)
        y_arr = np.asarray(y).astype(int)
        groups_arr = np.asarray(groups)

        if feature_names is None and isinstance(X, (pd.DataFrame, pl.DataFrame)):
            feature_names = list(X.columns)
        self._feature_names = feature_names

        self.close_tracker()
        self._tracker = TrainerArtifactManager.initialize_tuning_tracker(
            config=self.config,
            model_family=self.model_family,
            feature_names=feature_names,
        )
        optuna_callback = self._tracker.create_optuna_callback(metric_name="val_logloss")

        try:
            final_best_params, best_score, optimal_k, study = CrossValidationRunner.tune_hyperparameters(
                build_model_fn=self.build_model,
                fit_fold_fn=self.fit_fold,
                search_cfg=search_cfg,
                X_mat=X_mat,
                y_arr=y_arr,
                groups_arr=groups_arr,
                feature_ranking_indices=feature_ranking_indices,
                n_splits=self.config.n_splits,
                n_trials=self.config.n_trials,
                patience=self.config.patience,
                random_state=self.config.random_state,
                optuna_callback=optuna_callback,
                initial_params=self.config.initial_params,
            )
        except Exception:
            self.close_tracker(exc_info=sys.exc_info())
            raise

        self.best_params = final_best_params
        self.optimal_features_count = optimal_k
        self.optimal_features_param_name = "top_n_features" if "top_n_features" in study.best_params else None
        return TuningResult(
            best_params=final_best_params,
            best_score=best_score,
            optimal_features_count=optimal_k,
            study=study,
        )

    def _ensure_fit_tracker(
        self,
        full_fit_df: pl.DataFrame,
        y_tr_arr: np.ndarray,
        final_feature_names: list[str],
    ) -> ExperimentTracker:
        if self._tracker is None:
            self._tracker = TrainerArtifactManager.initialize_tracker(
                config=self.config,
                model_family=self.model_family,
                full_fit_df=full_fit_df,
                y_tr_arr=y_tr_arr,
                final_feature_names=final_feature_names,
            )
        elif not self._tracker._has_logged_dataset:
            self._tracker.log_dataset(
                data=full_fit_df,
                name=self.config.dataset_name,
                targets=y_tr_arr,
                feature_names=final_feature_names,
            )
        return self._tracker

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
    ) -> TrainingResult:
        """Обучает ансамбль по 5 фолдам, строит full-refit модель, калибрует и логирует в MLflow."""
        X_tr_mat = self._to_numpy(X_train)
        X_te_mat = self._to_numpy(X_test)
        y_tr_arr = np.asarray(y_train).astype(int)
        groups_arr = np.asarray(groups)

        use_params = (params or self.best_params or {}).copy()
        for k, v in self.search_config.get_fixed_parameters().items():
            use_params.setdefault(k, v)

        if feature_names is None:
            if not isinstance(X_train, (pd.DataFrame, pl.DataFrame)):
                raise ValueError("feature_names must be explicitly provided when X_train is a numpy array!")
            feature_names = list(X_train.columns)

        X_tr_final, X_te_final, final_feature_names, sel_indices = TrainerArtifactManager.prepare_feature_subsets(
            x_tr_mat=X_tr_mat,
            x_te_mat=X_te_mat,
            feature_names=feature_names,
            feature_ranking_indices=feature_ranking_indices,
            optimal_features_count=self.optimal_features_count,
        )
        effective_sel_indices = sel_indices if sel_indices is not None else selected_feature_indices
        full_fit_df = pl.DataFrame({col: X_tr_final[:, i] for i, col in enumerate(final_feature_names)})
        tracker = self._ensure_fit_tracker(full_fit_df, y_tr_arr, final_feature_names)

        try:
            run_id = tracker.active_run.info.run_id if tracker.active_run else ""
            log_params_dict = use_params.copy()
            if self.optimal_features_count is not None:
                log_params_dict["optimal_top_features"] = self.optimal_features_count
                if self.optimal_features_param_name:
                    log_params_dict[f"optimal_{self.optimal_features_param_name}"] = self.optimal_features_count
            if log_params_dict:
                tracker.log_params(**log_params_dict)

            oof_raw, test_preds_folds, trained_models, fold_indices = CrossValidationRunner.execute_kfold(
                build_model_fn=self.build_model,
                fit_fold_fn=self.fit_fold,
                X_tr=X_tr_final,
                y_tr=y_tr_arr,
                groups=groups_arr,
                X_te=X_te_final,
                n_splits=self.config.n_splits,
                use_params=use_params,
            )

            oof_calibrated, test_calibrated, metrics = CrossValidationRunner.evaluate_and_calibrate(
                oof_raw=oof_raw,
                test_preds=test_preds_folds,
                y_tr=y_tr_arr,
                dialog_ids_train=dialog_ids_train,
                participant_indices_train=participant_indices_train,
                dialog_ids_test=dialog_ids_test,
                participant_indices_test=participant_indices_test,
                pairwise_eps=self.config.pairwise_eps,
            )
            tracker.log_metrics(extra_metrics=metrics)

            feat_imp = TrainerArtifactManager.log_artifacts_and_graphics(
                tracker=tracker,
                model_family=self.model_family,
                trained_models=trained_models,
                final_feature_names=final_feature_names,
                y_tr_arr=y_tr_arr,
                oof_calibrated=oof_calibrated,
                x_tr_final=X_tr_final,
            )

            submission_df = pl.DataFrame({"ID": test_ids, "is_bot": test_calibrated})
            tracker.log_submission(submission_df, filename="submission.csv")

            full_model: Any = None
            best_iteration: int | None = None
            if refit_full or self.config.register_as:
                full_model, best_iteration = CrossValidationRunner.refit_full_model(
                    build_model_fn=self.build_model,
                    trained_models=trained_models,
                    use_params=use_params,
                    model_family=self.model_family,
                    X_tr=X_tr_final,
                    y_tr=y_tr_arr,
                )
            TrainerArtifactManager.register_model_if_configured(
                tracker=tracker,
                config=self.config,
                full_model=full_model,
                model_family=self.model_family,
            )
            self.close_tracker()
        except Exception:
            self.close_tracker(exc_info=sys.exc_info())
            raise

        return TrainingResult(
            oof_raw=oof_raw,
            oof_calibrated=oof_calibrated,
            test_raw=test_preds_folds,
            test_calibrated=test_calibrated,
            metrics=metrics,
            best_params=use_params,
            models=trained_models,
            selected_features=final_feature_names,
            submission_df=submission_df,
            run_id=run_id,
            fold_indices=fold_indices,
            full_model=full_model,
            best_iteration=best_iteration,
            feature_importances=feat_imp,
            selected_feature_indices=effective_sel_indices,
        )
