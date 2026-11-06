from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from project.core.logging import ProjectLogger
from project.models.callbacks import OptunaEarlyStoppingCallback
from project.models.classical.configs.base_search_config import BaseSearchConfig
from project.models.postprocessors.pairwise import PairwisePostprocessor
from project.tracking.experiment_tracker.loggers.features import FeatureImportanceExtractor

logger = ProjectLogger.get_logger(__name__)


class CrossValidationRunner:
    """Исполнитель кросс-валидации, калибровки и финального обучения классических моделей."""

    @staticmethod
    def _evaluate_trial(
        trial: optuna.Trial,
        build_model_fn: Callable[[dict[str, Any]], Any],
        fit_fold_fn: Callable[..., None],
        search_cfg: BaseSearchConfig,
        X_mat: np.ndarray,
        y_arr: np.ndarray,
        groups_arr: np.ndarray,
        feature_ranking_indices: np.ndarray | None,
        n_splits: int,
    ) -> float:
        sampled_params = search_cfg.sample_parameters(trial)
        trial_params = {k: v for k, v in sampled_params.items() if not k.startswith("top_")}

        if feature_ranking_indices is not None and "top_n_features" in sampled_params:
            sel_idx = feature_ranking_indices[: sampled_params["top_n_features"]]
            X_trial = X_mat[:, sel_idx]
        else:
            X_trial = X_mat

        fold_scores: list[float] = []
        cv = GroupKFold(n_splits=n_splits)
        for trn_idx, val_idx in cv.split(X_trial, y_arr, groups_arr):
            clf = build_model_fn(trial_params)
            fit_fold_fn(
                model=clf,
                X_tr=X_trial[trn_idx],
                y_tr=y_arr[trn_idx],
                X_va=X_trial[val_idx],
                y_va=y_arr[val_idx],
            )
            preds = clf.predict_proba(X_trial[val_idx])[:, 1]
            fold_scores.append(float(log_loss(y_arr[val_idx], preds)))
        return float(np.mean(fold_scores))

    @classmethod
    def tune_hyperparameters(
        cls,
        build_model_fn: Callable[[dict[str, Any]], Any],
        fit_fold_fn: Callable[..., None],
        search_cfg: BaseSearchConfig,
        X_mat: np.ndarray,
        y_arr: np.ndarray,
        groups_arr: np.ndarray,
        feature_ranking_indices: np.ndarray | None,
        n_splits: int,
        n_trials: int,
        patience: int,
        random_state: int,
        optuna_callback: Any,
        initial_params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float, int | None, optuna.Study]:
        """Оптимизация гиперпараметров и подбор числа признаков через Optuna."""
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        if initial_params:
            try:
                valid_init = {k: v for k, v in initial_params.items() if not k.startswith("top_")}
                study.enqueue_trial(valid_init)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to enqueue champion initial params into Optuna study: %s", exc)

        optuna_early_stopping = OptunaEarlyStoppingCallback(patience=patience)
        study.optimize(
            lambda trial: cls._evaluate_trial(
                trial=trial,
                build_model_fn=build_model_fn,
                fit_fold_fn=fit_fold_fn,
                search_cfg=search_cfg,
                X_mat=X_mat,
                y_arr=y_arr,
                groups_arr=groups_arr,
                feature_ranking_indices=feature_ranking_indices,
                n_splits=n_splits,
            ),
            n_trials=n_trials,
            callbacks=[optuna_early_stopping, optuna_callback],
        )

        raw_best = study.best_params.copy()
        optimal_k = int(raw_best["top_n_features"]) if "top_n_features" in raw_best else None
        clean_best_params = {k: v for k, v in raw_best.items() if k != "top_n_features"}

        final_best_params = search_cfg.get_fixed_parameters().copy()
        final_best_params.update(clean_best_params)
        return final_best_params, float(study.best_value), optimal_k, study

    @staticmethod
    def execute_kfold(
        build_model_fn: Callable[[dict[str, Any]], Any],
        fit_fold_fn: Callable[..., None],
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        groups: np.ndarray,
        X_te: np.ndarray,
        n_splits: int,
        use_params: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, list[Any], np.ndarray]:
        """Исполнение GroupKFold кросс-валидации и сбор OOF/Test вероятностей и номеров фолдов."""
        cv = GroupKFold(n_splits=n_splits)
        oof_raw = np.zeros(len(y_tr), dtype=np.float32)
        test_preds = np.zeros(len(X_te), dtype=np.float32)
        fold_indices = np.zeros(len(y_tr), dtype=np.int32)
        trained_models: list[Any] = []

        for fold_idx, (trn_idx, val_idx) in enumerate(cv.split(X_tr, y_tr, groups)):
            clf = build_model_fn(use_params)
            fit_fold_fn(
                model=clf,
                X_tr=X_tr[trn_idx],
                y_tr=y_tr[trn_idx],
                X_va=X_tr[val_idx],
                y_va=y_tr[val_idx],
            )
            oof_raw[val_idx] = clf.predict_proba(X_tr[val_idx])[:, 1]
            fold_indices[val_idx] = fold_idx
            test_preds += clf.predict_proba(X_te)[:, 1] / float(n_splits)
            trained_models.append(clf)

        return oof_raw, test_preds, trained_models, fold_indices

    @staticmethod
    def compute_cv_feature_importances(
        build_model_fn: Callable[[dict[str, Any]], Any],
        fit_fold_fn: Callable[..., None],
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        groups: np.ndarray,
        feature_names: list[str],
        params: dict[str, Any],
        n_splits: int = 5,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Усредненные важности признаков по 5-фолдовой GroupKFold кросс-валидации (без утечки)."""
        cv = GroupKFold(n_splits=n_splits)
        imp_accum = np.zeros(X_tr.shape[1], dtype=np.float64)
        for trn_idx, val_idx in cv.split(X_tr, y_tr, groups):
            clf = build_model_fn(params)
            fit_fold_fn(
                model=clf,
                X_tr=X_tr[trn_idx],
                y_tr=y_tr[trn_idx],
                X_va=X_tr[val_idx],
                y_va=y_tr[val_idx],
            )
            imp = FeatureImportanceExtractor.extract(clf)
            if imp is not None and len(imp) == X_tr.shape[1]:
                imp_accum += np.asarray(imp, dtype=np.float64)

        imp_mean = imp_accum / float(max(1, n_splits))
        ranking_idx = np.argsort(imp_mean)[::-1]
        imp_dict = {
            str(feature_names[idx]): round(float(imp_mean[idx]), 5) for idx in ranking_idx if idx < len(feature_names)
        }
        return ranking_idx, imp_dict

    @staticmethod
    def evaluate_and_calibrate(
        oof_raw: np.ndarray,
        test_preds: np.ndarray,
        y_tr: np.ndarray,
        dialog_ids_train: np.ndarray,
        participant_indices_train: np.ndarray,
        dialog_ids_test: np.ndarray,
        participant_indices_test: np.ndarray,
        pairwise_eps: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Парная пост-обработка вероятностей и расчет стандартных метрик классификации."""
        pairwise = PairwisePostprocessor(max_pair_sum=1.0, eps=pairwise_eps)
        oof_calibrated = pairwise.transform(
            X=oof_raw,
            dialog_ids=dialog_ids_train,
            participant_indices=participant_indices_train,
        )
        test_calibrated = pairwise.transform(
            X=test_preds,
            dialog_ids=dialog_ids_test,
            participant_indices=participant_indices_test,
        )

        loss_raw = float(log_loss(y_tr, oof_raw))
        loss_calib = float(log_loss(y_tr, oof_calibrated))
        metrics = {
            "val_logloss": loss_calib,
            "val_logloss_raw": loss_raw,
            "val_roc_auc": float(roc_auc_score(y_tr, oof_calibrated)),
            "val_accuracy": float(accuracy_score(y_tr, (oof_calibrated >= 0.5).astype(int))),
            "val_f1": float(f1_score(y_tr, (oof_calibrated >= 0.5).astype(int))),
        }
        return oof_calibrated, test_calibrated, metrics

    @staticmethod
    def _extract_best_iteration(model: Any) -> int | None:
        lgb_iter = getattr(model, "best_iteration_", None)
        if lgb_iter is not None and int(lgb_iter) > 0:
            return int(lgb_iter)
        xgb_iter = getattr(model, "best_iteration", None)
        if xgb_iter is not None and int(xgb_iter) >= 0:
            return int(xgb_iter) + 1
        if hasattr(model, "get_best_iteration"):
            cb_iter = model.get_best_iteration()
            if cb_iter is not None and int(cb_iter) >= 0:
                return int(cb_iter) + 1
        return None

    @classmethod
    def refit_full_model(
        cls,
        build_model_fn: Callable[[dict[str, Any]], Any],
        trained_models: list[Any],
        use_params: dict[str, Any],
        model_family: str,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
    ) -> tuple[Any, int | None]:
        """Обучение финального классификатора на 100% данных с усредненным числом итераций."""
        best_iters = [it for m in trained_models if (it := cls._extract_best_iteration(m)) is not None]
        full_params = use_params.copy()
        avg_iter: int | None = None
        if best_iters:
            avg_iter = max(10, round(float(np.mean(best_iters))))
            if "iterations" in full_params or model_family == "CatBoostClassifier":
                full_params["iterations"] = avg_iter
            if "n_estimators" in full_params or model_family in ("LGBMClassifier", "XGBClassifier"):
                full_params["n_estimators"] = avg_iter

        full_params.pop("early_stopping_rounds", None)
        full_clf = build_model_fn(full_params)
        if hasattr(full_clf, "early_stopping_rounds"):
            full_clf.early_stopping_rounds = None

        full_clf.fit(X_tr, y_tr)
        return full_clf, avg_iter
