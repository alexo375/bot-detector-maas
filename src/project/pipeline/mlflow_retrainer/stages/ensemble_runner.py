from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import log_loss

from project.core.logging import ProjectLogger
from project.models.blueprints import (
    CATBOOST_BLUEPRINT,
    LIGHTGBM_BLUEPRINT,
    XGBOOST_BLUEPRINT,
)
from project.models.classical import (
    CatBoostTrainer,
    ClassicalTrainerConfig,
    LightGBMTrainer,
    XGBoostTrainer,
)
from project.models.classical.configs.catboost_search_config import CatBoostSearchConfig
from project.models.classical.configs.lightgbm_search_config import LightGBMSearchConfig
from project.models.classical.configs.xgboost_search_config import XGBoostSearchConfig
from project.pipeline.mlflow_retrainer.components.champion_evaluator import ChampionEvaluator
from project.pipeline.mlflow_retrainer.stages.feature_selection import SelectedFeatures
from project.tracking.experiment_tracker.schemas import ExperimentEnvironment

logger = ProjectLogger.get_logger(__name__)


@dataclass
class Level1Results:
    """Результаты обучения, full-refit моделей и OOF/Test скоров первого уровня (Level-1 GBDT)."""

    level1_runs: dict[str, str]
    cb_oof: np.ndarray
    cb_test: np.ndarray
    cb_loss: float
    lgb_oof: np.ndarray
    lgb_test: np.ndarray
    lgb_loss: float
    xgb_oof: np.ndarray
    xgb_test: np.ndarray
    xgb_loss: float
    cb_full_model: Any = None
    lgb_full_model: Any = None
    xgb_full_model: Any = None
    cb_feature_importances: dict[str, float] = field(default_factory=dict)


class Level1Stage:
    """Этап обучения бустинговых моделей первого уровня (Level-1 GBDT) с Optuna и ClickHouse OOF."""

    def __init__(
        self,
        experiment_name: str,
        n_trials: int,
        patience: int,
        n_splits: int,
        early_stopping_rounds: int,
        n_estimators: int,
        champion_params: dict[str, Any] | None = None,
        dwh: Any = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.patience = patience
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.n_estimators = n_estimators
        self.champion_params = champion_params or {}
        self.dwh = dwh

    def _fit_single_gbdt(
        self,
        *,
        model_key: str,
        trainer: Any,
        blueprint: Any,
        iter_key: str,
        default_k: int,
        all_feature_names: list[str],
        ranking_indices: np.ndarray,
        X_tr: np.ndarray,
        X_te: np.ndarray,
        y_tr: np.ndarray,
        groups_tr: np.ndarray,
        dialog_ids_tr: np.ndarray,
        participant_indices_tr: np.ndarray,
        dialog_ids_te: np.ndarray,
        participant_indices_te: np.ndarray,
        test_ids: np.ndarray,
    ) -> tuple[Any, dict[str, Any], list[int]]:
        """Строгий тюнинг в Optuna, 5-fold CV и full-refit по среднему `best_iteration` (без фолбэков)."""
        if self.n_trials > 0:
            trainer.tune(
                X=X_tr,
                y=y_tr,
                groups=groups_tr,
                feature_ranking_indices=ranking_indices,
                feature_names=all_feature_names,
            )

        if trainer.optimal_features_count is None:
            trainer.optimal_features_count = default_k

        fit_params = {**blueprint.default_params, **(trainer.best_params or {}), iter_key: self.n_estimators}
        res = trainer.fit(
            X_train=X_tr,
            y_train=y_tr,
            groups=groups_tr,
            X_test=X_te,
            dialog_ids_train=dialog_ids_tr,
            participant_indices_train=participant_indices_tr,
            dialog_ids_test=dialog_ids_te,
            participant_indices_test=participant_indices_te,
            test_ids=test_ids,
            feature_names=all_feature_names,
            feature_ranking_indices=ranking_indices,
            params=fit_params,
            refit_full=True,
        )
        final_indices = res.selected_feature_indices or [int(i) for i in ranking_indices[: len(res.selected_features)]]
        return res, dict(res.best_params or fit_params), final_indices

    def _build_model_specs(self, total_feats: int) -> tuple[tuple[Any, ...], ...]:
        min_feats = min(5, total_feats)
        return (
            (
                "catboost",
                "CatBoost",
                CatBoostTrainer,
                CatBoostSearchConfig(
                    iterations=self.n_estimators,
                    tune_feature_count=True,
                    min_features=min_feats,
                    max_features=total_feats,
                ),
                CATBOOST_BLUEPRINT,
                "iterations",
                "k_cb",
                "cb_feature_indices",
            ),
            (
                "lightgbm",
                "LightGBM",
                LightGBMTrainer,
                LightGBMSearchConfig(
                    n_estimators=self.n_estimators,
                    tune_feature_count=True,
                    min_features=min_feats,
                    max_features=total_feats,
                ),
                LIGHTGBM_BLUEPRINT,
                "n_estimators",
                "m_lgb",
                "lgb_feature_indices",
            ),
            (
                "xgboost",
                "XGBoost",
                XGBoostTrainer,
                XGBoostSearchConfig(
                    n_estimators=self.n_estimators,
                    tune_feature_count=True,
                    min_features=min_feats,
                    max_features=total_feats,
                ),
                XGBOOST_BLUEPRINT,
                "n_estimators",
                "p_xgb",
                "xgb_feature_indices",
            ),
        )

    def _persist_l1_predictions(
        self,
        bp_run_name: str,
        res: Any,
        y_tr: np.ndarray,
        dialog_ids_tr: np.ndarray,
        participant_indices_tr: np.ndarray,
        dialog_ids_te: np.ndarray,
        participant_indices_te: np.ndarray,
    ) -> None:
        ChampionEvaluator.save_predictions_to_dwh(
            dwh=self.dwh,
            run_id=res.run_id,
            run_name=bp_run_name,
            dataset_split="train_oof",
            dialog_ids=dialog_ids_tr,
            participant_indices=participant_indices_tr,
            prob_bot=res.oof_raw,
            prob_bot_calibrated=res.oof_calibrated,
            is_bot=y_tr,
            fold_indices=res.fold_indices,
        )
        ChampionEvaluator.save_predictions_to_dwh(
            dwh=self.dwh,
            run_id=res.run_id,
            run_name=bp_run_name,
            dataset_split="test",
            dialog_ids=dialog_ids_te,
            participant_indices=participant_indices_te,
            prob_bot=res.test_raw,
            prob_bot_calibrated=res.test_calibrated,
        )

    def run(
        self,
        X_tr: np.ndarray,
        X_te: np.ndarray,
        y_tr: np.ndarray,
        groups_tr: np.ndarray,
        dialog_ids_tr: np.ndarray,
        participant_indices_tr: np.ndarray,
        dialog_ids_te: np.ndarray,
        participant_indices_te: np.ndarray,
        test_ids: np.ndarray,
        rankings: dict[str, np.ndarray],
        selected: SelectedFeatures,
    ) -> Level1Results:
        """Обучение CatBoost, LightGBM и XGBoost с Optuna отбором и сохранением OOF в ClickHouse."""
        level1_runs: dict[str, str] = {}
        all_names = selected.all_feature_names or [f"f_{i}" for i in range(X_tr.shape[1])]
        outputs: dict[str, tuple[np.ndarray, np.ndarray, float, Any, dict[str, float]]] = {}

        for key, label, trainer_cls, search_cfg, bp, iter_key, k_attr, idx_attr in self._build_model_specs(
            len(all_names)
        ):
            logger.info("Tuning & fitting Level-1 %sTrainer (candidates: %d)...", label, len(all_names))
            cfg = ClassicalTrainerConfig(
                run_name=bp.run_name,
                experiment_name=self.experiment_name,
                description=bp.description,
                n_trials=self.n_trials,
                patience=self.patience,
                n_splits=self.n_splits,
                early_stopping_rounds=self.early_stopping_rounds,
                dataset_name="train_tabular_features",
                environment=ExperimentEnvironment.ONLINE,
                initial_params=self.champion_params.get(key),
            )
            trainer = trainer_cls(config=cfg, search_config=search_cfg)
            res, _, indices = self._fit_single_gbdt(
                model_key=label,
                trainer=trainer,
                blueprint=bp,
                iter_key=iter_key,
                default_k=getattr(selected, k_attr),
                all_feature_names=all_names,
                ranking_indices=rankings[key],
                X_tr=X_tr,
                X_te=X_te,
                y_tr=y_tr,
                groups_tr=groups_tr,
                dialog_ids_tr=dialog_ids_tr,
                participant_indices_tr=participant_indices_tr,
                dialog_ids_te=dialog_ids_te,
                participant_indices_te=participant_indices_te,
                test_ids=test_ids,
            )
            selected_names = list(res.selected_features)
            setattr(selected, k_attr, len(indices))
            setattr(selected, idx_attr, indices)
            selected.selected_by_model[key] = selected_names
            level1_runs[bp.run_name] = res.run_id
            self._persist_l1_predictions(
                bp.run_name, res, y_tr, dialog_ids_tr, participant_indices_tr, dialog_ids_te, participant_indices_te
            )
            oof = np.asarray(res.oof_raw, dtype=np.float64)
            test = np.asarray(res.test_raw, dtype=np.float64)
            loss = float(res.metrics.get("val_logloss", log_loss(y_tr, res.oof_calibrated)))
            outputs[key] = (oof, test, loss, res.full_model, dict(res.feature_importances or {}))

        cb_oof, cb_test, cb_loss, cb_full, cb_imp = outputs["catboost"]
        lgb_oof, lgb_test, lgb_loss, lgb_full, _ = outputs["lightgbm"]
        xgb_oof, xgb_test, xgb_loss, xgb_full, _ = outputs["xgboost"]
        return Level1Results(
            level1_runs=level1_runs,
            cb_oof=cb_oof,
            cb_test=cb_test,
            cb_loss=cb_loss,
            lgb_oof=lgb_oof,
            lgb_test=lgb_test,
            lgb_loss=lgb_loss,
            xgb_oof=xgb_oof,
            xgb_test=xgb_test,
            xgb_loss=xgb_loss,
            cb_full_model=cb_full,
            lgb_full_model=lgb_full,
            xgb_full_model=xgb_full,
            cb_feature_importances=cb_imp,
        )
