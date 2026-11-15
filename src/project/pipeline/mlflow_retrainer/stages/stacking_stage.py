"""Этап обучения Level-2 мета-стекера (из ClickHouse DWH OOF), оценки Gatekeeper и промоушена."""

from typing import Any

import numpy as np
from sklearn.metrics import log_loss

from project.core.logging import ProjectLogger
from project.core.utils import LogitTransformer
from project.drift.concept_drift import ConceptDriftDetector
from project.drift.feature_importance_diff import (
    FeatureImportanceDiffAnalyzer,
    FeatureImportanceDiffPrometheusExporter,
)
from project.models.classical import (
    ClassicalTrainerConfig,
    StackingSearchConfig,
    TwoLevelStackingTrainer,
)
from project.models.classical.schemas.training_result import TrainingResult
from project.pipeline.mlflow_retrainer.components.champion_evaluator import ChampionEvaluator
from project.pipeline.mlflow_retrainer.schemas import GatekeeperDecision, GatekeeperMetrics
from project.pipeline.mlflow_retrainer.stages.ensemble_runner import Level1Results
from project.pipeline.mlflow_retrainer.stages.feature_selection import SelectedFeatures
from project.pipeline.mlflow_retrainer.stages.promotion import ModelPromotionService
from project.tracking.experiment_tracker.schemas import ExperimentEnvironment
from project.tracking.experiment_tracker.tracker import ExperimentTracker

logger = ProjectLogger.get_logger(__name__)


class StackingAndPromotionStage:
    """Этап загрузки L1 OOF из ClickHouse, обучения Meta-Stacker и оценки Champion vs Challenger."""

    def __init__(
        self,
        experiment_name: str,
        n_trials: int,
        patience: int,
        n_splits: int,
        n_estimators: int,
        dwh: Any = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.patience = patience
        self.n_splits = n_splits
        self.n_estimators = n_estimators
        self.dwh = dwh

    @staticmethod
    def _logits_from_l1(l1: Level1Results) -> tuple[np.ndarray, np.ndarray]:
        Z_tr = np.column_stack(
            [
                LogitTransformer.transform(l1.cb_oof),
                LogitTransformer.transform(l1.lgb_oof),
                LogitTransformer.transform(l1.xgb_oof),
            ]
        )
        Z_te = np.column_stack(
            [
                LogitTransformer.transform(l1.cb_test),
                LogitTransformer.transform(l1.lgb_test),
                LogitTransformer.transform(l1.xgb_test),
            ]
        )
        return Z_tr, Z_te

    def load_meta_matrices(
        self,
        *,
        l1: Level1Results,
        y_tr: np.ndarray,
        dialog_ids_tr: np.ndarray,
        participant_indices_tr: np.ndarray,
        groups_tr: np.ndarray,
        is_pseudo_tr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Строгая загрузка Level-2 матрицы логитов из ClickHouse `model_predictions` (при активном DWH)."""
        run_ids = [
            l1.level1_runs.get("catboost_detector", ""),
            l1.level1_runs.get("lightgbm_detector", ""),
            l1.level1_runs.get("xgboost_detector", ""),
        ]
        if self.dwh is not None:
            if not all(run_ids):
                raise RuntimeError(f"Level-1 run_ids incomplete for ClickHouse meta-dataset: {l1.level1_runs}")
            meta_tr_df = self.dwh.load_meta_dataset(run_ids=run_ids, dataset_split="train_oof", use_calibrated=False)
            meta_te_df = self.dwh.load_meta_dataset(run_ids=run_ids, dataset_split="test", use_calibrated=False)
            prob_cols = [c for c in meta_tr_df.columns if c.startswith("prob_")]
            if len(meta_tr_df) != len(y_tr) or len(prob_cols) != 3 or len(meta_te_df) == 0:
                raise RuntimeError(
                    f"ClickHouse meta_dataset incomplete: train={len(meta_tr_df)}/{len(y_tr)}, "
                    f"test={len(meta_te_df)}, cols={prob_cols}"
                )
            Z_tr = np.column_stack([LogitTransformer.transform(meta_tr_df[col].to_numpy()) for col in prob_cols])
            Z_te = np.column_stack([LogitTransformer.transform(meta_te_df[col].to_numpy()) for col in prob_cols])
            tr_d_ids = meta_tr_df["dialog_id"].to_numpy()
            tr_p_idx = meta_tr_df["participant_index"].to_numpy()
            tr_y = meta_tr_df["is_bot"].to_numpy()
            id_to_group = {str(d): int(g) for d, g in zip(dialog_ids_tr, groups_tr, strict=False)}
            id_to_pseudo = {str(d): bool(flag) for d, flag in zip(dialog_ids_tr, is_pseudo_tr, strict=False)}
            tr_groups = np.array([id_to_group.get(str(d), i // 2) for i, d in enumerate(tr_d_ids)])
            tr_pseudo = np.array([id_to_pseudo.get(str(d), False) for d in tr_d_ids], dtype=bool)
            logger.info(
                "Loaded Level-2 meta dataset via ClickHouse INNER JOIN (%d train, %d test)", len(Z_tr), len(Z_te)
            )
            return Z_tr, Z_te, tr_y, tr_groups, tr_d_ids, tr_p_idx, tr_pseudo

        Z_tr, Z_te = self._logits_from_l1(l1)
        return Z_tr, Z_te, y_tr, groups_tr, dialog_ids_tr, participant_indices_tr, is_pseudo_tr

    @staticmethod
    def compute_challenger_loss(
        *,
        y_te: np.ndarray,
        stacker_res: TrainingResult,
    ) -> float:
        """Строгий расчет целевого `challenger_loss` на изолированном holdout-срезе (без фолбэков)."""
        if len(y_te) < 4 or len(np.unique(y_te)) < 2:
            raise ValueError(f"Holdout split invalid for challenger evaluation: n={len(y_te)}")
        return float(log_loss(y_te, np.clip(stacker_res.test_calibrated, 1e-6, 1.0 - 1e-6)))

    def evaluate_decision(
        self,
        challenger_loss: float,
        champion_loss: float | None,
        l1: Level1Results,
    ) -> GatekeeperMetrics:
        """Оценка победы Challenger над Champion и формирование метрик."""
        if champion_loss is None:
            promoted = True
            decision = GatekeeperDecision.PROMOTED
            reason = "initial_champion_bootstrap"
        else:
            promoted = challenger_loss < champion_loss
            decision = GatekeeperDecision.PROMOTED if promoted else GatekeeperDecision.REJECTED
            reason = "challenger_improved_champion" if promoted else "challenger_loss_degraded"
        return GatekeeperMetrics(
            champion_loss=round(champion_loss, 5) if champion_loss is not None else None,
            challenger_loss=round(challenger_loss, 5),
            catboost_val_loss=round(l1.cb_loss, 5),
            lightgbm_val_loss=round(l1.lgb_loss, 5),
            xgboost_val_loss=round(l1.xgb_loss, 5),
            meta_stacker_val_loss=round(challenger_loss, 5),
            val_logloss=round(challenger_loss, 5),
            promoted=promoted,
            decision=decision,
            reason=reason,
        )

    @staticmethod
    def analyze_drift(
        *,
        l1: Level1Results,
        reference_importance: dict[str, float],
        model_probs: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, Any], Any]:
        """Анализ сдвига важности признаков (переиспользует L1 CatBoost) и Concept Drift без лишнего fit."""
        current_importance = dict(l1.cb_feature_importances)
        diff_report = FeatureImportanceDiffAnalyzer(reference_importance=reference_importance).analyze(
            current_importance, top_n=max(1, len(current_importance))
        )
        FeatureImportanceDiffPrometheusExporter().export(diff_report)
        if model_probs and len(model_probs) >= 2:
            ConceptDriftDetector().detect_drift(
                model_probs=model_probs,
                current_importance=current_importance,
                reference_importance=reference_importance,
            )
        return diff_report.to_dict(), diff_report

    def train_and_evaluate_stacker(
        self,
        *,
        Z_tr: np.ndarray,
        Z_te: np.ndarray,
        y_tr: np.ndarray,
        groups_tr: np.ndarray,
        dialog_ids_tr: np.ndarray,
        participant_indices_tr: np.ndarray,
        dialog_ids_te: np.ndarray,
        participant_indices_te: np.ndarray,
        y_te: np.ndarray,
        is_pseudo_tr: np.ndarray,
        test_ids: np.ndarray,
        meta_feature_names: list[str],
        l1: Level1Results,
        selected: SelectedFeatures,
        champion_loss: float | None,
        champion_version: str | None,
        champion_run_id: str | None,
        champion_eval_mode: str,
        initial_params: dict[str, Any] | None = None,
        feature_store_state: dict[str, Any] | None = None,
    ) -> tuple[TrainingResult, GatekeeperMetrics, dict[str, Any] | None]:
        """Обучает `logistic_meta_stacker`, пишет Gatekeeper Tags и при победе промоутит ONNX-бандл."""
        stacker_trainer = TwoLevelStackingTrainer(
            config=ClassicalTrainerConfig(
                experiment_name=self.experiment_name,
                run_name="logistic_meta_stacker",
                description=(
                    "Двухуровневый мета-стекер TwoLevelStackingClassifier: логистическая регрессия "
                    "над логитами CatBoost, LightGBM и XGBoost с парной калибровкой диалогов."
                ),
                n_trials=self.n_trials,
                patience=self.patience,
                n_splits=self.n_splits,
                random_state=42,
                register_as=None,
                dataset_name="train_meta_level2",
                environment=ExperimentEnvironment.ONLINE,
                initial_params=initial_params,
            ),
            search_config=StackingSearchConfig(),
            level1_runs=l1.level1_runs,
            use_logits=True,
        )
        if self.n_trials > 0:
            stacker_trainer.tune(X=Z_tr, y=y_tr, groups=groups_tr, feature_names=meta_feature_names)

        fit_params = dict(stacker_trainer.best_params) if stacker_trainer.best_params else {"C": 1.0}
        state_holder: dict[str, Any] = {"metrics": None, "trained_models": None}

        def _post_fit(tracker: ExperimentTracker, res: TrainingResult) -> None:
            challenger_loss = self.compute_challenger_loss(y_te=y_te, stacker_res=res)
            metrics = self.evaluate_decision(challenger_loss=challenger_loss, champion_loss=champion_loss, l1=l1)
            state_holder["metrics"] = metrics
            tracker.log_tags(
                {
                    "decision": str(metrics.decision),
                    "promoted": str(metrics.promoted),
                    "gatekeeper_reason": metrics.reason,
                    "champion_eval_mode": champion_eval_mode,
                    "champion_version": champion_version,
                    "champion_run_id": champion_run_id,
                    "champion_val_logloss": round(champion_loss, 5) if champion_loss is not None else None,
                    "candidate_val_logloss": round(challenger_loss, 5),
                    "pseudo_train_rows": int(np.sum(is_pseudo_tr)),
                    "catboost_val_logloss": round(l1.cb_loss, 5),
                    "lightgbm_val_logloss": round(l1.lgb_loss, 5),
                    "xgboost_val_logloss": round(l1.xgb_loss, 5),
                    **{f"level1_{k}_run_id": v for k, v in l1.level1_runs.items() if v},
                }
            )
            if metrics.promoted:
                state_holder["trained_models"] = ModelPromotionService.promote_in_active_run(
                    tracker=tracker,
                    stacker_full_model=res.full_model,
                    l1=l1,
                    selected=selected,
                    feature_store_state=feature_store_state,
                )

        stacker_res = stacker_trainer.fit(
            X_train=Z_tr,
            y_train=y_tr,
            groups=groups_tr,
            X_test=Z_te,
            dialog_ids_train=dialog_ids_tr,
            participant_indices_train=participant_indices_tr,
            dialog_ids_test=dialog_ids_te,
            participant_indices_test=participant_indices_te,
            test_ids=test_ids,
            feature_names=meta_feature_names,
            params=fit_params,
            post_fit_callback=_post_fit,
        )

        ChampionEvaluator.save_predictions_to_dwh(
            dwh=self.dwh,
            run_id=stacker_res.run_id,
            run_name="logistic_meta_stacker",
            dataset_split="train_oof",
            dialog_ids=dialog_ids_tr,
            participant_indices=participant_indices_tr,
            prob_bot=stacker_res.oof_raw,
            prob_bot_calibrated=stacker_res.oof_calibrated,
            is_bot=y_tr,
            fold_indices=stacker_res.fold_indices,
        )
        ChampionEvaluator.save_predictions_to_dwh(
            dwh=self.dwh,
            run_id=stacker_res.run_id,
            run_name="logistic_meta_stacker",
            dataset_split="test",
            dialog_ids=dialog_ids_te,
            participant_indices=participant_indices_te,
            prob_bot=stacker_res.test_raw,
            prob_bot_calibrated=stacker_res.test_calibrated,
        )
        return stacker_res, state_holder["metrics"], state_holder["trained_models"]
