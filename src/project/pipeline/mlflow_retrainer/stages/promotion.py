"""Сервис ONNX-экспорта, сохранения `feature_store_state.json` и промоушена в MLflow Registry."""

from typing import Any

from project.core.logging import ProjectLogger
from project.pipeline.mlflow_retrainer.stages.ensemble_runner import Level1Results
from project.pipeline.mlflow_retrainer.stages.feature_selection import SelectedFeatures
from project.tracking.experiment_tracker.loggers.model_registry import ModelRegistryService
from project.tracking.experiment_tracker.loggers.onnx_model import OnnxModelLogger
from project.tracking.experiment_tracker.schemas import ExperimentEnvironment
from project.tracking.experiment_tracker.tracker import ExperimentTracker

logger = ProjectLogger.get_logger(__name__)


class ModelPromotionService:
    """Экспорт готовых full-refit моделей ансамбля в ONNX и промоушн `bot_detector@champion`."""

    @staticmethod
    def promote_in_active_run(
        *,
        tracker: ExperimentTracker,
        stacker_full_model: Any,
        l1: Level1Results,
        selected: SelectedFeatures,
        feature_store_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Экспорт pre-fitted full-моделей в ONNX, сохранение FeatureStore JSON и промоушн `@champion`."""
        cb_full = l1.cb_full_model
        lgb_full = l1.lgb_full_model
        xgb_full = l1.xgb_full_model
        if cb_full is None or lgb_full is None or xgb_full is None or stacker_full_model is None:
            raise ValueError("Pre-fitted full_model instances are required for promotion")

        lgb_booster = lgb_full.booster_ if hasattr(lgb_full, "booster_") else lgb_full
        total_feats = selected.total_features or len(selected.all_feature_names)
        model_logger = OnnxModelLogger(n_features=total_feats)
        model_logger.log_ensemble_models(
            catboost_model=cb_full,
            lightgbm_booster=lgb_booster,
            meta_stacker=stacker_full_model,
            xgboost_model=xgb_full,
            register_individual=False,
            cb_feature_indices=list(selected.cb_feature_indices),
            lgb_feature_indices=list(selected.lgb_feature_indices),
            xgb_feature_indices=list(selected.xgb_feature_indices),
            total_features=total_feats,
        )

        if feature_store_state is not None:
            tracker.log_feature_store_state(feature_store_state)

        if tracker.active_run is not None:
            ModelRegistryService.register_promoted_version(
                model=stacker_full_model,
                active_run=tracker.active_run,
                register_as="bot_detector",
                tracking_uri=tracker.tracking_uri,
                tags={
                    "environment": ExperimentEnvironment.ONLINE.value,
                    "model_family": "TwoLevelStackingClassifier",
                    "stage": "meta_stack",
                },
            )

        return {
            "catboost": cb_full,
            "lightgbm": lgb_booster,
            "xgboost": xgb_full,
            "meta_stacker": stacker_full_model,
        }


__all__ = ["ModelPromotionService"]
