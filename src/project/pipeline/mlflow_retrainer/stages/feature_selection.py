from dataclasses import dataclass, field
from typing import Any

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from project.core.logging import ProjectLogger
from project.models.classical.trainers.cross_validation import CrossValidationRunner
from project.tracking.experiment_tracker.loggers.features import FeatureImportanceExtractor

logger = ProjectLogger.get_logger(__name__)


@dataclass
class SelectedFeatures:
    """Контейнер отобранных признаков для моделей бустинга."""

    selected_by_model: dict[str, list[str]]
    k_cb: int
    m_lgb: int
    p_xgb: int
    cb_feature_indices: list[int] = field(default_factory=list)
    lgb_feature_indices: list[int] = field(default_factory=list)
    xgb_feature_indices: list[int] = field(default_factory=list)
    total_features: int = 0
    all_feature_names: list[str] = field(default_factory=list)


class FeaturePruner:
    """Sklearn-совместимый прунер константных и мультиколлинеарных признаков (`|r| > corr_threshold`)."""

    def __init__(self, corr_threshold: float = 0.95) -> None:
        self.corr_threshold = corr_threshold
        self.selected_indices_: list[int] = []
        self.selected_features_: list[str] = []
        self.dropped_constant_features_: list[str] = []
        self.dropped_correlated_features_: list[str] = []

    def fit(
        self,
        X: np.ndarray | pd.DataFrame | pl.DataFrame,
        y: np.ndarray | None = None,
        *,
        feature_names: list[str] | None = None,
    ) -> "FeaturePruner":
        """Вычисляет маску неконстантных и некоррелированных колонок строго по `X_train`."""
        del y
        if isinstance(X, (pd.DataFrame, pl.DataFrame)):
            names = feature_names or [str(c) for c in X.columns]
            X_mat = X.to_numpy()
        else:
            X_mat = np.asarray(X, dtype=np.float64)
            names = feature_names or [f"f_{i}" for i in range(X_mat.shape[1])]

        n_cols = X_mat.shape[1]
        if X_mat.shape[0] < 2 or n_cols <= 1:
            self.selected_indices_ = list(range(n_cols))
            self.selected_features_ = list(names)
            return self

        col_std = np.nanstd(X_mat, axis=0)
        non_const_idx = [int(i) for i in range(n_cols) if float(col_std[i]) > 1e-8]
        if not non_const_idx:
            non_const_idx = list(range(n_cols))
        self.dropped_constant_features_ = [names[i] for i in range(n_cols) if i not in set(non_const_idx)]

        if len(non_const_idx) == 1:
            self.selected_indices_ = non_const_idx
            self.selected_features_ = [names[i] for i in non_const_idx]
            return self

        X_sub = X_mat[:, non_const_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_mat = np.abs(np.corrcoef(X_sub, rowvar=False))
        corr_mat = np.nan_to_num(corr_mat, nan=0.0)
        upper = np.triu(corr_mat, k=1)

        kept_idx: list[int] = []
        dropped_corr: list[str] = []
        for local_j, orig_idx in enumerate(non_const_idx):
            if bool(np.any(upper[:, local_j] > self.corr_threshold)):
                dropped_corr.append(names[orig_idx])
            else:
                kept_idx.append(orig_idx)

        self.dropped_correlated_features_ = dropped_corr
        self.selected_indices_ = kept_idx or non_const_idx
        self.selected_features_ = [names[i] for i in self.selected_indices_]
        logger.info(
            "FeaturePruner (|r| > %.2f): %d -> %d features (%d constant, %d correlated dropped)",
            self.corr_threshold,
            n_cols,
            len(self.selected_features_),
            len(self.dropped_constant_features_),
            len(self.dropped_correlated_features_),
        )
        return self

    def transform(self, X: np.ndarray | pd.DataFrame | pl.DataFrame) -> np.ndarray:
        """Проецирует матрицу признаков на отобранные `selected_indices_`."""
        X_mat = X.to_numpy() if isinstance(X, (pd.DataFrame, pl.DataFrame)) else np.asarray(X)
        if not self.selected_indices_:
            return X_mat
        return X_mat[:, self.selected_indices_]

    def fit_transform(
        self,
        X: np.ndarray | pd.DataFrame | pl.DataFrame,
        y: np.ndarray | None = None,
        *,
        feature_names: list[str] | None = None,
    ) -> np.ndarray:
        """Обучает прунер и возвращает очищенную матрицу."""
        self.fit(X, y=y, feature_names=feature_names)
        return self.transform(X)

    def get_feature_names_out(self) -> list[str]:
        """Sklearn-совместимый список оставленных после прунинга признаков."""
        return list(self.selected_features_)

    @staticmethod
    def save_pruned_snapshot(
        *,
        dwh: Any,
        X_tr: np.ndarray,
        X_te: np.ndarray,
        dialog_ids_tr: np.ndarray,
        participant_indices_tr: np.ndarray,
        dialog_ids_te: np.ndarray,
        participant_indices_te: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """Перезаписывает актуальный срез `pruned_features` (train и test) в ClickHouse DWH."""
        if dwh is None or not feature_names:
            return
        try:
            dwh.features.drop_snapshot("pruned_features")
            desc = f"Отобранные после прунинга ({len(feature_names)} признаков, |corr| <= 0.95)"
            df_tr = pl.DataFrame(
                {
                    "dialog_id": [str(x) for x in dialog_ids_tr],
                    "participant_index": [int(x) for x in participant_indices_tr],
                    **{col: X_tr[:, idx].astype(np.float32).tolist() for idx, col in enumerate(feature_names)},
                }
            )
            dwh.features.save_snapshot(
                snapshot_tag="pruned_features",
                features_df=df_tr,
                dataset_type="train",
                description=desc,
            )
            if len(X_te) > 0 and len(dialog_ids_te) == len(X_te):
                df_te = pl.DataFrame(
                    {
                        "dialog_id": [str(x) for x in dialog_ids_te],
                        "participant_index": [int(x) for x in participant_indices_te],
                        **{col: X_te[:, idx].astype(np.float32).tolist() for idx, col in enumerate(feature_names)},
                    }
                )
                dwh.features.save_snapshot(
                    snapshot_tag="pruned_features",
                    features_df=df_te,
                    dataset_type="test",
                    description=desc,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist pruned_features snapshot to ClickHouse: %s", exc)


class FeatureSelectionStage:
    """Этап вычисления 5-fold CV важности и индивидуального отбора признаков для GBDT моделей."""

    @staticmethod
    def _rank_single_family(
        *,
        build_fn: Any,
        fit_fold_fn: Any,
        X_train: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        feature_names: list[str],
        params: dict[str, Any],
        n_splits: int,
    ) -> np.ndarray:
        if len(y) < 10 or len(np.unique(groups)) < 2:
            clf = build_fn(params)
            clf.fit(X_train, y)
            imp = FeatureImportanceExtractor.extract(clf)
            scores = np.asarray(imp, dtype=np.float64) if imp is not None else np.ones(X_train.shape[1])
            return np.argsort(scores)[::-1]
        eff_splits = max(2, min(n_splits, len(np.unique(groups))))
        ranking, _ = CrossValidationRunner.compute_cv_feature_importances(
            build_model_fn=build_fn,
            fit_fold_fn=fit_fold_fn,
            X_tr=X_train,
            y_tr=y,
            groups=groups,
            feature_names=feature_names,
            params=params,
            n_splits=eff_splits,
        )
        return ranking

    @classmethod
    def compute_rankings(
        cls,
        X_train: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 750,
        *,
        groups: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        n_splits: int = 5,
    ) -> dict[str, np.ndarray]:
        """Вычисление 5-fold CV ранжирования признаков для CatBoost, LightGBM и XGBoost."""
        rank_iters = min(150, n_estimators)
        groups_arr = groups if groups is not None else np.arange(len(y)) // 2
        names = feature_names or [f"f_{i}" for i in range(X_train.shape[1])]

        cb_ranking = cls._rank_single_family(
            build_fn=lambda p: cb.CatBoostClassifier(**p),
            fit_fold_fn=lambda model, X_tr, y_tr, X_va, y_va: model.fit(
                X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=25, verbose=False
            ),
            X_train=X_train,
            y=y,
            groups=groups_arr,
            feature_names=names,
            params={
                "iterations": rank_iters,
                "learning_rate": 0.05,
                "depth": 5,
                "verbose": 0,
                "random_seed": 42,
                "allow_writing_files": False,
            },
            n_splits=n_splits,
        )
        lgb_params = {
            "n_estimators": rank_iters,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "verbose": -1,
        }
        lgb_ranking = cls._rank_single_family(
            build_fn=lambda p: lgb.LGBMClassifier(**p),
            fit_fold_fn=lambda model, X_tr, y_tr, X_va, y_va: model.fit(
                X_tr, y_tr, eval_X=X_va, eval_y=y_va, callbacks=[lgb.early_stopping(25, verbose=False)]
            ),
            X_train=X_train,
            y=y,
            groups=groups_arr,
            feature_names=names,
            params=lgb_params,
            n_splits=n_splits,
        )
        xgb_params = {
            "n_estimators": rank_iters,
            "learning_rate": 0.05,
            "max_depth": 5,
            "random_state": 42,
            "eval_metric": "logloss",
            "n_jobs": 4,
        }
        xgb_ranking = cls._rank_single_family(
            build_fn=lambda p: xgb.XGBClassifier(**p),
            fit_fold_fn=lambda model, X_tr, y_tr, X_va, y_va: model.fit(
                X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False
            ),
            X_train=X_train,
            y=y,
            groups=groups_arr,
            feature_names=names,
            params=xgb_params,
            n_splits=n_splits,
        )
        return {"catboost": cb_ranking, "lightgbm": lgb_ranking, "xgboost": xgb_ranking}

    @staticmethod
    def select_features(
        all_feature_names: list[str],
        rankings: dict[str, np.ndarray],
        total_features: int,
        feature_keep_ratio: float,
    ) -> SelectedFeatures:
        """Отбор стартовых размеров сеток признаков для CatBoost, LightGBM и XGBoost."""
        min_feats = min(5, total_features)
        k_cb = max(min_feats, int(total_features * feature_keep_ratio))
        m_lgb = max(min_feats, min(total_features, int(total_features * min(1.0, feature_keep_ratio + 0.05))))
        p_xgb = max(min_feats, int(total_features * max(0.4, feature_keep_ratio - 0.05)))
        cb_idx = [int(i) for i in rankings["catboost"][:k_cb]]
        lgb_idx = [int(i) for i in rankings["lightgbm"][:m_lgb]]
        xgb_idx = [int(i) for i in rankings["xgboost"][:p_xgb]]
        cb_names = [all_feature_names[i] for i in cb_idx]
        lgb_names = [all_feature_names[i] for i in lgb_idx]
        xgb_names = [all_feature_names[i] for i in xgb_idx]
        return SelectedFeatures(
            selected_by_model={"catboost": cb_names, "lightgbm": lgb_names, "xgboost": xgb_names},
            k_cb=k_cb,
            m_lgb=m_lgb,
            p_xgb=p_xgb,
            cb_feature_indices=cb_idx,
            lgb_feature_indices=lgb_idx,
            xgb_feature_indices=xgb_idx,
            total_features=total_features,
            all_feature_names=list(all_feature_names),
        )
