"""Разделение диалоговых пар с сохранением реальных dialog_id для конвейера ретрейна."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from project.features.bayesian import BayesianTemplateView
from project.features.store import FeatureStore
from project.pipeline.mlflow_retrainer.components.data_loader import TrainingBundle


@dataclass(frozen=True)
class DatasetSplit:
    """Строго изолированные обучающая и отложенная выборки по парам диалогов."""

    X_tr: np.ndarray
    X_te: np.ndarray
    y_tr: np.ndarray
    y_te: np.ndarray
    tr_dlgs: list[list[dict[str, Any]]]
    te_dlgs: list[list[dict[str, Any]]]
    is_pseudo_tr: np.ndarray
    dialog_ids_tr: np.ndarray
    dialog_ids_te: np.ndarray
    participant_indices_tr: np.ndarray
    participant_indices_te: np.ndarray
    groups_tr: np.ndarray
    groups_te: np.ndarray


class DatasetSplitter:
    """Управление разделением данных на обучающую и валидационную выборки с изоляцией диалоговых пар."""

    @staticmethod
    def split_dialogue_pairs(bundle: TrainingBundle) -> DatasetSplit:
        """Разделение строго по парам диалогов: holdout только на gold-разметке, pseudo — только в train."""
        X_all, y_all, is_pseudo = bundle.X, bundle.y, bundle.is_pseudo
        texts_df = bundle.texts_df if len(bundle.texts_df) == len(y_all) else None

        gold_idx = np.where(~is_pseudo)[0]
        pseudo_idx = np.where(is_pseudo)[0]

        n_gold = len(gold_idx)
        n_gold_pairs = n_gold // 2
        if n_gold_pairs >= 2:
            val_pairs = max(1, n_gold_pairs // 5)
            split_gold = (n_gold_pairs - val_pairs) * 2
            tr_gold_idx = gold_idx[:split_gold]
            te_gold_idx = gold_idx[split_gold:]
        else:
            split_idx = max(4, (int(0.8 * n_gold) // 2) * 2) if n_gold >= 4 else n_gold
            tr_gold_idx = gold_idx[:split_idx]
            te_gold_idx = gold_idx[split_idx:] if split_idx < n_gold else gold_idx[: max(2, split_idx // 2)]

        tr_idx = np.concatenate([tr_gold_idx, pseudo_idx]) if len(pseudo_idx) > 0 else tr_gold_idx
        X_tr = X_all[tr_idx].copy()
        X_te = X_all[te_gold_idx].copy()
        y_tr = y_all[tr_idx]
        y_te = y_all[te_gold_idx]

        tr_dlgs: list[list[dict[str, Any]]] = []
        te_dlgs: list[list[dict[str, Any]]] = []
        if bundle.dialogues and len(bundle.dialogues) * 2 == len(y_all):
            tr_dlgs = [bundle.dialogues[i // 2] for i in tr_idx[::2] if (i // 2) < len(bundle.dialogues)]
            te_dlgs = [bundle.dialogues[i // 2] for i in te_gold_idx[::2] if (i // 2) < len(bundle.dialogues)]

        groups_tr = np.arange(len(y_tr)) // 2
        groups_te = np.arange(len(y_te)) // 2
        p_idx_tr = (
            texts_df[tr_idx.tolist()]["participant_index"].to_numpy()
            if texts_df is not None and "participant_index" in texts_df.columns
            else np.arange(len(y_tr)) % 2
        )
        p_idx_te = (
            texts_df[te_gold_idx.tolist()]["participant_index"].to_numpy()
            if texts_df is not None and "participant_index" in texts_df.columns
            else np.arange(len(y_te)) % 2
        )
        d_ids_tr = (
            texts_df[tr_idx.tolist()]["dialog_id"].to_numpy()
            if texts_df is not None and "dialog_id" in texts_df.columns
            else groups_tr.astype(str)
        )
        d_ids_te = (
            texts_df[te_gold_idx.tolist()]["dialog_id"].to_numpy()
            if texts_df is not None and "dialog_id" in texts_df.columns
            else np.array([f"te_{g}" for g in groups_te])
        )

        return DatasetSplit(
            X_tr=X_tr,
            X_te=X_te,
            y_tr=y_tr,
            y_te=y_te,
            tr_dlgs=tr_dlgs,
            te_dlgs=te_dlgs,
            is_pseudo_tr=is_pseudo[tr_idx],
            dialog_ids_tr=np.asarray(d_ids_tr, dtype=str),
            dialog_ids_te=np.asarray(d_ids_te, dtype=str),
            participant_indices_tr=np.asarray(p_idx_tr, dtype=np.int32),
            participant_indices_te=np.asarray(p_idx_te, dtype=np.int32),
            groups_tr=groups_tr,
            groups_te=groups_te,
        )

    @staticmethod
    def fit_full_feature_store(
        tr_dlgs: list[list[dict[str, Any]]],
        te_dlgs: list[list[dict[str, Any]]],
        y_tr: np.ndarray,
        groups_tr: np.ndarray,
        n_splits: int = 5,
    ) -> tuple[FeatureStore, np.ndarray, np.ndarray, list[str]]:
        """Обучение полного 8-view `FeatureStore` строго на train с 5-fold OOF Bayesian."""
        store = FeatureStore(auto_load_from_repo=False)
        for view in store.views:
            if isinstance(view, BayesianTemplateView):
                view.n_splits = max(2, min(n_splits, len(np.unique(groups_tr))))
        X_tr_raw = store.fit_transform(tr_dlgs, y=y_tr, groups=groups_tr)
        X_te_raw = store.transform(te_dlgs) if te_dlgs else np.empty((0, X_tr_raw.shape[1]), dtype=np.float32)
        return store, X_tr_raw, X_te_raw, list(store.default_features)


__all__ = ["DatasetSplit", "DatasetSplitter"]
