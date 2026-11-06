from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from project.tracking.experiment_tracker.loggers.features import FeatureImportanceExtractor


class TwoLevelStackingClassifier(BaseEstimator, ClassifierMixin):
    """Универсальный стекинг-классификатор над ансамблем базовых моделей с мета-моделью второго уровня.

    Поддерживает два режима работы:
    1. End-to-end: принимает базовые модели и исходные признаки, обучает базовые модели по фолдам,
       собирает OOF-матрицу и обучает мета-модель.
    2. Direct meta-features: принимает предобученные мета-признаки (логиты/вероятности моделей 1-го уровня)
       и обучает мета-модель с честной оценкой OOF-метрик.
    """

    def __init__(
        self,
        base_models: list[tuple[str, Any]] | None = None,
        meta_model: Any | None = None,
        n_splits: int = 5,
        random_state: int = 42,
        use_logits: bool = True,
        eps: float = 1e-5,
    ) -> None:
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.use_logits = use_logits
        self.eps = eps

        # Fitted attributes
        self.fold_models_: list[dict[str, Any]] = []
        self.oof_matrix_: np.ndarray | None = None
        self.oof_predictions_: np.ndarray | None = None
        self.oof_metrics_: dict[str, dict[str, float]] = {}
        self.feature_names_in_: list[str] | None = None
        self._base_names: list[str] = []
        self.meta_model_: Any | None = None

    @property
    def coef_(self) -> np.ndarray:
        """Коэффициенты мета-модели второго уровня."""
        meta = self.meta_model_ if self.meta_model_ is not None else self.meta_model
        if meta is None or not hasattr(meta, "coef_"):
            raise AttributeError("Мета-модель еще не обучена или не содержит атрибута coef_.")
        return np.asarray(meta.coef_)

    @property
    def intercept_(self) -> np.ndarray:
        """Свободный член мета-модели второго уровня."""
        meta = self.meta_model_ if self.meta_model_ is not None else self.meta_model
        if meta is None or not hasattr(meta, "intercept_"):
            raise AttributeError("Мета-модель еще не обучена или не содержит атрибута intercept_.")
        return np.asarray(meta.intercept_)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Важность мета-признаков (абсолютные значения весов логистической регрессии)."""
        coef = self.coef_
        return np.abs(coef[0]) if getattr(coef, "ndim", 1) > 1 else np.abs(coef)

    def _to_meta_features(self, probs_or_logits: np.ndarray) -> np.ndarray:
        """Преобразование вероятностей в пространство мета-модели (логиты или вероятности)."""
        arr = np.asarray(probs_or_logits, dtype=np.float64)
        if not self.use_logits:
            return arr

        # Если данные уже в пространстве логитов (есть значения < 0 или > 1), возвращаем как есть
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            return arr

        p_clip = np.clip(arr, self.eps, 1.0 - self.eps)
        return np.log(p_clip / (1.0 - p_clip))

    def _fit_base_models(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        models_list: list[tuple[str, Any]],
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        self.oof_matrix_ = np.zeros((len(y_arr), len(models_list)), dtype=np.float64)
        self.fold_models_ = []
        for fold, (trn_idx, val_idx) in enumerate(splits):
            fold_dict = {}
            for m_idx, (name, base_estimator) in enumerate(models_list):
                clf = clone(base_estimator)
                if hasattr(clf, "random_seed"):
                    clf.set_params(random_seed=self.random_state + fold)
                elif hasattr(clf, "random_state"):
                    clf.set_params(random_state=self.random_state + fold)
                clf.fit(X_arr[trn_idx], y_arr[trn_idx])
                self.oof_matrix_[val_idx, m_idx] = clf.predict_proba(X_arr[val_idx])[:, 1]
                fold_dict[name] = clf
            self.fold_models_.append(fold_dict)

        for m_idx, name in enumerate(self._base_names):
            m_oof = self.oof_matrix_[:, m_idx]
            self.oof_metrics_[name] = {
                "logloss": float(log_loss(y_arr, m_oof)),
                "roc_auc": float(roc_auc_score(y_arr, m_oof)),
                "accuracy": float(accuracy_score(y_arr, (m_oof >= 0.5).astype(int))),
                "f1": float(f1_score(y_arr, (m_oof >= 0.5).astype(int))),
            }
        return self._to_meta_features(self.oof_matrix_)

    def _clone_meta_model(self) -> Any:
        if self.meta_model is None:
            return LogisticRegression(C=1.0, solver="lbfgs", random_state=self.random_state)
        return clone(self.meta_model)

    def fit(
        self,
        X: Any,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "TwoLevelStackingClassifier":
        """Обучение ансамбля: сбор OOF и обучение мета-модели второго уровня."""
        X_arr = np.asarray(X, dtype=np.float64)
        if hasattr(X, "columns"):
            self.feature_names_in_ = [str(c) for c in X.columns]
        elif feature_names is not None:
            self.feature_names_in_ = [str(f) for f in feature_names]
        else:
            self.feature_names_in_ = [f"meta_f_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=int)
        models_list = list(self.base_models or [])
        self._base_names = [name for name, _ in models_list]
        self.meta_model_ = self._clone_meta_model()

        splitter = (
            GroupKFold(n_splits=self.n_splits)
            if groups is not None
            else KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        )
        splits = list(splitter.split(X_arr, y_arr, groups) if groups is not None else splitter.split(X_arr, y_arr))

        if models_list:
            meta_train_features = self._fit_base_models(X_arr, y_arr, models_list, splits)
        else:
            meta_train_features = self._to_meta_features(X_arr)
            self.oof_matrix_ = meta_train_features.copy()
            self.fold_models_ = []

        self.oof_predictions_ = np.zeros(len(y_arr), dtype=np.float64)
        for trn_idx, val_idx in splits:
            fold_meta = clone(self.meta_model_)
            fold_meta.fit(meta_train_features[trn_idx], y_arr[trn_idx])
            self.oof_predictions_[val_idx] = fold_meta.predict_proba(meta_train_features[val_idx])[:, 1]

        self.meta_model_.fit(meta_train_features, y_arr)
        self.classes_ = np.array([0, 1])
        self.oof_metrics_["meta"] = {
            "logloss": float(log_loss(y_arr, self.oof_predictions_)),
            "roc_auc": float(roc_auc_score(y_arr, self.oof_predictions_)),
            "accuracy": float(accuracy_score(y_arr, (self.oof_predictions_ >= 0.5).astype(int))),
            "f1": float(f1_score(y_arr, (self.oof_predictions_ >= 0.5).astype(int))),
        }
        return self

    def transform_meta_features(self, X: Any) -> np.ndarray:
        """Генерация мета-признаков из входных данных."""
        X_arr = np.asarray(X, dtype=np.float64)
        models_list = list(self.base_models or [])

        if models_list and self.fold_models_:
            n_samples = len(X_arr)
            n_models = len(self._base_names)
            meta_features = np.zeros((n_samples, n_models), dtype=np.float64)

            for fold_dict in self.fold_models_:
                for m_idx, name in enumerate(self._base_names):
                    clf = fold_dict[name]
                    meta_features[:, m_idx] += clf.predict_proba(X_arr)[:, 1]

            meta_features /= len(self.fold_models_)
            return self._to_meta_features(meta_features)

        return self._to_meta_features(X_arr)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Предсказание вероятностей мета-моделью."""
        if self.meta_model_ is None:
            raise RuntimeError("Модель не обучена! Вызовите .fit() перед предсказанием.")
        meta_features = self.transform_meta_features(X)
        return np.asarray(self.meta_model_.predict_proba(meta_features))

    def predict(self, X: Any) -> np.ndarray:
        """Дискретное предсказание классов (0 или 1)."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def _extract_base_model_features(self, name: str, fallback_model: Any) -> list[dict[str, Any]]:
        """Усредняет важности признаков базовой модели по фолдам и возвращает отсортированный список."""
        feat_names = self.feature_names_in_ or []
        fitted = [fold[name] for fold in self.fold_models_ if name in fold] or [fallback_model]
        vectors = [
            np.asarray(imp, dtype=np.float64).ravel()
            for m in fitted
            if (imp := FeatureImportanceExtractor.extract(m)) is not None
        ]
        if not vectors:
            return []
        mean_imp = np.mean(np.vstack(vectors), axis=0)
        resolved = [feat_names[i] if i < len(feat_names) else f"f_{i}" for i in range(len(mean_imp))]
        items: list[dict[str, Any]] = [
            {"name": str(resolved[i]), "importance": round(float(mean_imp[i]), 5)} for i in range(len(mean_imp))
        ]
        items.sort(key=lambda item: float(item["importance"]), reverse=True)
        return items

    def get_ensemble_tree(self) -> dict[str, Any]:
        """Динамическая генерация иерархического дерева архитектуры ансамбля без хардкода."""
        meta = self.meta_model_ if self.meta_model_ is not None else self.meta_model
        meta_params = meta.get_params() if (meta is not None and hasattr(meta, "get_params")) else {}

        weights: dict[str, float] = {}
        feat_names = self._base_names if self._base_names else (self.feature_names_in_ or [])
        if meta is not None and hasattr(meta, "coef_"):
            coef = meta.coef_[0] if getattr(meta.coef_, "ndim", 1) > 1 else meta.coef_
            for idx, c in enumerate(coef):
                name = feat_names[idx] if idx < len(feat_names) else f"feature_{idx}"
                weights[str(name)] = round(float(c), 5)

        intercept: float | None = None
        if meta is not None and hasattr(meta, "intercept_"):
            intercept = round(float(meta.intercept_[0]), 5)

        tree_features: list[dict[str, Any]] = [
            {"name": str(name), "importance": round(abs(w), 5), "weight": w} for name, w in weights.items()
        ]
        tree_features.sort(key=lambda item: float(item["importance"]), reverse=True)

        base_summary: dict[str, Any] = {}
        for name, model in self.base_models or ():
            params = model.get_params() if hasattr(model, "get_params") else {}
            base_feats = self._extract_base_model_features(name, model)
            w = weights.get(name)
            base_summary[name] = {
                "model_family": model.__class__.__name__,
                "weight_in_stacker": w,
                "importance_in_stacker": round(abs(w), 5) if w is not None else None,
                "hyperparameters": {k: v for k, v in params.items() if not k.startswith("_")},
                "n_features": len(base_feats) or (len(self.feature_names_in_) if self.feature_names_in_ else 0),
                "feature_count": len(self.feature_names_in_) if self.feature_names_in_ else 0,
                "oof_metrics": self.oof_metrics_.get(name, {}),
                "features": base_feats,
            }

        return {
            "architecture": "TwoLevelStackingClassifier",
            "ensemble_architecture": "TwoLevelStackingClassifier",
            "model_family": "TwoLevelStackingClassifier",
            "n_features": len(tree_features),
            "features": tree_features,
            "weights": weights,
            "intercept": intercept,
            "n_splits": self.n_splits,
            "level2_meta_model": {
                "model_family": meta.__class__.__name__ if meta is not None else "LogisticRegression",
                "use_logits": self.use_logits,
                "hyperparameters": meta_params,
                "weights": weights,
                "intercept": intercept,
                "oof_metrics": self.oof_metrics_.get("meta", {}),
                "features": tree_features,
            },
            "level1_base_models": base_summary,
            "feature_names": feat_names,
        }
