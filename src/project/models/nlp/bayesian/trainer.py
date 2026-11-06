import json
from collections import Counter
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold, KFold

BAYESIAN_FEATURE_NAMES: list[str] = [
    "phrase_mean",
    "phrase_max",
    "pure_bot_count",
    "log_odds_sum",
    "greeting_prob",
    "farewell_prob",
    "beta_variance",
]


class BayesianPhraseTrainer(BaseEstimator, TransformerMixin):
    """Байесовский фразовый трансформер для обучения вероятностного словаря реплик.

    Соответствует контракту Scikit-Learn (fit, transform, fit_transform).
    Обеспечивает несмещенное Out-of-Fold (OOF) извлечение признаков при обучении для исключения
    Target Leakage, а также сериализует обученный словарь вероятностей в JSON-байты.
    """

    def __init__(
        self,
        alpha: float = 3.0,
        min_len: int = 4,
        min_count: int = 2,
        cv: int = 5,
        n_splits: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.min_len = min_len
        self.min_count = min_count
        self.cv = cv
        self.n_splits = n_splits

        self.prior_: float = 0.5
        self.tot_cnt_: Counter[str] = Counter()
        self.bot_cnt_: Counter[str] = Counter()
        self.feature_names_: list[str] = list(BAYESIAN_FEATURE_NAMES)

    def get_feature_names_out(self, _input_features: list[str] | None = None) -> list[str]:
        """Каноничный метод Scikit-Learn для получения имен сгенерированных признаков."""
        return list(self.feature_names_)

    def _score_phrase(self, phrase: str, tot_c: Counter[str], bot_c: Counter[str], prior: float) -> tuple[float, float]:
        """Расчет сглаженной вероятности и апостериорной дисперсии Beta-распределения."""
        n_tot = tot_c.get(phrase, 0)
        n_bot = bot_c.get(phrase, 0)
        n_hum = n_tot - n_bot

        a = n_bot + self.alpha * prior
        b = n_hum + self.alpha * (1.0 - prior)
        denom = a + b

        prob = a / denom
        var = (a * b) / ((denom**2) * (denom + 1.0))
        return prob, var

    def _extract_sample_features(
        self,
        msgs: list[str],
        tot_c: Counter[str],
        bot_c: Counter[str],
        prior: float,
    ) -> np.ndarray:
        """Извлечение вектора из 7 признаков для одного участника."""
        normalized_msgs = [m.strip().lower() for m in msgs]
        scores: list[float] = []
        variances: list[float] = []
        pure_c = 0

        for t in normalized_msgs:
            if len(t) >= self.min_len and tot_c.get(t, 0) >= self.min_count:
                p_hat, var = self._score_phrase(t, tot_c, bot_c, prior)
                scores.append(p_hat)
                variances.append(var)
                if bot_c.get(t, 0) >= 4 and bot_c.get(t, 0) == tot_c.get(t, 0):
                    pure_c += 1

        if scores:
            max_idx = int(np.argmax(scores))
            p_max = float(scores[max_idx])
            p_mean = float(np.mean(scores))
            p_var = float(variances[max_idx])
            clipped = np.clip(scores, 1e-4, 1.0 - 1e-4)
            log_odds = float(np.sum(np.log(clipped / (1.0 - clipped))))
        else:
            p_max = prior
            p_mean = prior
            p_var = (prior * (1.0 - prior)) / (self.alpha + 1.0)
            log_odds = 0.0

        if (
            normalized_msgs
            and len(normalized_msgs[0]) >= self.min_len
            and tot_c.get(normalized_msgs[0], 0) >= self.min_count
        ):
            p_greet, _ = self._score_phrase(normalized_msgs[0], tot_c, bot_c, prior)
        else:
            p_greet = prior

        if (
            normalized_msgs
            and len(normalized_msgs[-1]) >= self.min_len
            and tot_c.get(normalized_msgs[-1], 0) >= self.min_count
        ):
            p_farewell, _ = self._score_phrase(normalized_msgs[-1], tot_c, bot_c, prior)
        else:
            p_farewell = prior

        return np.array([p_mean, p_max, float(pure_c), log_odds, p_greet, p_farewell, p_var], dtype=np.float64)

    def fit(self, X: list[list[str]], y: np.ndarray) -> "BayesianPhraseTrainer":
        """Построение глобального байесовского частотного словаря фраз."""
        y_arr = np.asarray(y, dtype=int)
        self.prior_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.5
        self.tot_cnt_ = Counter()
        self.bot_cnt_ = Counter()

        for idx, msgs in enumerate(X):
            is_bot = y_arr[idx]
            for raw_t in msgs:
                t = (raw_t or "").strip().lower()
                if len(t) >= self.min_len:
                    self.tot_cnt_[t] += 1
                    if is_bot == 1:
                        self.bot_cnt_[t] += 1
        return self

    def transform(self, X: list[list[str]]) -> np.ndarray:
        """Трансформация реплик участников в матрицу признаков (N, 7)."""
        n_samples = len(X)
        feat_matrix = np.zeros((n_samples, len(self.feature_names_)), dtype=np.float64)
        for i, msgs in enumerate(X):
            feat_matrix[i] = self._extract_sample_features(msgs, self.tot_cnt_, self.bot_cnt_, self.prior_)
        return feat_matrix

    def fit_transform(
        self,
        X: list[list[str]],
        y: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        """Несмещенное Out-of-Fold извлечение признаков с обучением глобального словаря."""
        y_arr = np.asarray(y, dtype=int)
        n_samples = len(y_arr)
        oof_matrix = np.zeros((n_samples, len(self.feature_names_)), dtype=np.float64)
        if n_samples == 0:
            self.fit(X, y_arr)
            return oof_matrix

        effective_cv = self.n_splits if self.n_splits is not None else self.cv
        if groups is not None:
            splitter = GroupKFold(n_splits=effective_cv)
            splits = list(splitter.split(X, y_arr, groups))
        else:
            splitter = KFold(n_splits=effective_cv, shuffle=True, random_state=42)
            splits = list(splitter.split(X, y_arr))

        for trn_idx, val_idx in splits:
            fold_tot: Counter[str] = Counter()
            fold_bot: Counter[str] = Counter()
            for idx in trn_idx:
                is_b = y_arr[idx]
                for raw_t in X[idx]:
                    t = (raw_t or "").strip().lower()
                    if len(t) >= self.min_len:
                        fold_tot[t] += 1
                        if is_b == 1:
                            fold_bot[t] += 1

            fold_prior = float(np.mean(y_arr[trn_idx]))
            for idx in val_idx:
                oof_matrix[idx] = self._extract_sample_features(X[idx], fold_tot, fold_bot, fold_prior)

        # Обучаем глобальный словарь по всей выборке для теста и инференса
        self.fit(X, y_arr)
        return oof_matrix

    def to_json_dict(self) -> dict[str, Any]:
        """Сериализация словаря правил в JSON-совместимый словарь."""
        phrases = {
            phrase: [int(self.bot_cnt_.get(phrase, 0)), int(tot_c)]
            for phrase, tot_c in self.tot_cnt_.items()
            if tot_c >= self.min_count
        }
        return {
            "alpha": float(self.alpha),
            "prior": float(self.prior_),
            "min_len": int(self.min_len),
            "min_count": int(self.min_count),
            "phrases": phrases,
        }

    def to_json_bytes(self) -> bytes:
        """Сериализация словаря правил в UTF-8 JSON байты для MLflow и Triton."""
        return json.dumps(self.to_json_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
