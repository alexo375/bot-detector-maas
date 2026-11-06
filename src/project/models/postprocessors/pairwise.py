from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


class PairwisePostprocessor(BaseEstimator):
    """Калибровка вероятностей диалога с соблюдением физического ограничения P(Bot0) + P(Bot1) <= 1.0."""

    def __init__(
        self,
        max_pair_sum: float = 1.0,
        eps: float = 0.005,
    ) -> None:
        self.max_pair_sum = max_pair_sum
        self.eps = eps

    def fit(self, _X: Any = None, _y: Any = None) -> "PairwisePostprocessor":
        """Соответствие интерфейсу sklearn Transformer."""
        return self

    def fit_transform(self, X: Any = None, y: Any = None, **kwargs: Any) -> np.ndarray:
        """Совмещенное обучение и применение постобработки."""
        return self.fit(X, y).transform(X, **kwargs)

    def transform(
        self,
        X: np.ndarray | None = None,
        dialog_ids: np.ndarray | None = None,
        participant_indices: np.ndarray | None = None,
        probabilities: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Нормировка вероятностей диалога при превышении допустимой суммы пары."""
        del kwargs
        raw_probs = X if X is not None else probabilities
        if raw_probs is None:
            raise ValueError("Требуется передать массив вероятностей X или probabilities.")

        if dialog_ids is None:
            raise ValueError("Требуется передать dialog_ids.")

        if participant_indices is None:
            raise ValueError("Требуется передать participant_indices.")

        probs = np.clip(np.asarray(raw_probs, dtype=np.float64), self.eps, 1.0 - self.eps)
        d_ids = np.asarray(dialog_ids)
        p_indices = np.asarray(participant_indices, dtype=int)

        pair_dict: dict[Any, dict[int, float]] = {}
        for d, p, prob in zip(d_ids, p_indices, probs, strict=False):
            if d not in pair_dict:
                pair_dict[d] = {}
            pair_dict[d][p] = float(prob)

        adjusted_dict: dict[Any, dict[int, float]] = {}
        target_limit = self.max_pair_sum - self.eps

        for d, p_map in pair_dict.items():
            adjusted_dict[d] = {}
            if 0 in p_map and 1 in p_map:
                p0 = p_map[0]
                p1 = p_map[1]
                pair_sum = p0 + p1

                if pair_sum > self.max_pair_sum + 1e-5:
                    scale = target_limit / pair_sum
                    adjusted_dict[d][0] = p0 * scale
                    adjusted_dict[d][1] = p1 * scale
                else:
                    adjusted_dict[d][0] = p0
                    adjusted_dict[d][1] = p1
            else:
                for p_idx, prob_val in p_map.items():
                    adjusted_dict[d][p_idx] = prob_val

        result = np.zeros_like(probs)
        for i in range(len(probs)):
            d = d_ids[i]
            p = p_indices[i]
            val = adjusted_dict[d].get(p, probs[i])
            result[i] = np.clip(val, self.eps, 1.0 - self.eps)

        return result
