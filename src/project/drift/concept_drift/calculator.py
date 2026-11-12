"""Математический расчет метрик рассогласования ансамбля и ранговой корреляции важности признаков."""

import numpy as np
from scipy.stats import spearmanr


class ConceptDriftCalculator:
    """Калькулятор метрик рассогласования ансамбля и ранговой корреляции важности признаков."""

    @staticmethod
    def ensemble_disagreement(model_probs: dict[str, np.ndarray]) -> float:
        """Вычисление доли диалогов с рассогласованием базовых моделей ансамбля."""
        if not model_probs or len(model_probs) < 2:
            return 0.0

        model_names = list(model_probs.keys())
        first_len = len(model_probs[model_names[0]])
        if first_len == 0:
            return 0.0

        binary_preds = [
            (np.asarray(model_probs[name], dtype=np.float64)[:first_len] >= 0.5).astype(int) for name in model_names
        ]
        stacked = np.stack(binary_preds, axis=0)
        unanimous = np.all(stacked == stacked[0:1, :], axis=0)
        disagreements = np.sum(~unanimous)
        return float(round(float(disagreements / first_len), 4))

    @staticmethod
    def feature_rank_correlation(
        current_importance: dict[str, float],
        reference_importance: dict[str, float],
    ) -> float:
        """Вычисление ранговой корреляции Спирмена важности признаков между версиями моделей."""
        if not current_importance or not reference_importance:
            return 1.0

        common_features = sorted(set(current_importance.keys()).intersection(reference_importance.keys()))
        if len(common_features) < 3:
            return 1.0

        cur_ranks = [current_importance[f] for f in common_features]
        ref_ranks = [reference_importance[f] for f in common_features]
        corr, _ = spearmanr(cur_ranks, ref_ranks)
        if np.isnan(corr):
            return 1.0
        return float(round(float(corr), 4))


__all__ = ["ConceptDriftCalculator"]
