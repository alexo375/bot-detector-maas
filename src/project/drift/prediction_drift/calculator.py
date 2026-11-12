"""Математический расчет метрик Population Stability Index (PSI) и энтропии предсказаний."""

import numpy as np


class PredictionDriftCalculator:
    """Калькулятор метрик PSI, энтропии предсказаний и долей вердиктов."""

    @staticmethod
    def psi(
        reference_probs: np.ndarray,
        current_probs: np.ndarray,
        num_bins: int = 10,
        epsilon: float = 1e-4,
    ) -> float:
        """Вычисление Population Stability Index (PSI) для вероятностных скоров моделей."""
        ref = np.asarray(reference_probs, dtype=np.float64)
        cur = np.asarray(current_probs, dtype=np.float64)
        ref = ref[~np.isnan(ref)]
        cur = cur[~np.isnan(cur)]

        if len(ref) < 2 or len(cur) < 2:
            return 0.0

        effective_bins = min(num_bins, max(2, min(len(ref), len(cur)) // 5))
        bins = np.linspace(0.0, 1.0, effective_bins + 1)
        bins[0] = -1e-6
        bins[-1] = 1.0 + 1e-6

        ref_counts, _ = np.histogram(ref, bins=bins)
        cur_counts, _ = np.histogram(cur, bins=bins)

        ref_props = ref_counts / float(len(ref))
        cur_props = cur_counts / float(len(cur))
        ref_fractions = (ref_props + epsilon) / (1.0 + effective_bins * epsilon)
        cur_fractions = (cur_props + epsilon) / (1.0 + effective_bins * epsilon)

        psi_val = np.sum((cur_fractions - ref_fractions) * np.log(cur_fractions / ref_fractions))
        return float(max(0.0, round(psi_val, 4)))

    @staticmethod
    def prediction_entropy(probs: np.ndarray, epsilon: float = 1e-6) -> float:
        """Вычисление средней энтропии Шеннона предсказаний в битах."""
        arr = np.asarray(probs, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return 0.0

        p = np.clip(arr, epsilon, 1.0 - epsilon)
        entropies = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
        return float(round(float(np.mean(entropies)), 4))

    @staticmethod
    def verdict_shares(verdicts: list[str]) -> dict[str, float]:
        """Вычисление процентного соотношения классов вердиктов в пачке."""
        if not verdicts:
            return {}
        total = len(verdicts)
        counts: dict[str, int] = {}
        for v in verdicts:
            v_str = str(v).lower()
            counts[v_str] = counts.get(v_str, 0) + 1
        return {k: round(v / total, 4) for k, v in counts.items()}


__all__ = ["PredictionDriftCalculator"]
