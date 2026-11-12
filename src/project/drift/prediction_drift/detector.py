"""Детектор дрифта вероятностей предсказаний и энтропии."""

import numpy as np

from project.core.logging import ProjectLogger
from project.drift.prediction_drift.calculator import PredictionDriftCalculator
from project.drift.prediction_drift.prometheus_exporter import PredictionDriftPrometheusExporter
from project.drift.prediction_drift.schemas import PredictionDriftReport

logger = ProjectLogger.get_logger(__name__)


class PredictionDriftDetector:
    """Детектор сдвига вероятностей предсказаний и неопределенности моделей."""

    def __init__(
        self,
        metrics_exporter: PredictionDriftPrometheusExporter | None = None,
    ) -> None:
        self.metrics_exporter = metrics_exporter or PredictionDriftPrometheusExporter()

    def detect_drift(
        self,
        reference_probs: np.ndarray,
        current_probs: np.ndarray,
        current_verdicts: list[str] | None = None,
        psi_threshold: float = 0.25,
        entropy_threshold: float = 0.85,
    ) -> PredictionDriftReport:
        """Статистический анализ сдвига вероятностей с расчетом PSI и энтропии."""
        logger.info(
            "Analyzing prediction drift (Ref: %d probs, Cur: %d probs)",
            len(reference_probs),
            len(current_probs),
        )

        psi = PredictionDriftCalculator.psi(reference_probs, current_probs)
        entropy = PredictionDriftCalculator.prediction_entropy(current_probs)
        verdict_shares = PredictionDriftCalculator.verdict_shares(current_verdicts or [])

        is_drifted = bool(psi >= psi_threshold or entropy >= entropy_threshold)
        report = PredictionDriftReport(
            psi=psi,
            mean_entropy=entropy,
            verdict_shares=verdict_shares,
            is_drifted=is_drifted,
        )
        self.metrics_exporter.export(report)
        return report
