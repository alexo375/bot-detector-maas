"""Детектор сдвига концепта (Concept Drift) в ансамбле моделей."""

import numpy as np

from project.core.logging import ProjectLogger
from project.drift.concept_drift.calculator import ConceptDriftCalculator
from project.drift.concept_drift.prometheus_exporter import ConceptDriftPrometheusExporter
from project.drift.concept_drift.schemas import ConceptDriftReport

logger = ProjectLogger.get_logger(__name__)


class ConceptDriftDetector:
    """Детектор рассогласования ансамбля и смещения концепта признаков."""

    def __init__(
        self,
        metrics_exporter: ConceptDriftPrometheusExporter | None = None,
    ) -> None:
        self.metrics_exporter = metrics_exporter or ConceptDriftPrometheusExporter()

    def detect_drift(
        self,
        model_probs: dict[str, np.ndarray] | None = None,
        current_importance: dict[str, float] | None = None,
        reference_importance: dict[str, float] | None = None,
        disagreement_threshold: float = 0.35,
        correlation_threshold: float = 0.70,
    ) -> ConceptDriftReport:
        """Анализ сдвига концепта по рассогласованию моделей и корреляции важности признаков."""
        disagreement_rate = ConceptDriftCalculator.ensemble_disagreement(model_probs or {})
        feature_corr = ConceptDriftCalculator.feature_rank_correlation(
            current_importance or {},
            reference_importance or {},
        )

        has_disagreement_drift = bool(disagreement_rate >= disagreement_threshold)
        has_rank_drift = bool(feature_corr < correlation_threshold)
        concept_drift_detected = has_disagreement_drift or has_rank_drift

        logger.info(
            "Concept drift check: disagreement=%.2f%%, rank_corr=%.2f, detected=%s",
            disagreement_rate * 100,
            feature_corr,
            concept_drift_detected,
        )

        report = ConceptDriftReport(
            disagreement_rate=disagreement_rate,
            feature_rank_correlation=feature_corr,
            concept_drift_detected=concept_drift_detected,
        )
        self.metrics_exporter.export(report)
        return report
