"""Unit-тесты для модуля Prediction Drift и энтропии моделей."""

import numpy as np

from project.drift.prediction_drift.calculator import PredictionDriftCalculator
from project.drift.prediction_drift.detector import PredictionDriftDetector
from project.drift.prediction_drift.schemas import PredictionDriftReport


def test_psi_identical_distributions():
    """Тест PSI на идентичных распределениях (значение должно быть близко к 0)."""
    np.random.seed(42)
    ref = np.random.uniform(0.0, 1.0, 1000)
    cur = np.random.uniform(0.0, 1.0, 1000)

    psi = PredictionDriftCalculator.psi(ref, cur)
    assert 0.0 <= psi < 0.05


def test_psi_heavily_shifted_distributions():
    """Тест PSI на сильно сдвинутых распределениях (значение должно превышать 0.25)."""
    ref = np.random.normal(0.2, 0.05, 1000).clip(0.0, 1.0)
    cur = np.random.normal(0.8, 0.05, 1000).clip(0.0, 1.0)

    psi = PredictionDriftCalculator.psi(ref, cur)
    assert psi >= 0.25


def test_psi_empty_or_nan():
    """Тест PSI на пустых массивах и массивах с NaN."""
    assert PredictionDriftCalculator.psi(np.array([]), np.array([])) == 0.0
    assert PredictionDriftCalculator.psi(np.array([np.nan]), np.array([0.5, 0.6])) == 0.0


def test_prediction_entropy_confident():
    """Тест энтропии на уверенных предсказаниях (близка к 0)."""
    confident_probs = np.array([0.01, 0.99, 0.02, 0.98])
    entropy = PredictionDriftCalculator.prediction_entropy(confident_probs)
    assert 0.0 <= entropy < 0.20


def test_prediction_entropy_uncertain():
    """Тест энтропии на максимально неопределенных предсказаниях (p=0.5 -> H=1.0)."""
    uncertain_probs = np.array([0.5, 0.5, 0.5])
    entropy = PredictionDriftCalculator.prediction_entropy(uncertain_probs)
    assert abs(entropy - 1.0) < 1e-3


def test_prediction_entropy_empty():
    """Тест энтропии на пустом массиве."""
    assert PredictionDriftCalculator.prediction_entropy(np.array([])) == 0.0


def test_verdict_shares():
    """Тест вычисления долей вердиктов."""
    verdicts = ["both_humans", "both_humans", "both_bots", "p0_bot"]
    shares = PredictionDriftCalculator.verdict_shares(verdicts)
    assert shares["both_humans"] == 0.50
    assert shares["both_bots"] == 0.25
    assert shares["p0_bot"] == 0.25


def test_prediction_drift_detector():
    """Тест комплексного детектора PredictionDriftDetector."""
    detector = PredictionDriftDetector()
    ref = np.array([0.1, 0.2, 0.15, 0.85, 0.9, 0.8])
    cur = np.array([0.12, 0.18, 0.16, 0.84, 0.88, 0.82])
    verdicts = ["both_humans", "both_bots", "both_humans"]

    report = detector.detect_drift(ref, cur, verdicts)
    assert isinstance(report, PredictionDriftReport)
    assert report.psi < 0.10
    assert report.is_drifted is False
    assert "both_humans" in report.verdict_shares
