"""Unit-тесты для модуля Concept Drift и рассогласования моделей."""

import numpy as np

from project.drift.concept_drift.calculator import ConceptDriftCalculator
from project.drift.concept_drift.detector import ConceptDriftDetector
from project.drift.concept_drift.schemas import ConceptDriftReport


def test_ensemble_disagreement_full_agreement():
    """Тест рассогласования ансамбля при полном согласии моделей (disagreement=0.0)."""
    model_probs = {
        "catboost": np.array([0.8, 0.9, 0.2, 0.1]),
        "lightgbm": np.array([0.7, 0.85, 0.3, 0.05]),
        "xgboost": np.array([0.65, 0.95, 0.4, 0.2]),
    }
    rate = ConceptDriftCalculator.ensemble_disagreement(model_probs)
    assert rate == 0.0


def test_ensemble_disagreement_partial():
    """Тест рассогласования при противоречивых предсказаниях."""
    model_probs = {
        "catboost": np.array([0.8, 0.2]),  # bot, human
        "lightgbm": np.array([0.3, 0.2]),  # human (disagrees on 0), human
    }
    rate = ConceptDriftCalculator.ensemble_disagreement(model_probs)
    assert rate == 0.50


def test_ensemble_disagreement_empty():
    """Тест рассогласования на пустых входных данных."""
    assert ConceptDriftCalculator.ensemble_disagreement({}) == 0.0
    assert ConceptDriftCalculator.ensemble_disagreement({"m1": np.array([])}) == 0.0


def test_feature_rank_correlation_identical():
    """Тест корреляции Спирмена при идентичном ранжировании признаков (rho=1.0)."""
    imp1 = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.1}
    imp2 = {"feat_a": 0.9, "feat_b": 0.4, "feat_c": 0.2}
    corr = ConceptDriftCalculator.feature_rank_correlation(imp1, imp2)
    assert abs(corr - 1.0) < 1e-4


def test_feature_rank_correlation_reversed():
    """Тест корреляции Спирмена при полностью инвертированном ранжировании (rho=-1.0)."""
    imp1 = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.1}
    imp2 = {"feat_a": 0.1, "feat_b": 0.3, "feat_c": 0.5}
    corr = ConceptDriftCalculator.feature_rank_correlation(imp1, imp2)
    assert abs(corr - (-1.0)) < 1e-4


def test_feature_rank_correlation_empty():
    """Тест корреляции на пустых или непересекающихся словарях."""
    assert ConceptDriftCalculator.feature_rank_correlation({}, {}) == 1.0
    assert ConceptDriftCalculator.feature_rank_correlation({"f1": 1.0}, {"f2": 1.0}) == 1.0


def test_concept_drift_detector():
    """Тест комплексного детектора ConceptDriftDetector."""
    detector = ConceptDriftDetector()
    model_probs = {
        "cb": np.array([0.8, 0.1, 0.9]),
        "lgb": np.array([0.85, 0.15, 0.95]),
    }
    cur_imp = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    ref_imp = {"a": 0.45, "b": 0.25, "c": 0.2, "d": 0.1}

    report = detector.detect_drift(model_probs, cur_imp, ref_imp)
    assert isinstance(report, ConceptDriftReport)
    assert report.disagreement_rate == 0.0
    assert report.feature_rank_correlation > 0.90
    assert report.concept_drift_detected is False

    # 30% расхождения < 0.35 (не вызывает ложной тревоги на чистом трафике), 50% >= 0.35 -> детектируется
    borderline_probs = {
        "cb": np.array([0.8, 0.1, 0.2, 0.1, 0.8, 0.2, 0.1, 0.2, 0.9, 0.1]),
        "lgb": np.array([0.2, 0.8, 0.8, 0.1, 0.8, 0.2, 0.1, 0.2, 0.9, 0.1]),
    }
    assert detector.detect_drift(borderline_probs, cur_imp, ref_imp).concept_drift_detected is False

    drifted_probs = {
        "cb": np.array([0.8, 0.1, 0.9, 0.2]),
        "lgb": np.array([0.2, 0.9, 0.9, 0.2]),
    }
    assert detector.detect_drift(drifted_probs, cur_imp, ref_imp).concept_drift_detected is True
