"""Тесты для HMMPostprocessor."""

import numpy as np

from project.models.postprocessors import HMMPostprocessor


def test_hmm_postprocessor_fit_and_properties():
    dialog_order = ["d1", "d2", "d3", "d4"]
    dialog_ids = ["d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4"]
    participant_indices = [0, 1, 0, 1, 0, 1, 0, 1]
    y_true = [0, 0, 1, 0, 0, 1, 0, 0]  # states: 0, 1, 2, 0

    hmm = HMMPostprocessor(alpha_laplace=1.0, blend_weight=0.7)
    hmm.fit(dialog_order, np.array(y_true), np.array(participant_indices), np.array(dialog_ids))

    assert hmm.transition_matrix_ is not None
    assert hmm.transition_matrix_.shape == (3, 3)
    np.testing.assert_allclose(hmm.transition_matrix_.sum(axis=1), np.ones(3), rtol=1e-5)

    assert hmm.initial_probs_ is not None
    assert hmm.initial_probs_.shape == (3,)
    assert np.isclose(hmm.initial_probs_.sum(), 1.0)


def test_hmm_postprocessor_transform():
    dialog_order = ["d1", "d2"]
    dialog_ids = np.array(["d1", "d1", "d2", "d2"])
    participant_indices = np.array([0, 1, 0, 1])
    y_true = np.array([0, 0, 1, 0])

    hmm = HMMPostprocessor(alpha_laplace=1.0, blend_weight=0.5)
    hmm.fit(dialog_order, y_true, participant_indices, dialog_ids)

    probs = np.array([0.1, 0.2, 0.8, 0.1])
    smoothed = hmm.transform(probs, dialog_ids, participant_indices, dialog_order)

    assert len(smoothed) == len(probs)
    assert np.all(smoothed >= 0.0)
    assert np.all(smoothed <= 1.0)


def test_hmm_postprocessor_zero_weight_returns_identical():
    dialog_order = ["d1", "d2"]
    dialog_ids = np.array(["d1", "d1", "d2", "d2"])
    participant_indices = np.array([0, 1, 0, 1])
    y_true = np.array([0, 0, 1, 0])

    hmm = HMMPostprocessor(alpha_laplace=1.0, blend_weight=0.0)
    hmm.fit(dialog_order, y_true, participant_indices, dialog_ids)

    probs = np.array([0.15, 0.25, 0.75, 0.35])
    smoothed = hmm.transform(probs, dialog_ids, participant_indices, dialog_order)
    np.testing.assert_allclose(smoothed, probs)
