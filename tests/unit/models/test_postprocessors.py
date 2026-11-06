"""Тесты для специализированных классов постпроцессинга и логитов стекера."""

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from project.models.classical import TwoLevelStackingClassifier
from project.models.postprocessors import PairwisePostprocessor


def test_pairwise_dialogue_postprocessor_scales_exceeding_pairs():
    """Проверка нормировки вероятностей при превышении суммы пары P0 + P1 > 1.0."""
    postprocessor = PairwisePostprocessor(max_pair_sum=1.0, eps=0.005)

    probs = np.array([0.8, 0.6, 0.2, 0.3])
    dialog_ids = np.array(["d1", "d1", "d2", "d2"])
    p_indices = np.array([0, 1, 0, 1])

    res = postprocessor.transform(probs, dialog_ids, p_indices)

    # Для d1: 0.8 + 0.6 = 1.4 > 1.0 => сумма масштабируется до ~0.995
    assert res[0] + res[1] <= 1.0
    assert np.isclose(res[0] / res[1], 0.8 / 0.6, atol=1e-5)

    # Для d2: 0.2 + 0.3 = 0.5 <= 1.0 => вероятности сохраняются
    assert np.isclose(res[2], 0.2, atol=1e-5)
    assert np.isclose(res[3], 0.3, atol=1e-5)


def test_pairwise_dialogue_postprocessor_polars_arrays():
    """Проверка работы PairwisePostprocessor с массивами из Polars DataFrame."""
    postprocessor = PairwisePostprocessor()
    df = pl.DataFrame(
        {
            "dialog_id": ["d1", "d1"],
            "participant_index": [0, 1],
            "is_bot": [0.7, 0.7],
        }
    )

    res = postprocessor.transform(
        df["is_bot"].to_numpy(),
        dialog_ids=df["dialog_id"].to_numpy(),
        participant_indices=df["participant_index"].to_numpy(),
    )
    assert res[0] + res[1] <= 1.0
    assert np.isclose(res[0], res[1])


def test_two_level_stacking_classifier_use_logits():
    """Проверка работы TwoLevelStackingClassifier с опцией use_logits=True."""
    X = np.random.RandomState(42).randn(40, 4)
    y = np.array([0, 1] * 20)
    groups = np.repeat(np.arange(20), 2)

    m1 = LogisticRegression(random_state=42)
    m2 = LogisticRegression(random_state=43)
    meta = LogisticRegression(random_state=44)

    stacker = TwoLevelStackingClassifier(
        base_models=[("m1", m1), ("m2", m2)],
        meta_model=meta,
        n_splits=3,
        use_logits=True,
    )
    stacker.fit(X, y, groups=groups)

    preds = stacker.predict_proba(X)
    assert preds.shape == (40, 2)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)

    tree = stacker.get_ensemble_tree()
    assert tree["level2_meta_model"]["use_logits"] is True
