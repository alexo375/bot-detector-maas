"""Property-based тесты пайплайна признаков и постобработки на базе Hypothesis."""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from project.features.store import FeatureStore
from project.models.postprocessors.pairwise import PairwisePostprocessor


@st.composite
def arbitrary_dialogues(draw: st.DrawFn) -> list[dict[str, int | str]]:
    """Генератор произвольных диалогов со случайными участниками, текстами и эмодзи."""
    n_msgs = draw(st.integers(min_value=0, max_value=25))
    messages: list[dict[str, int | str]] = []
    for _ in range(n_msgs):
        p_idx = draw(st.integers(min_value=0, max_value=1))
        text = draw(st.text(max_size=200))
        messages.append({"participant_index": p_idx, "text": text})
    return messages


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(msgs=arbitrary_dialogues())
def test_feature_store_robustness_against_random_inputs(msgs: list[dict[str, int | str]]) -> None:
    """Property-based тест: FeatureStore не падает, не возвращает NaN или Inf на любых текстах."""
    store = FeatureStore()
    features = store.transform([msgs])

    assert isinstance(features, np.ndarray)
    assert features.shape == (2, len(store.default_features))
    assert not np.isnan(features).any(), "Обнаружены значения NaN в матрице признаков!"
    assert not np.isinf(features).any(), "Обнаружены значения Inf в матрице признаков!"


@settings(max_examples=50, deadline=None)
@given(
    p0=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    p1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_pairwise_postprocessor_physical_constraint(p0: float, p1: float) -> None:
    """Property-based тест: сумма вероятностей пары участников гарантированно <= 1.0."""
    postprocessor = PairwisePostprocessor(max_pair_sum=1.0, eps=0.005)

    probs = np.array([p0, p1], dtype=np.float64)
    dialog_ids = np.array(["d_test", "d_test"])
    participant_indices = np.array([0, 1])

    calibrated = postprocessor.transform(
        probs,
        dialog_ids=dialog_ids,
        participant_indices=participant_indices,
    )

    assert len(calibrated) == 2
    assert np.all(calibrated >= 0.005 - 1e-6)
    assert np.all(calibrated <= 0.995 + 1e-6)
    assert calibrated[0] + calibrated[1] <= 1.0 + 1e-4, f"Сумма вероятностей {calibrated} превысила 1.0!"


def test_pairwise_postprocessor_single_participant_no_penalty() -> None:
    """Диалог с одним участником не должен штрафоваться занижением из-за отсутствующего собеседника."""
    postprocessor = PairwisePostprocessor(max_pair_sum=1.0, eps=0.005)

    probs = np.array([0.7], dtype=np.float64)
    dialog_ids = np.array(["d_single"])
    participant_indices = np.array([0])

    calibrated = postprocessor.transform(
        probs,
        dialog_ids=dialog_ids,
        participant_indices=participant_indices,
    )

    assert len(calibrated) == 1
    assert np.isclose(calibrated[0], 0.7, atol=1e-3), (
        f"Вероятность участника в монологе была занижена: {calibrated[0]} вместо 0.7"
    )
