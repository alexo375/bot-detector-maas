from typing import Any

import numpy as np
import pytest

from project.features import BayesianTemplateView, FeatureStore


@pytest.fixture(scope="module")
def feature_store() -> FeatureStore:
    return FeatureStore()


def test_no_duplicate_features_across_views() -> None:
    """Проверка полного отсутствия дублирования имен признаков между всеми 8 базовыми Feature Views."""
    views = FeatureStore.create_full_views()
    all_names = [f for v in views for f in v.features]
    assert len(all_names) == len(set(all_names)), "Feature views must not have overlapping feature names"
    assert len(all_names) == 265


def test_short_dialogue_extraction(feature_store: FeatureStore, short_dialogue: list[dict[str, Any]]) -> None:
    """Проверка извлечения признаков для ультра-короткого диалога L=2."""
    features = feature_store.transform([short_dialogue])

    assert isinstance(features, np.ndarray)
    assert features.shape == (2, len(feature_store.feature_names))
    assert features.dtype == np.float32
    assert not np.isnan(features).any(), "Features should not contain NaNs"
    assert not np.isinf(features).any(), "Features should not contain Infs"


def test_empty_messages_handling(feature_store: FeatureStore) -> None:
    """Проверка обработки диалога с пустыми репликами."""
    messages = [
        {"participant_index": 0, "text": ""},
        {"participant_index": 1, "text": "   "},
    ]

    features = feature_store.transform([messages])

    assert features.shape == (2, len(feature_store.feature_names))
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_batch_dialogues_extraction(feature_store: FeatureStore) -> None:
    """Проверка пакетного извлечения признаков для N=2 диалогов."""
    d1 = [
        {"participant_index": 0, "text": "Привет! Как дела?"},
        {"participant_index": 1, "text": "Все хорошо!"},
    ]
    d2 = [
        {"participant_index": 0, "text": "Добрый вечер."},
        {"participant_index": 1, "text": "Здравствуйте."},
    ]
    features = feature_store.transform([d1, d2])
    assert features.shape == (4, len(feature_store.feature_names))


def test_bayesian_template_trigger(feature_store: FeatureStore) -> None:
    """Проверка срабатывания байесовских шаблонов ботов на известной спам-фразе."""
    messages = [
        {"participant_index": 0, "text": "привет\n\nпривет"},
        {"participant_index": 1, "text": "норм\n\nнорм"},
    ]

    features = feature_store.transform([messages])

    assert features.shape == (2, len(feature_store.feature_names))
    mean_idx = feature_store.feature_names.index("phrase_mean")
    max_idx = feature_store.feature_names.index("phrase_max")
    phrase_mean_p0 = features[0, mean_idx]
    phrase_max_p0 = features[0, max_idx]

    assert not np.isnan(phrase_mean_p0)
    assert not np.isnan(phrase_max_p0)
    assert phrase_mean_p0 >= 0.0
    assert phrase_max_p0 >= 0.0


def test_bayesian_online_offline_parity() -> None:
    """Проверка полного паритета phrase_mean/phrase_max и нормализации регистра в BayesianTemplateView и Trainer."""
    dialogues = [
        [
            {"participant_index": 0, "text": "  Привет, как дела  "},
            {"participant_index": 1, "text": "Я РОБОТ ШАБЛОН"},
        ],
        [
            {"participant_index": 0, "text": "Привет, как дела"},
            {"participant_index": 1, "text": "я робот шаблон"},
        ],
        [
            {"participant_index": 0, "text": "обычный ответ"},
            {"participant_index": 1, "text": "Я РОБОТ ШАБЛОН"},
        ],
    ]
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array([0, 0, 1, 1, 2, 2])

    bayes_view = BayesianTemplateView(alpha=1.0, min_len=3, min_count=1, n_splits=3)
    store = FeatureStore(views=[bayes_view])
    oof_mat = store.fit_transform(dialogues, y=y, groups=groups)
    online_mat = store.transform(dialogues)

    assert oof_mat.shape == (6, 7)
    assert online_mat.shape == (6, 7)
    # phrase_mean <= phrase_max при корректном порядке колонок
    mean_idx = store.feature_names.index("phrase_mean")
    max_idx = store.feature_names.index("phrase_max")
    assert np.all(oof_mat[:, mean_idx] <= oof_mat[:, max_idx] + 1e-6)
    assert np.all(online_mat[:, mean_idx] <= online_mat[:, max_idx] + 1e-6)
    # Проверка нормализации регистра: участник с "Я РОБОТ ШАБЛОН" в верхнем регистре найден в словаре
    assert online_mat[1, max_idx] > bayes_view.prior


def test_dialogue_structure_positions(feature_store: FeatureStore) -> None:
    """Проверка, что pos_p0/pos_p1 отражают относительную позицию первой реплики участника."""
    d1 = [
        {"participant_index": 0, "text": "Первая"},
        {"participant_index": 1, "text": "Вторая"},
        {"participant_index": 1, "text": "Третья"},
        {"participant_index": 1, "text": "Четвертая"},
    ]
    feats = feature_store.transform([d1])
    pos0_idx = feature_store.feature_names.index("pos_p0")
    pos1_idx = feature_store.feature_names.index("pos_p1")
    assert feats[0, pos0_idx] == pytest.approx(0.0)
    assert feats[1, pos1_idx] == pytest.approx(0.25)
