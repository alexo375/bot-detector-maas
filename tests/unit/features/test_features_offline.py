from typing import Any

import numpy as np

from project.features import (
    BaselineHeuristicsView,
    BayesianTemplateView,
    CharTFIDFConfig,
    CharTFIDFView,
    ChatTypographyView,
    DialogueStructureView,
    FeatureStore,
    InteractivityView,
    TextMetricsView,
    TFIDFConfig,
    TFIDFView,
)


def test_baseline_heuristics_view(sample_dialogue: list[dict[str, Any]]) -> None:
    view = BaselineHeuristicsView()
    assert len(view.features) == 32
    store = FeatureStore(views=[view])
    feats = store.transform([sample_dialogue])
    assert feats.shape == (2, 32)
    assert not np.isnan(feats).any()


def test_typography_and_interactivity_views(sample_dialogue: list[dict[str, Any]]) -> None:
    typo_view = ChatTypographyView()
    inter_view = InteractivityView()
    assert len(typo_view.features) == 14
    assert len(inter_view.features) == 11
    store = FeatureStore(views=[typo_view, inter_view])
    feats = store.transform([sample_dialogue])
    assert feats.shape == (2, 25)
    assert not np.isnan(feats).any()


def test_word_and_char_tfidf_views(sample_dialogue: list[dict[str, Any]]) -> None:
    texts = [
        "Привет! Как твои дела сегодня? Я тоже работаю. Что именно делаешь?",
        "Привет! Отлично, работаю над проектом. А у тебя? Пишу код на python и обучаю модели.",
        "Всем привет, нормальная погода на улице сегодня.",
        "Да, согласен, пойдем гулять вечером в парк.",
    ]
    word_view = TFIDFView(config=TFIDFConfig(max_features=10, min_df=1, svd_components=2))
    word_view.fit(texts)

    char_view = CharTFIDFView(config=CharTFIDFConfig(max_features=50, min_df=1, n_components=3))
    char_view.fit(texts)

    store = FeatureStore(views=[word_view, char_view])
    feats = store.transform([sample_dialogue])
    assert feats.shape == (2, len(word_view.features) + len(char_view.features))
    assert not np.isnan(feats).any()


def test_full_combined_store_has_zero_duplicates(sample_dialogue: list[dict[str, Any]]) -> None:
    texts = [
        "Привет! Как твои дела сегодня? Я тоже работаю.",
        "Привет! Отлично, работаю над проектом.",
        "Всем привет, нормальная погода.",
        "Да, согласен, пойдем гулять.",
    ]
    word_view = TFIDFView(config=TFIDFConfig(max_features=15, min_df=1, svd_components=2))
    word_view.fit(texts)
    char_view = CharTFIDFView(config=CharTFIDFConfig(max_features=40, min_df=1, n_components=3))
    char_view.fit(texts)

    store = FeatureStore(
        views=[
            BaselineHeuristicsView(),
            DialogueStructureView(),
            TextMetricsView(),
            ChatTypographyView(),
            InteractivityView(),
            BayesianTemplateView(),
            word_view,
            char_view,
        ]
    )
    assert len(store.feature_names) == len(set(store.feature_names))
    feats = store.transform([sample_dialogue])
    assert feats.shape == (2, len(store.feature_names))
    assert not np.isnan(feats).any()


def test_feature_store_fit_transform_oof(sample_dialogue: list[dict[str, str | int]]) -> None:
    dialogues = [sample_dialogue, sample_dialogue]
    y = np.array([0, 1, 0, 1])
    groups = np.array(["d1", "d1", "d2", "d2"])

    bayes_view = BayesianTemplateView(alpha=3.0, min_len=3, min_count=1, n_splits=2)
    word_view = TFIDFView(config=TFIDFConfig(max_features=10, min_df=1, svd_components=2))
    char_view = CharTFIDFView(config=CharTFIDFConfig(max_features=20, min_df=1, n_components=2))

    store = FeatureStore(
        views=[
            BaselineHeuristicsView(),
            DialogueStructureView(),
            TextMetricsView(),
            ChatTypographyView(),
            InteractivityView(),
            bayes_view,
            word_view,
            char_view,
        ]
    )

    X_tr = store.fit_transform(dialogues, y=y, groups=groups)
    assert X_tr.shape == (4, len(store.default_features))
    assert not np.isnan(X_tr).any()

    X_te = store.transform(dialogues)
    assert X_te.shape == (4, len(store.default_features))
    assert not np.isnan(X_te).any()


def test_feature_store_json_state_roundtrip(tmp_path: Any, sample_dialogue: list[dict[str, Any]]) -> None:
    texts = [
        "Привет! Как твои дела сегодня? Я тоже работаю.",
        "Привет! Отлично, работаю над проектом.",
        "Всем привет, нормальная погода на улице.",
        "Да, согласен, пойдем гулять вечером в парк.",
    ]
    dialogues = [
        [{"participant_index": 0, "text": texts[0]}, {"participant_index": 1, "text": texts[1]}],
        [{"participant_index": 0, "text": texts[2]}, {"participant_index": 1, "text": texts[3]}],
    ]
    y = np.array([0, 1, 0, 1])
    groups = np.array(["d1", "d1", "d2", "d2"])

    word_view = TFIDFView(config=TFIDFConfig(max_features=12, min_df=1, svd_components=2))
    char_view = CharTFIDFView(config=CharTFIDFConfig(max_features=20, min_df=1, n_components=2))
    store = FeatureStore(
        views=[
            BaselineHeuristicsView(),
            DialogueStructureView(),
            TextMetricsView(),
            ChatTypographyView(),
            InteractivityView(),
            BayesianTemplateView(alpha=2.0, min_len=3, min_count=1, n_splits=2),
            word_view,
            char_view,
        ]
    )
    store.fit_transform(dialogues, y=y, groups=groups)
    store.set_active_features(store.default_features[:10])

    state_file = store.save_state(tmp_path / "feature_store_state.json")
    restored = FeatureStore.from_state_file(state_file)
    restored_word = TFIDFView.from_state_dict(word_view.to_state_dict())
    restored_char = CharTFIDFView.from_state_dict(char_view.to_state_dict())

    assert restored.feature_names == store.default_features[:10]
    assert restored_word.features == word_view.features
    assert restored_char.features == char_view.features
    np.testing.assert_allclose(
        restored.transform([sample_dialogue]), store.transform([sample_dialogue]), rtol=1e-5, atol=1e-6
    )
