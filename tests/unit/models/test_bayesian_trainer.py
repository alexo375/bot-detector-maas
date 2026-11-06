"""Модульные тесты для BayesianPhraseTrainer (Scikit-Learn совместимый тренер)."""

import numpy as np

from project.models.nlp.bayesian import BayesianPhraseTrainer


def test_bayesian_phrase_trainer_sklearn_contract() -> None:
    """Проверка соответствия BayesianPhraseTrainer контракту Scikit-Learn: fit, transform, fit_transform."""
    messages = [
        ["привет", "как дела"],
        ["привет", "я бот"],
        ["норм", "ты кто"],
        ["привет", "я бот", "уходи"],
        ["добрый день", "человек"],
        ["привет", "я бот"],
    ]
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array([0, 0, 1, 1, 2, 2])

    trainer = BayesianPhraseTrainer(alpha=1.0, min_len=3, min_count=1, cv=3)

    # 1. Out-of-fold fit_transform
    oof_features = trainer.fit_transform(messages, y, groups=groups)
    assert oof_features.shape == (6, 7)
    assert len(trainer.get_feature_names_out()) == 7
    assert np.all(oof_features[:, 0] >= 0.0)
    assert np.all(oof_features[:, 0] <= 1.0)

    # 2. Transform нового примера
    test_msgs = [["привет", "я бот"]]
    test_features = trainer.transform(test_msgs)
    assert test_features.shape == (1, 7)

    # 3. JSON сериализация в байты
    json_bytes = trainer.to_json_bytes()
    assert isinstance(json_bytes, bytes)
    assert len(json_bytes) > 0
