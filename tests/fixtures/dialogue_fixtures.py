"""Модульные фикстуры канонических сообщений и полезной нагрузки диалогов."""

from typing import Any

import pytest


@pytest.fixture
def sample_dialogue() -> list[dict[str, Any]]:
    """Канонический 4-сообщенный диалог с репликами обоих участников."""
    return [
        {"participant_index": 0, "text": "Привет! Ты бот? Скажи 333))"},
        {"participant_index": 1, "text": "Привет! Отлично, работаю над проектом. А у тебя?"},
        {"participant_index": 0, "text": "хз норм че как дела))"},
        {"participant_index": 1, "text": "Пишу код на python и обучаю модели."},
    ]


@pytest.fixture
def short_dialogue() -> list[dict[str, Any]]:
    """Ультра-короткий диалог из двух сообщений (L=2) для краевых проверок."""
    return [
        {"participant_index": 0, "text": "Привет! Как дела?"},
        {"participant_index": 1, "text": "Привет, все супер, ты как?"},
    ]
