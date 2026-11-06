"""Модульные тесты для RuBERTBackboneTrainer и валидации входного Polars DataFrame."""

import polars as pl
import pytest

from project.models.nlp.rubert.trainer import RuBERTBackboneTrainer


def test_rubert_trainer_validates_polars_input() -> None:
    """Проверка строгой валидации типа и колонок входного Polars DataFrame."""
    trainer = RuBERTBackboneTrainer()

    # Некорректный тип
    with pytest.raises(TypeError, match="train_data должен быть экземпляром polars.DataFrame"):
        trainer.fit(train_data=[1, 2, 3])  # type: ignore

    # Отсутствует колонка целевой метки
    invalid_df = pl.DataFrame({"dialog_id": ["d1"], "text": ["abc"]})
    with pytest.raises(ValueError, match="не найдена колонка целевой метки"):
        trainer.fit(train_data=invalid_df)

    # Отсутствует колонка текстов
    invalid_df_2 = pl.DataFrame({"dialog_id": ["d1"], "is_bot": [1]})
    with pytest.raises(ValueError, match="не найдена колонка текстов"):
        trainer.fit(train_data=invalid_df_2)


def test_rubert_trainer_extract_embeddings() -> None:
    """Проверка извлечения эмбеддингов CLS-токена через RuBERTBackboneTrainer."""
    from unittest.mock import MagicMock

    import numpy as np
    import torch

    trainer = RuBERTBackboneTrainer(device="cpu")
    mock_model = MagicMock()
    mock_model.bert.return_value.last_hidden_state = torch.ones((2, 4, 312), dtype=torch.float32)

    mock_tokenizer = MagicMock()
    enc_mock = MagicMock()
    enc_mock.to.return_value = {"input_ids": torch.zeros((2, 4))}
    mock_tokenizer.return_value = enc_mock

    texts = pl.DataFrame({"text": ["Привет", "Как дела?"]})
    embs = trainer.extract_embeddings(mock_model, mock_tokenizer, texts, batch_size=2)

    assert isinstance(embs, np.ndarray)
    assert embs.shape == (2, 312)
    assert np.allclose(embs, 1.0)
