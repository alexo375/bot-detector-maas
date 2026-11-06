"""Модульные тесты для датасета DialogueTextDataset (RuBERT)."""

import torch

from project.models.nlp.rubert.dataset import DialogueTextDataset


def test_dialogue_text_dataset_with_labels() -> None:
    """Проверка формирования батчей с метками классов."""
    input_ids = torch.randint(0, 1000, (4, 16))
    attention_mask = torch.ones((4, 16), dtype=torch.long)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    dataset = DialogueTextDataset(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert len(dataset) == 4
    sample = dataset[1]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["labels"].item() == 1
    assert torch.equal(sample["input_ids"], input_ids[1])


def test_dialogue_text_dataset_without_labels() -> None:
    """Проверка формирования батчей без меток классов (режим инференса)."""
    input_ids = torch.randint(0, 1000, (2, 8))
    attention_mask = torch.ones((2, 8), dtype=torch.long)

    dataset = DialogueTextDataset(input_ids=input_ids, attention_mask=attention_mask, labels=None)

    assert len(dataset) == 2
    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" not in sample
