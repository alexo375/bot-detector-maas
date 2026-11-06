"""Модульные тесты для RuBERTModelLoader."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from project.models.nlp.rubert.loader import RuBERTModelLoader


def test_rubert_model_loader_missing_tokenizer(tmp_path: Path) -> None:
    """Проверка fail-fast в RuBERTModelLoader при отсутствии локальных файлов токенизатора."""
    mock_vm = MagicMock()
    expected_path = tmp_path / "rubert_detector" / "1" / "model.onnx"
    mock_vm.get_model_path.return_value = expected_path

    loader = RuBERTModelLoader(
        version_manager=mock_vm,
        tokenizer_source=tmp_path / "non_existent_tok",
    )

    assert loader.model_path == expected_path
    with pytest.raises(FileNotFoundError, match="tokenizer directory does not exist"):
        loader.load_tokenizer()


def test_rubert_model_loader_resolves_latest_version(tmp_path: Path, monkeypatch: object) -> None:
    """Проверка разрешения последней версии модели rubert_detector в репозитории Triton."""
    from project.core.config import config

    repo_dir = tmp_path / "rubert_detector"
    (repo_dir / "1").mkdir(parents=True)
    (repo_dir / "3").mkdir(parents=True)
    monkeypatch.setattr(config.triton, "model_repository", str(tmp_path))  # type: ignore[attr-defined]

    loader = RuBERTModelLoader()
    assert loader.model_path == repo_dir / "3" / "model.onnx"
