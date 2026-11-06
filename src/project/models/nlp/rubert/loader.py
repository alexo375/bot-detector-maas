from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from project.core.config import config


class RuBERTModelLoader:
    """Фабрика подготовки артефактов RuBERT для инференса.

    Модель загружается из Triton model repository (путь резолвится через
    TritonModelVersionManager), а токенизатор — из того же репозитория или по
    имени HuggingFace. Зависимости от жестко заданных путей в проекте
    исключены — все пути приходят через параметры или окружение.
    """

    def __init__(
        self,
        version_manager: Any = None,
        model_path: str | Path | None = None,
        tokenizer_source: str | Path | None = None,
    ) -> None:
        """Инициализация загрузчика."""
        self.version_manager = version_manager
        if model_path is not None:
            self.model_path = Path(model_path)
        elif self.version_manager is not None and hasattr(self.version_manager, "get_model_path"):
            self.model_path = self.version_manager.get_model_path("rubert_detector")
        else:
            repo = Path(config.triton.model_repository) / "rubert_detector"
            versions = (
                sorted([int(p.name) for p in repo.iterdir() if p.is_dir() and p.name.isdigit()])
                if repo.exists()
                else []
            )
            latest = versions[-1] if versions else 1
            self.model_path = repo / str(latest) / "model.onnx"

        self.tokenizer_source = Path(tokenizer_source) if tokenizer_source is not None else None

    def load_tokenizer(self) -> Any:
        """Строгая загрузка токенизатора из директории версии в Triton repository (без фолбэков)."""
        src = self.tokenizer_source if self.tokenizer_source is not None else self.model_path.parent
        tok_path = Path(str(src))
        if not tok_path.is_dir():
            raise FileNotFoundError(
                f"RuBERT tokenizer directory does not exist: {tok_path}. Publish rubert_detector first."
            )
        return AutoTokenizer.from_pretrained(str(tok_path), local_files_only=True)
