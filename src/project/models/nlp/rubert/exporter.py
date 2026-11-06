"""Экспорт RuBERT из MLflow-артефактов в ONNX для Triton model repository."""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

from project.core.logging import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class RuBertOnnxExporter:
    """Конвертер SequenceClassification-модели RuBERT и файлов токенизатора для Triton."""

    @staticmethod
    def export(
        weights_dir: Path,
        tokenizer_dir: Path,
        output_path: Path,
        max_seq_len: int = 128,
    ) -> Path:
        """Экспорт SequenceClassification-модели из MLflow-артефактов в ONNX."""
        del tokenizer_dir
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = AutoModelForSequenceClassification.from_pretrained(str(weights_dir), local_files_only=True)
        model.eval()

        dummy_ids = torch.ones(1, max_seq_len, dtype=torch.long)
        dummy_mask = torch.ones(1, max_seq_len, dtype=torch.long)

        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_ids, dummy_mask),
                str(output_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "seq_len"},
                    "attention_mask": {0: "batch_size", 1: "seq_len"},
                    "logits": {0: "batch_size"},
                },
                opset_version=18,
                do_constant_folding=True,
                dynamo=False,
            )

        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info("RuBERT exported to ONNX: %s (%.1f MB)", output_path, size_mb)
        return output_path

    @staticmethod
    def collect_tokenizer_files(tokenizer_dir: Path) -> dict[str, Path]:
        """Собрать файлы токенизатора, необходимые worker'у для `AutoTokenizer.from_pretrained`."""
        required = ["tokenizer_config.json"]
        optional = ["tokenizer.json", "vocab.txt", "special_tokens_map.json", "config.json"]

        files: dict[str, Path] = {}
        for name in required + optional:
            candidate = tokenizer_dir / name
            if candidate.is_file():
                files[name] = candidate
        missing_required = [name for name in required if name not in files]
        if missing_required:
            raise FileNotFoundError(f"Missing required tokenizer files {missing_required} in {tokenizer_dir}")
        return files


__all__ = ["RuBertOnnxExporter"]
