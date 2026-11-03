import hashlib
import json
import os
import random
from typing import Any

import numpy as np


class Reproducibility:
    """Управление фиксацией генераторов случайных чисел."""

    @staticmethod
    def seed(seed: int = 42) -> None:
        """Фиксация генераторов случайных чисел для обеспечения полной воспроизводимости."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass


class DialogueHasher:
    """Детерминированное хэширование содержимого диалогов для кэша и дедупликации."""

    @staticmethod
    def compute(messages: list[dict[str, Any]]) -> str:
        """Вычисление детерминированного SHA-256 хэша списка сообщений диалога."""
        serialized = json.dumps(
            [{"p": m.get("participant_index"), "t": m.get("text", "").strip()} for m in messages],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class LogitTransformer:
    """Преобразование вероятностей в пространство логитов для мета-стэкинга."""

    @staticmethod
    def transform(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Преобразование вероятности в логит с защитным усечением краев."""
        p_clip = np.clip(p, eps, 1.0 - eps)
        return np.asarray(np.log(p_clip / (1.0 - p_clip)))


__all__ = [
    "DialogueHasher",
    "LogitTransformer",
    "Reproducibility",
]
