import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from project.core.config import config
from project.features.base import BaseFeatureView
from project.features.bayesian.schema import BayesianFeatures
from project.features.common.sklearn_codec import SklearnTextCodec
from project.features.store.context import DialogueContext

BAYESIAN_DICTIONARY_FILENAME = "bayesian_dictionary.json"


class BayesianTemplateView(BaseFeatureView):
    """Извлекает 7 байесовских признаков вероятности бот-шаблонов и лог-шансов."""

    schema = BayesianFeatures

    def __init__(
        self,
        dict_path: str | Path | None = None,
        dict_bytes: bytes | None = None,
        alpha: float = 3.0,
        min_len: int = 4,
        min_count: int = 2,
        n_splits: int = 5,
        auto_load_from_repo: bool = False,
    ) -> None:
        """Инициализация словаря байесовских правил из JSON-артефакта или параметров обучения."""
        self._explicit_dict_path: Path | None = Path(dict_path) if dict_path is not None else None
        self.dict_path = self._explicit_dict_path
        self.auto_load_from_repo = auto_load_from_repo
        self.alpha = alpha
        self.min_len = min_len
        self.min_count = min_count
        self.n_splits = n_splits
        self.prior = 0.5
        self.phrase_tot_cnt: dict[str, int] = {}
        self.phrase_bot_cnt: dict[str, int] = {}
        self.trainer: Any | None = None
        self._loaded_mtime_ns: int | None = None
        self._lock = threading.RLock()

        if dict_bytes is not None:
            self._load_from_json_bytes(dict_bytes)
        elif self.dict_path is not None:
            self._load_dictionary(self.dict_path)
        elif self.auto_load_from_repo:
            self.reload_if_changed()

    @staticmethod
    def resolve_latest_dict_path(triton_repo: Path | str | None = None) -> Path | None:
        """Находит путь к самой свежей версии `bayesian_dictionary.json` в `bot_ensemble` Triton."""
        repo_root = Path(triton_repo or config.triton.model_repository)
        ensemble_dir = repo_root / "bot_ensemble"
        if not ensemble_dir.is_dir():
            return None
        versions = sorted([int(p.name) for p in ensemble_dir.iterdir() if p.is_dir() and p.name.isdigit()])
        for ver in reversed(versions):
            cand = ensemble_dir / str(ver) / BAYESIAN_DICTIONARY_FILENAME
            if cand.is_file():
                return cand
        return None

    def get_feature_names_out(self) -> list[str]:
        """Sklearn-совместимое получение списка имен признаков."""
        return self.features

    def reload_if_changed(self) -> bool:
        """Перезагружает словарь из Triton-репозитория; падает, если файл отсутствует."""
        if self.trainer is not None:
            return False
        target = self._explicit_dict_path or self.resolve_latest_dict_path()
        if target is None or not target.is_file():
            raise FileNotFoundError(
                f"{BAYESIAN_DICTIONARY_FILENAME} not found in {config.triton.model_repository}/bot_ensemble"
            )
        mtime_ns = target.stat().st_mtime_ns
        with self._lock:
            if self.dict_path == target and self._loaded_mtime_ns == mtime_ns:
                return False
            self._load_dictionary(target)
            self.dict_path = target
            self._loaded_mtime_ns = mtime_ns
            return True

    @staticmethod
    def _to_participant_msgs(dialogues_or_msgs: list[list[dict[str, Any]]] | list[list[str]]) -> list[list[str]]:
        if (
            dialogues_or_msgs
            and isinstance(dialogues_or_msgs[0], list)
            and dialogues_or_msgs[0]
            and isinstance(dialogues_or_msgs[0][0], dict)
        ):
            return [
                [m.get("text") or "" for m in d if m.get("participant_index") == p]
                for d in dialogues_or_msgs
                for p in (0, 1)
            ]
        return dialogues_or_msgs  # type: ignore[return-value]

    def fit(
        self,
        dialogues_or_msgs: list[list[dict[str, Any]]] | list[list[str]],
        y: np.ndarray,
    ) -> "BayesianTemplateView":
        """Обучение глобального частотного словаря фраз."""
        from project.models.nlp.bayesian import BayesianPhraseTrainer

        if not dialogues_or_msgs:
            self.prior = 0.5
            self.phrase_tot_cnt = {}
            self.phrase_bot_cnt = {}
            self.trainer = None
            return self

        p_msgs = self._to_participant_msgs(dialogues_or_msgs)
        trainer = BayesianPhraseTrainer(
            alpha=self.alpha,
            min_len=self.min_len,
            min_count=self.min_count,
            n_splits=self.n_splits,
        )
        trainer.fit(p_msgs, y)
        self.trainer = trainer
        self.prior = float(trainer.prior_)
        self.phrase_tot_cnt = dict(trainer.tot_cnt_)
        self.phrase_bot_cnt = dict(trainer.bot_cnt_)
        return self

    def fit_transform_participant_features(
        self,
        dialogues: list[list[dict[str, Any]]],
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Несмещенное Out-of-Fold извлечение признаков с одновременным обучением словаря."""
        if y is None:
            return None

        from project.models.nlp.bayesian import BayesianPhraseTrainer

        _ = SklearnTextCodec
        p_msgs: list[list[str]] = [
            [m.get("text") or "" for m in d if m.get("participant_index") == p] for d in dialogues for p in (0, 1)
        ]
        trainer = BayesianPhraseTrainer(
            alpha=self.alpha,
            min_len=self.min_len,
            min_count=self.min_count,
            n_splits=self.n_splits,
        )
        oof_matrix = trainer.fit_transform(p_msgs, y, groups=groups)
        self.trainer = trainer
        self.prior = float(trainer.prior_)
        self.phrase_tot_cnt = dict(trainer.tot_cnt_)
        self.phrase_bot_cnt = dict(trainer.bot_cnt_)
        return oof_matrix

    def to_state_dict(self) -> dict[str, Any]:
        """Экспорт состояния байесовского словаря в JSON-словарь."""
        if self.trainer is not None:
            loaded = json.loads(bytes(self.trainer.to_json_bytes()).decode("utf-8"))
            if isinstance(loaded, dict):
                return loaded
        return {
            "alpha": float(self.alpha),
            "prior": float(self.prior),
            "min_len": int(self.min_len),
            "min_count": int(self.min_count),
            "phrases": {
                phrase: [int(self.phrase_bot_cnt.get(phrase, 0)), int(tot_c)]
                for phrase, tot_c in self.phrase_tot_cnt.items()
                if tot_c >= self.min_count
            },
        }

    def load_state_dict(self, payload: dict[str, Any]) -> "BayesianTemplateView":
        """Восстановление состояния байесовского словаря из JSON-словаря."""
        self.trainer = None
        self._populate_from_dict(payload)
        return self

    def to_json_bytes(self) -> bytes:
        """Сериализация словаря правил в байтовую строку UTF-8 JSON."""
        return json.dumps(self.to_state_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def _populate_from_dict(self, payload: dict[str, Any]) -> None:
        new_alpha = float(payload.get("alpha", 3.0))
        new_prior = float(payload.get("prior", 0.5))
        new_min_len = int(payload["min_len"]) if "min_len" in payload else self.min_len
        new_min_count = int(payload["min_count"]) if "min_count" in payload else self.min_count

        new_tot: dict[str, int] = {}
        new_bot: dict[str, int] = {}
        raw_phrases = payload.get("phrases", {})
        if isinstance(raw_phrases, dict):
            for phrase, counts in raw_phrases.items():
                if isinstance(counts, (list, tuple)) and len(counts) >= 2:
                    new_bot[str(phrase)] = int(counts[0])
                    new_tot[str(phrase)] = int(counts[1])

        with self._lock:
            self.alpha = new_alpha
            self.prior = new_prior
            self.min_len = new_min_len
            self.min_count = new_min_count
            self.phrase_tot_cnt = new_tot
            self.phrase_bot_cnt = new_bot

    def _load_from_json_bytes(self, raw_bytes: bytes) -> None:
        payload = json.loads(raw_bytes.decode("utf-8"))
        if isinstance(payload, dict):
            self._populate_from_dict(payload)

    def _load_dictionary(self, path: Path) -> None:
        if not path.is_file():
            return
        self._load_from_json_bytes(path.read_bytes())

    def _score_phrase(self, t: str) -> tuple[float, float]:
        n_tot = self.phrase_tot_cnt.get(t, 0)
        n_bot = self.phrase_bot_cnt.get(t, 0)
        n_hum = n_tot - n_bot
        a = n_bot + self.alpha * self.prior
        b = n_hum + self.alpha * (1.0 - self.prior)
        denom = a + b
        return a / denom, (a * b) / ((denom**2) * (denom + 1.0))

    def _boundary_phrase_prob(self, p_msgs_lower: list[str], index: int) -> float:
        if p_msgs_lower and len(p_msgs_lower[index]) >= self.min_len:
            text = p_msgs_lower[index]
            if self.phrase_tot_cnt.get(text, 0) >= self.min_count:
                return self._score_phrase(text)[0]
        return self.prior

    def _extract_phrase_stats(self, p_msgs_lower: list[str]) -> tuple[float, float, float, float, int]:
        scores: list[float] = []
        variances: list[float] = []
        pure_c = 0
        for text in p_msgs_lower:
            tot = self.phrase_tot_cnt.get(text, 0)
            if len(text) < self.min_len or tot < self.min_count:
                continue
            p_hat, var = self._score_phrase(text)
            scores.append(p_hat)
            variances.append(var)
            if self.phrase_bot_cnt.get(text, 0) >= 4 and self.phrase_bot_cnt.get(text, 0) == tot:
                pure_c += 1

        if not scores:
            return (
                self.prior,
                self.prior,
                (self.prior * (1.0 - self.prior)) / (self.alpha + 1.0),
                0.0,
                pure_c,
            )

        max_idx = int(np.argmax(scores))
        clipped = np.clip(scores, 1e-4, 1.0 - 1e-4)
        log_odds_sum = float(np.sum(np.log(clipped / (1.0 - clipped))))
        return float(np.mean(scores)), float(scores[max_idx]), float(variances[max_idx]), log_odds_sum, pure_c

    def extract_participant_features(
        self,
        messages: list[dict[str, Any]],
        p: int,
        context: DialogueContext,
    ) -> dict[str, float]:
        """Расчет апостериорных вероятностей фраз и дисперсии Бета-распределения для участника p."""
        p_msgs_lower = [m.strip().lower() for m in context.p_msgs[p]]
        with self._lock:
            if self.auto_load_from_repo:
                self.reload_if_changed()
            phrase_mean, phrase_max, beta_variance, log_odds_sum, pure_c = self._extract_phrase_stats(p_msgs_lower)
            greeting_prob = self._boundary_phrase_prob(p_msgs_lower, 0)
            farewell_prob = self._boundary_phrase_prob(p_msgs_lower, -1)

        return BayesianFeatures(
            phrase_mean=phrase_mean,
            phrase_max=phrase_max,
            pure_bot_count=float(pure_c),
            log_odds_sum=log_odds_sum,
            greeting_prob=greeting_prob,
            farewell_prob=farewell_prob,
            beta_variance=beta_variance,
        ).to_dict()
