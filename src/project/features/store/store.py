import inspect
import json
import threading
from pathlib import Path
from typing import Any, cast

import numpy as np

from project.core.config import config
from project.features.base import BaseFeatureView
from project.features.store.context import DialogueContext

FEATURE_STORE_STATE_FILENAME = "feature_store_state.json"


class FeatureStore:
    """Единое хранилище признаков (8 Feature Views) с сериализацией состояния без pickle."""

    def __init__(
        self,
        views: list[BaseFeatureView] | None = None,
        state_path: str | Path | None = None,
        auto_load_from_repo: bool = False,
    ) -> None:
        """Инициализация хранилища признаков, проверка дубликатов и опциональная загрузка состояния."""
        self.auto_load_from_repo = auto_load_from_repo
        self._explicit_state_path: Path | None = Path(state_path) if state_path is not None else None
        self._loaded_state_path: Path | None = None
        self._loaded_state_mtime_ns: int | None = None
        self._state_lock = threading.RLock()

        self.views: list[BaseFeatureView] = views if views is not None else self.create_full_views()
        self.feature_to_view: dict[str, BaseFeatureView] = {}
        self.default_features: list[str] = []
        self.active_features: list[str] | None = None
        self._update_feature_maps()

        if self._explicit_state_path is not None:
            if not self._explicit_state_path.is_file():
                raise FileNotFoundError(f"FeatureStore state file not found: {self._explicit_state_path}")
            self.load_state(self._explicit_state_path)
        elif self.auto_load_from_repo:
            self.reload_if_changed()

    @staticmethod
    def create_full_views() -> list[BaseFeatureView]:
        """Фабрика полного канонического набора из 8 непересекающихся Feature Views (265 признаков)."""
        from project.features.baseline_heuristics import BaselineHeuristicsView
        from project.features.bayesian import BayesianTemplateView
        from project.features.char_tfidf import CharTFIDFView
        from project.features.interactivity import InteractivityView
        from project.features.structure import DialogueStructureView
        from project.features.text import TextMetricsView
        from project.features.tfidf import TFIDFView
        from project.features.typography import ChatTypographyView

        return [
            BaselineHeuristicsView(),
            DialogueStructureView(),
            TextMetricsView(),
            ChatTypographyView(),
            InteractivityView(),
            BayesianTemplateView(),
            TFIDFView(),
            CharTFIDFView(),
        ]

    @property
    def feature_names(self) -> list[str]:
        """Активный срез признаков (после прунинга) либо полный список всех 8 Feature Views."""
        return list(self.active_features) if self.active_features is not None else list(self.default_features)

    def get_feature_names_out(self) -> list[str]:
        """Sklearn-совместимое получение списка выходных признаков."""
        return self.feature_names

    def set_active_features(self, feature_names: list[str] | None) -> "FeatureStore":
        """Фиксация активного (отобранного) списка признаков для `transform`."""
        with self._state_lock:
            self.active_features = list(feature_names) if feature_names is not None else None
        return self

    @staticmethod
    def resolve_latest_state_path(triton_repo: Path | str | None = None) -> Path | None:
        """Поиск актуального `feature_store_state.json` в репозитории Triton (`bot_ensemble`)."""
        repo_root = Path(triton_repo or config.triton.model_repository)
        ensemble_dir = repo_root / "bot_ensemble"
        if not ensemble_dir.is_dir():
            return None
        versions = sorted([int(p.name) for p in ensemble_dir.iterdir() if p.is_dir() and p.name.isdigit()])
        for ver in reversed(versions):
            cand = ensemble_dir / str(ver) / FEATURE_STORE_STATE_FILENAME
            if cand.is_file():
                return cand
        return None

    def reload_if_changed(self) -> bool:
        """Потокобезопасная перезагрузка состояния FeatureStore; падает, если state-файл отсутствует."""
        target = self._explicit_state_path or self.resolve_latest_state_path()
        if target is None or not target.is_file():
            raise FileNotFoundError(
                f"{FEATURE_STORE_STATE_FILENAME} not found in {config.triton.model_repository}/bot_ensemble"
            )
        mtime_ns = target.stat().st_mtime_ns
        with self._state_lock:
            if self._loaded_state_path == target and self._loaded_state_mtime_ns == mtime_ns:
                return False
            self.load_state(target)
            self._loaded_state_path = target
            self._loaded_state_mtime_ns = mtime_ns
            return True

    def _prepare_dialogue_context(self, messages: list[dict[str, Any]]) -> DialogueContext:
        """Единый проход по диалогу для подготовки легковесного контекста."""
        total_msgs = len(messages)
        last_sender = int(messages[-1].get("participant_index", -1)) if total_msgs > 0 else -1
        last_msg_len = float(len(messages[-1].get("text", ""))) if total_msgs > 0 else 0.0
        p0_msgs = [m.get("text", "") for m in messages if m.get("participant_index") == 0]
        p1_msgs = [m.get("text", "") for m in messages if m.get("participant_index") == 1]

        pos_val: dict[int, float] = {0: 0.5, 1: 0.5}
        if total_msgs > 0:
            for p_idx in (0, 1):
                first_idx = next(
                    (idx for idx, m in enumerate(messages) if m.get("participant_index") == p_idx),
                    None,
                )
                if first_idx is not None:
                    pos_val[p_idx] = float(first_idx) / float(total_msgs)

        consec = {0: 0, 1: 0}
        cur_p = -1
        cur_c = 0
        for msg in messages:
            p_idx = msg.get("participant_index", -1)
            cur_c = cur_c + 1 if p_idx == cur_p else 1
            cur_p = p_idx
            if cur_p in consec:
                consec[cur_p] = max(consec[cur_p], cur_c)

        return DialogueContext(
            total_msgs=total_msgs,
            last_sender=last_sender,
            last_msg_len=last_msg_len,
            pos_val=pos_val,
            p_msgs={0: p0_msgs, 1: p1_msgs},
            p_text={0: " ".join(p0_msgs), 1: " ".join(p1_msgs)},
            consec=consec,
        )

    def _extract_single_dialogue(self, messages: list[dict[str, Any]], selected_features: list[str]) -> np.ndarray:
        active_views = {self.feature_to_view[feat] for feat in selected_features if feat in self.feature_to_view}
        context = self._prepare_dialogue_context(messages)
        rows = []
        for p in (0, 1):
            extracted: dict[str, float] = {}
            for view in active_views:
                extracted.update(view.extract_participant_features(messages, p, context))
            rows.append([extracted.get(feat, 0.0) for feat in selected_features])
        return np.array(rows, dtype=np.float32)

    def _update_feature_maps(self) -> None:
        """Обновление карты признаков после обучения или загрузки Feature Views."""
        self.feature_to_view = {}
        all_names: list[str] = []
        for view in self.views:
            for feat in view.features:
                if feat in self.feature_to_view:
                    raise ValueError(
                        f"Обнаружено дублирование признака '{feat}' между "
                        f"{self.feature_to_view[feat].__class__.__name__} и {view.__class__.__name__}"
                    )
                self.feature_to_view[feat] = view
                all_names.append(feat)
        self.default_features = all_names

    def fit(self, dialogues: list[list[dict[str, Any]]], y: np.ndarray | None = None) -> "FeatureStore":
        """Обучение трансформеров и Feature Views на диалогах."""
        for view in self.views:
            if hasattr(view, "fit"):
                sig = inspect.signature(view.fit)
                if "y" in sig.parameters and y is not None:
                    view.fit(dialogues, y=y)
                else:
                    view.fit(dialogues)
        self._update_feature_maps()
        return self

    def fit_transform(
        self,
        dialogues: list[list[dict[str, Any]]],
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        selected_features: list[str] | None = None,
    ) -> np.ndarray:
        """Обучение и извлечение матрицы признаков с честным OOF для supervised-признаков."""
        oof_by_view: dict[BaseFeatureView, tuple[list[str], np.ndarray]] = {}
        for view in self.views:
            if hasattr(view, "fit_transform_participant_features"):
                oof_res = view.fit_transform_participant_features(dialogues, y=y, groups=groups)
                if oof_res is not None:
                    oof_by_view[view] = (view.features, oof_res)
            elif hasattr(view, "fit"):
                sig = inspect.signature(view.fit)
                if "y" in sig.parameters and y is not None:
                    view.fit(dialogues, y=y)
                else:
                    view.fit(dialogues)

        self._update_feature_maps()
        target_features = selected_features or self.active_features or self.default_features
        if not dialogues:
            return np.empty((0, len(target_features)), dtype=np.float32)

        X_mat = self.transform(dialogues, selected_features=target_features)
        for feat_names, oof_matrix in oof_by_view.values():
            for feat_idx, feat_name in enumerate(feat_names):
                if feat_name in target_features:
                    col_idx = target_features.index(feat_name)
                    X_mat[:, col_idx] = oof_matrix[:, feat_idx]
        return cast("np.ndarray", np.nan_to_num(X_mat, nan=0.0, posinf=1.0, neginf=-1.0, copy=False))

    def transform(
        self,
        dialogues: list[list[dict[str, Any]]],
        selected_features: list[str] | None = None,
    ) -> np.ndarray:
        """Извлечение матрицы признаков для батча диалогов с санитацией NaN/Inf."""
        if self.auto_load_from_repo:
            self.reload_if_changed()
        with self._state_lock:
            target_features = selected_features or self.feature_names
            if not dialogues:
                return np.empty((0, len(target_features)), dtype=np.float32)
            raw_mat = np.vstack([self._extract_single_dialogue(msgs, target_features) for msgs in dialogues])
        return cast("np.ndarray", np.nan_to_num(raw_mat, nan=0.0, posinf=1.0, neginf=-1.0, copy=False))

    def to_state_dict(self) -> dict[str, Any]:
        """Экспорт состояния трансформеров и активных признаков в UTF-8 JSON-словарь (без pickle)."""
        from project.features.bayesian import BayesianTemplateView
        from project.features.char_tfidf import CharTFIDFView
        from project.features.tfidf import TFIDFView

        bayesian_state: dict[str, Any] | None = None
        tfidf_state: dict[str, Any] | None = None
        char_tfidf_state: dict[str, Any] | None = None
        for view in self.views:
            if isinstance(view, BayesianTemplateView):
                bayesian_state = view.to_state_dict()
            elif isinstance(view, TFIDFView):
                tfidf_state = view.to_state_dict()
            elif isinstance(view, CharTFIDFView):
                char_tfidf_state = view.to_state_dict()

        return {
            "version": 1,
            "active_features": list(self.feature_names),
            "bayesian": bayesian_state,
            "tfidf": tfidf_state,
            "char_tfidf": char_tfidf_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> "FeatureStore":
        """Восстановление обученного состояния всех view и списка активных признаков из JSON-словаря."""
        from project.features.bayesian import BayesianTemplateView
        from project.features.char_tfidf import CharTFIDFView
        from project.features.tfidf import TFIDFView

        with self._state_lock:
            for view in self.views:
                if isinstance(view, BayesianTemplateView) and isinstance(state.get("bayesian"), dict):
                    view.load_state_dict(state["bayesian"])
                elif isinstance(view, TFIDFView) and isinstance(state.get("tfidf"), dict):
                    view.load_state_dict(state["tfidf"])
                elif isinstance(view, CharTFIDFView) and isinstance(state.get("char_tfidf"), dict):
                    view.load_state_dict(state["char_tfidf"])
            self._update_feature_maps()
            raw_active = state.get("active_features")
            self.active_features = [str(col) for col in raw_active] if raw_active else None
        return self

    def save_state(self, path: str | Path) -> Path:
        """Сохранение состояния `FeatureStore` на диск в формате JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_state_dict(), ensure_ascii=False, separators=(",", ":"))
        target.write_text(payload, encoding="utf-8")
        return target

    def load_state(self, path: str | Path) -> "FeatureStore":
        """Загрузка состояния `FeatureStore` из JSON-файла."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            self.load_state_dict(payload)
        return self

    @classmethod
    def from_state_file(cls, path: str | Path) -> "FeatureStore":
        """Фабрика создания `FeatureStore` по сохраненному `feature_store_state.json`."""
        return cls(state_path=path, auto_load_from_repo=False)
