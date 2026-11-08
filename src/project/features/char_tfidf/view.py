from dataclasses import asdict
from typing import Any

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from project.features.base import BaseFeatureView
from project.features.char_tfidf.schema import CharTFIDFConfig
from project.features.common.sklearn_codec import SklearnTextCodec
from project.features.store.context import DialogueContext


class CharTFIDFView(BaseFeatureView):
    """Извлекает 35 латентных SVD-компонент символьных n-грамм (3-5) текста участника."""

    def __init__(self, config: CharTFIDFConfig | None = None) -> None:
        """Инициализация символьного векторизатора и SVD проектора."""
        self.config = config or CharTFIDFConfig()
        self.vectorizer = self._create_vectorizer()
        self.svd: TruncatedSVD | None = TruncatedSVD(n_components=self.config.n_components, random_state=42)
        self.is_fitted = False
        self._feature_names = [f"tfidf_char_svd_{i:02d}" for i in range(self.config.n_components)]

    def _create_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(self.config.ngram_min, self.config.ngram_max),
            max_features=self.config.max_features,
            min_df=self.config.min_df,
            sublinear_tf=self.config.sublinear_tf,
        )

    @property
    def features(self) -> list[str]:
        """Список имен SVD компонент от символьного TF-IDF."""
        return list(self._feature_names)

    def get_feature_names_out(self) -> list[str]:
        """Sklearn-совместимое получение списка имен признаков."""
        return self.features

    def fit(self, texts_or_dialogues: list[str] | list[list[dict[str, Any]]]) -> "CharTFIDFView":
        """Обучение символьного векторизатора и SVD на коллекции текстов или диалогов участников."""
        texts = SklearnTextCodec.extract_texts(texts_or_dialogues)
        if not texts or not any(t.strip() for t in texts):
            self.svd = None
            self._feature_names = []
            self.is_fitted = False
            return self

        mat = self.vectorizer.fit_transform(texts)
        n_comp = min(self.config.n_components, mat.shape[1] - 1, mat.shape[0] - 1)
        if n_comp > 0:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
            self.svd.fit(mat)
            self._feature_names = [f"tfidf_char_svd_{i:02d}" for i in range(n_comp)]
        else:
            self.svd = None
            self._feature_names = []
        self.is_fitted = True
        return self

    def to_state_dict(self) -> dict[str, Any]:
        """Сериализация состояния символьного TF-IDF и SVD в JSON-совместимый словарь."""
        return {
            "config": asdict(self.config),
            "is_fitted": bool(self.is_fitted),
            "feature_names": list(self._feature_names),
            "vectorizer": SklearnTextCodec.dump_tfidf(self.vectorizer) if self.is_fitted else None,
            "svd": SklearnTextCodec.dump_svd(self.svd) if self.is_fitted else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> "CharTFIDFView":
        """Восстановление состояния символьного TF-IDF и SVD из JSON-словаря."""
        cfg_data = state.get("config")
        if isinstance(cfg_data, dict):
            self.config = CharTFIDFConfig(**cfg_data)
            self.vectorizer = self._create_vectorizer()

        self._feature_names = [str(x) for x in state.get("feature_names", self._feature_names)]
        self.is_fitted = bool(state.get("is_fitted", False))

        if self.is_fitted and state.get("vectorizer"):
            SklearnTextCodec.load_tfidf(self.vectorizer, state["vectorizer"])
            self.svd = SklearnTextCodec.load_svd(state.get("svd"))
        else:
            self.svd = None
            self.is_fitted = False
        return self

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "CharTFIDFView":
        """Создание экземпляра `CharTFIDFView` из сериализованного состояния."""
        cfg_data = state.get("config")
        view = cls(config=CharTFIDFConfig(**cfg_data) if isinstance(cfg_data, dict) else None)
        return view.load_state_dict(state)

    def extract_participant_features(
        self,
        messages: list[dict[str, Any]],
        p: int,
        context: DialogueContext,
    ) -> dict[str, float]:
        """Извлечение символьных SVD проекций для участника p."""
        if not self.is_fitted or self.svd is None or not self._feature_names:
            return dict.fromkeys(self._feature_names, 0.0)

        vec = self.vectorizer.transform([context.p_text[p]])
        proj = self.svd.transform(vec)[0]
        return {feat: float(val) for feat, val in zip(self._feature_names, proj, strict=False)}
