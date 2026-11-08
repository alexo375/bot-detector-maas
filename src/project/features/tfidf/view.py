from dataclasses import asdict
from typing import Any

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from project.features.base import BaseFeatureView
from project.features.common.sklearn_codec import SklearnTextCodec
from project.features.store.context import DialogueContext
from project.features.tfidf.schema import TFIDFConfig


class TFIDFView(BaseFeatureView):
    """Извлекает словесные TF-IDF n-граммы и их низкоразмерные SVD-компоненты."""

    def __init__(self, config: TFIDFConfig | None = None) -> None:
        """Инициализация словесного TF-IDF векторизатора и TruncatedSVD."""
        self.config = config or TFIDFConfig()
        self.vectorizer = self._create_vectorizer()
        self.svd: TruncatedSVD | None = (
            TruncatedSVD(n_components=self.config.svd_components, random_state=42)
            if self.config.svd_components > 0
            else None
        )
        self.is_fitted = False
        self._raw_names: list[str] = (
            [f"tfidf_{i:02d}" for i in range(self.config.max_features)] if self.config.include_raw else []
        )
        self._raw_terms: list[str] = []
        self._svd_names: list[str] = [f"tfidf_svd_{i}" for i in range(self.config.svd_components)]

    def _create_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=(self.config.ngram_min, self.config.ngram_max),
            min_df=self.config.min_df,
        )

    @property
    def features(self) -> list[str]:
        """Динамический список имен словесных TF-IDF и SVD признаков."""
        return list(self._raw_names) + list(self._svd_names)

    def get_feature_names_out(self) -> list[str]:
        """Sklearn-совместимое получение списка имен признаков."""
        return self.features

    def fit(self, texts_or_dialogues: list[str] | list[list[dict[str, Any]]]) -> "TFIDFView":
        """Обучение словаря TF-IDF и SVD проектора на коллекции текстов или диалогов."""
        texts = SklearnTextCodec.extract_texts(texts_or_dialogues)
        if not texts or not any(t.strip() for t in texts):
            self.svd = None
            self._svd_names = []
            self._raw_terms = []
            self.is_fitted = False
            return self

        mat = self.vectorizer.fit_transform(texts)
        self._raw_terms = [str(term) for term in self.vectorizer.get_feature_names_out()]
        if self.config.include_raw:
            self._raw_names = [f"tfidf_{i:02d}" for i in range(len(self._raw_terms))]
        else:
            self._raw_names = []

        n_comp = (
            min(self.config.svd_components, mat.shape[1] - 1, mat.shape[0] - 1)
            if self.config.svd_components > 0 and mat.shape[1] > 1
            else 0
        )
        if n_comp > 0:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
            self.svd.fit(mat)
            self._svd_names = [f"tfidf_svd_{i}" for i in range(n_comp)]
        else:
            self.svd = None
            self._svd_names = []

        self.is_fitted = True
        return self

    def to_state_dict(self) -> dict[str, Any]:
        """Сериализация состояния векторизатора и SVD в JSON-совместимый словарь."""
        return {
            "config": asdict(self.config),
            "is_fitted": bool(self.is_fitted),
            "raw_names": list(self._raw_names),
            "raw_terms": list(self._raw_terms),
            "svd_names": list(self._svd_names),
            "vectorizer": SklearnTextCodec.dump_tfidf(self.vectorizer) if self.is_fitted else None,
            "svd": SklearnTextCodec.dump_svd(self.svd) if self.is_fitted else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> "TFIDFView":
        """Восстановление состояния векторизатора и SVD из JSON-словаря."""
        cfg_data = state.get("config")
        if isinstance(cfg_data, dict):
            self.config = TFIDFConfig(**cfg_data)
            self.vectorizer = self._create_vectorizer()

        self._raw_names = [str(x) for x in state.get("raw_names", self._raw_names)]
        self._raw_terms = [str(x) for x in state.get("raw_terms", [])]
        self._svd_names = [str(x) for x in state.get("svd_names", self._svd_names)]
        self.is_fitted = bool(state.get("is_fitted", False))

        if self.is_fitted and state.get("vectorizer"):
            SklearnTextCodec.load_tfidf(self.vectorizer, state["vectorizer"])
            self.svd = SklearnTextCodec.load_svd(state.get("svd"))
        else:
            self.svd = None
            self.is_fitted = False
        return self

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "TFIDFView":
        """Создание экземпляра `TFIDFView` из сериализованного состояния."""
        cfg_data = state.get("config")
        view = cls(config=TFIDFConfig(**cfg_data) if isinstance(cfg_data, dict) else None)
        return view.load_state_dict(state)

    def extract_participant_features(
        self,
        messages: list[dict[str, Any]],
        p: int,
        context: DialogueContext,
    ) -> dict[str, float]:
        """Извлечение словесных TF-IDF весов и SVD-компонент для участника p."""
        if not self.is_fitted:
            return dict.fromkeys(self.features, 0.0)

        sparse_vec = self.vectorizer.transform([context.p_text[p]])
        result: dict[str, float] = {}
        if self.config.include_raw:
            dense_vec = sparse_vec.toarray()[0]
            for feat, val in zip(self._raw_names, dense_vec, strict=False):
                result[feat] = float(val)

        if self.svd is not None and self._svd_names:
            proj = self.svd.transform(sparse_vec)[0]
            for feat, val in zip(self._svd_names, proj, strict=False):
                result[feat] = float(val)

        return result
