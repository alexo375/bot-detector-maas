"""Чистая JSON-сериализация и реконструкция sklearn TF-IDF и TruncatedSVD без pickle."""

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer


class SklearnTextCodec:
    """Кодек состояния текстовых трансформеров sklearn (JSON-safe, без pickle)."""

    @staticmethod
    def extract_texts(texts_or_dialogues: list[str] | list[list[dict[str, Any]]]) -> list[str]:
        """Преобразование диалогов или сырых текстов в плоский список реплик участников."""
        if (
            texts_or_dialogues
            and isinstance(texts_or_dialogues[0], list)
            and texts_or_dialogues[0]
            and isinstance(texts_or_dialogues[0][0], dict)
        ):
            return [
                " ".join([m.get("text") or "" for m in d if m.get("participant_index") == p])
                for d in texts_or_dialogues
                for p in (0, 1)
            ]
        return [str(t or "") for t in texts_or_dialogues]

    @staticmethod
    def dump_tfidf(vectorizer: TfidfVectorizer) -> dict[str, Any]:
        """Экспорт словаря и IDF-вектора обученного `TfidfVectorizer` в JSON-словарь."""
        return {
            "vocabulary": {str(term): int(idx) for term, idx in vectorizer.vocabulary_.items()},
            "idf": [float(v) for v in np.asarray(vectorizer.idf_, dtype=np.float64).tolist()],
        }

    @staticmethod
    def load_tfidf(vectorizer: TfidfVectorizer, state: dict[str, Any]) -> TfidfVectorizer:
        """Восстановление состояния `TfidfVectorizer` из JSON-словаря без переобучения."""
        vocab = {str(k): int(v) for k, v in state["vocabulary"].items()}
        idf_arr = np.asarray(state["idf"], dtype=np.float64)
        vectorizer.vocabulary_ = vocab
        vectorizer.fixed_vocabulary_ = True
        vectorizer._tfidf = TfidfTransformer(
            norm=vectorizer.norm,
            use_idf=vectorizer.use_idf,
            smooth_idf=vectorizer.smooth_idf,
            sublinear_tf=vectorizer.sublinear_tf,
        )
        vectorizer._tfidf.idf_ = idf_arr
        vectorizer._tfidf._idf_diag = sparse.diags(
            idf_arr,
            offsets=0,
            shape=(len(idf_arr), len(idf_arr)),
            format="csr",
        )
        return vectorizer

    @staticmethod
    def dump_svd(svd: TruncatedSVD | None) -> dict[str, Any] | None:
        """Экспорт матриц `TruncatedSVD` в JSON-словарь."""
        if svd is None or not hasattr(svd, "components_"):
            return None
        return {
            "n_components": int(svd.n_components),
            "components": np.asarray(svd.components_, dtype=np.float64).tolist(),
            "explained_variance": np.asarray(svd.explained_variance_, dtype=np.float64).tolist(),
            "explained_variance_ratio": np.asarray(svd.explained_variance_ratio_, dtype=np.float64).tolist(),
            "singular_values": np.asarray(svd.singular_values_, dtype=np.float64).tolist(),
        }

    @staticmethod
    def load_svd(state: dict[str, Any] | None) -> TruncatedSVD | None:
        """Восстановление обученного `TruncatedSVD` из JSON-словаря."""
        if not state:
            return None
        components = np.asarray(state["components"], dtype=np.float64)
        svd = TruncatedSVD(n_components=int(state["n_components"]), random_state=42)
        svd.components_ = components
        svd.n_features_in_ = int(components.shape[1])
        svd.explained_variance_ = np.asarray(state["explained_variance"], dtype=np.float64)
        svd.explained_variance_ratio_ = np.asarray(state["explained_variance_ratio"], dtype=np.float64)
        svd.singular_values_ = np.asarray(state["singular_values"], dtype=np.float64)
        return svd
