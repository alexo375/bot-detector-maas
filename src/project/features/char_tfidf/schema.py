from dataclasses import dataclass


@dataclass(frozen=True)
class CharTFIDFConfig:
    """Параметры символьного TF-IDF (n-граммы 3-5) и SVD проекции (35 компонент)."""

    max_features: int = 15000
    ngram_min: int = 3
    ngram_max: int = 5
    min_df: int = 2
    n_components: int = 35
    sublinear_tf: bool = True
