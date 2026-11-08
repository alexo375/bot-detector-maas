from dataclasses import dataclass


@dataclass(frozen=True)
class TFIDFConfig:
    """Параметры словесного TF-IDF (1-2 граммы) и опционального SVD сжатия."""

    max_features: int = 120
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    svd_components: int = 15
    include_raw: bool = True
