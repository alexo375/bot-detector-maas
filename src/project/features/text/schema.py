from dataclasses import dataclass

from project.features.base import BaseFeatureSchema


@dataclass(frozen=True)
class TextFeatures(BaseFeatureSchema):
    """13 признаков энтропии, лексического разнообразия (TTR) и регистровых контрастов."""

    lower_rate_self: float
    lower_rate_partner: float
    diff_lower_rate: float
    ent_self: float
    ent_partner: float
    diff_ent: float
    ttr_self: float
    ttr_partner: float
    diff_ttr: float
    diff_dot: float
    diff_upper_ratio: float
    mean_word_len_self: float
    diff_mean_word_len: float
