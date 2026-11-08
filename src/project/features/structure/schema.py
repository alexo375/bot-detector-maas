from dataclasses import dataclass

from project.features.base import BaseFeatureSchema


@dataclass(frozen=True)
class StructureFeatures(BaseFeatureSchema):
    """18 структурных признаков очередности реплик, асимметрии и эластичности длин ответов."""

    is_first_sender: float
    pos_p0: float
    pos_p1: float
    last_msg_len: float
    first_msg_len_self: float
    last_msg_len_self: float
    std_chars_per_msg: float
    consec_partner: float
    has_double_self: float
    has_double_partner: float
    diff_chars: float
    char_ratio: float
    diff_words: float
    diff_mean_len: float
    response_len_ratio_mean: float
    response_len_ratio_max: float
    turn_length_correlation: float
    is_even_dialogue: float
