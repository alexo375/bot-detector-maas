from dataclasses import dataclass

from project.features.base import BaseFeatureSchema


@dataclass(frozen=True)
class TypographyFeatures(BaseFeatureSchema):
    """14 признаков разговорного сленга, скобочных смайлов и пунктуации."""

    slang_count_self: float
    slang_rate_self: float
    diff_slang: float
    bracket_smile_count: float
    bracket_smile_rate: float
    diff_bracket_smile_rate: float
    trailing_dot_rate: float
    diff_trailing_dot_rate: float
    comma_rate: float
    diff_comma_rate: float
    exclamation_rate: float
    ellipsis_count: float
    starts_lower_rate: float
    digit_ratio: float
