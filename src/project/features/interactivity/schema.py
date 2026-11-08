from dataclasses import dataclass

from project.features.base import BaseFeatureSchema


@dataclass(frozen=True)
class InteractivityFeatures(BaseFeatureSchema):
    """11 признаков эхо-цитирования, связности реплик и проверок собеседника на бота."""

    echo_all: float
    echo_non_greet: float
    has_echo_non_greet: float
    fuzzy_echo: float
    mean_coupling: float
    std_coupling: float
    partner_probing_count: float
    self_probing_count: float
    diff_probing_count: float
    probing_response_expansion: float
    question_answer_rate: float
