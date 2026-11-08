from dataclasses import dataclass

from project.features.base import BaseFeatureSchema


@dataclass(frozen=True)
class BayesianFeatures(BaseFeatureSchema):
    """7 признаков сглаженных байесовских вероятностей бот-шаблонов."""

    phrase_mean: float
    phrase_max: float
    pure_bot_count: float
    log_odds_sum: float
    greeting_prob: float
    farewell_prob: float
    beta_variance: float
