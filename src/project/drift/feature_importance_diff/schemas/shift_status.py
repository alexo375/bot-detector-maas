from enum import StrEnum


class FeatureShiftStatus(StrEnum):
    """Статусы динамики ранга и веса признака между итерациями переобучения."""

    SURGING = "SURGING"
    DROPPING = "DROPPING"
    STABLE = "STABLE"
    NEW = "NEW"
