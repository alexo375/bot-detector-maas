from enum import StrEnum


class GatekeeperDecision(StrEnum):
    """Решение гейткипера о валидации и промоушене модели Challenger."""

    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"
