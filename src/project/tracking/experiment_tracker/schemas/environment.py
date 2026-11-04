from enum import StrEnum


class ExperimentEnvironment(StrEnum):
    """Окружение исполнения эксперимента."""

    OFFLINE = "offline"
    ONLINE = "online"
