from enum import StrEnum


class TrackerRunStatus(StrEnum):
    """Статусы экспериментального запуска в MLflow."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
