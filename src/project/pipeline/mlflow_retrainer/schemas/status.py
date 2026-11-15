from enum import StrEnum


class FlowStatus(StrEnum):
    """Статусы завершения регулярного Flow переобучения."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
