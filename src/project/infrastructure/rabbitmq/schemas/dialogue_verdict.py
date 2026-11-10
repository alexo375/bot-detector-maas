from enum import StrEnum


class DialogueVerdict(StrEnum):
    """Итоговый вердикт по обоим участникам диалога."""

    BOTH_BOTS = "both_bots"
    P0_BOT = "p0_bot"
    P1_BOT = "p1_bot"
    BOTH_HUMANS = "both_humans"
    UNKNOWN = "unknown"
