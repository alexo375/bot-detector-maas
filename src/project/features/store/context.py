from dataclasses import dataclass


@dataclass
class DialogueContext:
    """Контекст диалоговой сессии, предварительно вычисляемый для Feature Views."""

    total_msgs: int
    last_sender: int
    last_msg_len: float
    pos_val: dict[int, float]
    p_msgs: dict[int, list[str]]
    p_text: dict[int, str]
    consec: dict[int, int]
