from dataclasses import dataclass

from project.features.base import BaseFeatureSchema


@dataclass(frozen=True)
class BaselineFeatures(BaseFeatureSchema):
    """32 базовых эвристических признака реплик и диалога."""

    is_last_sender: float
    n_messages: float
    dialog_total_messages: float
    msg_share: float
    max_consecutive_msgs: float
    mean_words_per_msg: float
    max_words_per_msg: float
    total_words: float
    mean_chars_per_msg: float
    max_chars_per_msg: float
    total_chars: float
    has_newline: float
    newline_count: float
    has_dot_end: float
    dot_end_count: float
    has_question: float
    question_count: float
    has_emoji: float
    emoji_count: float
    has_markdown: float
    upper_ratio: float
    cyr_ratio: float
    repeat_ratio: float
    has_code_syntax: float
    has_echo_marker: float
    has_spam_template: float
    partner_mean_words: float
    partner_n_messages: float
    mean_words_ratio: float
    n_messages_ratio: float
    jaccard_overlap: float
    partner_has_question: float
