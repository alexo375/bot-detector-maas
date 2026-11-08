from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FeatureMetadata:
    """Descriptive metadata for an engineered dialogue feature."""

    name: str
    display_name_en: str
    display_name_ru: str
    group: Literal["structural", "interactivity", "typography", "lexical", "probabilistic"]


_GROUP_ASSIGNMENTS: dict[str, Literal["structural", "interactivity", "typography", "lexical", "probabilistic"]] = {
    "is_last_sender": "interactivity",
    "is_first_sender": "interactivity",
    "pos_p0": "interactivity",
    "pos_p1": "interactivity",
    "consec_partner": "interactivity",
    "has_double_self": "interactivity",
    "has_double_partner": "interactivity",
    "turn_length_correlation": "interactivity",
    "is_even_dialogue": "interactivity",
    "question_answer_rate": "interactivity",
    "mean_coupling": "interactivity",
    "std_coupling": "interactivity",
    "partner_probing_count": "interactivity",
    "self_probing_count": "interactivity",
    "diff_probing_count": "interactivity",
    "probing_response_expansion": "interactivity",
    "n_messages": "structural",
    "dialog_total_messages": "structural",
    "msg_share": "structural",
    "max_consecutive_msgs": "structural",
    "mean_words_per_msg": "structural",
    "max_words_per_msg": "structural",
    "total_words": "structural",
    "mean_chars_per_msg": "structural",
    "max_chars_per_msg": "structural",
    "total_chars": "structural",
    "last_msg_len": "structural",
    "first_msg_len_self": "structural",
    "last_msg_len_self": "structural",
    "std_chars_per_msg": "structural",
    "diff_chars": "structural",
    "char_ratio": "structural",
    "diff_words": "structural",
    "diff_mean_len": "structural",
    "response_len_ratio_mean": "structural",
    "response_len_ratio_max": "structural",
    "has_newline": "typography",
    "newline_count": "typography",
    "has_dot_end": "typography",
    "dot_end_count": "typography",
    "has_question": "typography",
    "question_count": "typography",
    "has_emoji": "typography",
    "emoji_count": "typography",
    "has_markdown": "typography",
    "upper_ratio": "typography",
    "cyr_ratio": "typography",
    "repeat_ratio": "typography",
    "has_code_syntax": "typography",
    "diff_dot": "typography",
    "diff_upper_ratio": "typography",
    "slang_count_self": "typography",
    "slang_rate_self": "typography",
    "diff_slang": "typography",
    "bracket_smile_count": "typography",
    "bracket_smile_rate": "typography",
    "diff_bracket_smile_rate": "typography",
    "trailing_dot_rate": "typography",
    "diff_trailing_dot_rate": "typography",
    "comma_rate": "typography",
    "diff_comma_rate": "typography",
    "exclamation_rate": "typography",
    "ellipsis_count": "typography",
    "starts_lower_rate": "typography",
    "digit_ratio": "typography",
    "lower_rate_self": "typography",
    "lower_rate_partner": "typography",
    "diff_lower_rate": "typography",
    "jaccard_overlap": "lexical",
    "partner_mean_words": "lexical",
    "partner_n_messages": "lexical",
    "mean_words_ratio": "lexical",
    "n_messages_ratio": "lexical",
    "partner_has_question": "lexical",
    "ent_self": "lexical",
    "ent_partner": "lexical",
    "diff_ent": "lexical",
    "ttr_self": "lexical",
    "ttr_partner": "lexical",
    "diff_ttr": "lexical",
    "mean_word_len_self": "lexical",
    "diff_mean_word_len": "lexical",
    "has_echo_marker": "probabilistic",
    "has_spam_template": "probabilistic",
    "echo_all": "probabilistic",
    "echo_non_greet": "probabilistic",
    "has_echo_non_greet": "probabilistic",
    "fuzzy_echo": "probabilistic",
    "phrase_mean": "probabilistic",
    "phrase_max": "probabilistic",
    "pure_bot_count": "probabilistic",
    "log_odds_sum": "probabilistic",
    "greeting_prob": "probabilistic",
    "farewell_prob": "probabilistic",
    "beta_variance": "probabilistic",
}


class FeatureMetadataRegistry:
    """Реестр метаданных и человекочитаемых названий признаков диалогов."""

    @staticmethod
    def _build_entry(name: str) -> FeatureMetadata:
        group = _GROUP_ASSIGNMENTS.get(name, "structural")
        clean_en = name.replace("_", " ").title()
        clean_ru = name.replace("_", " ").capitalize()
        return FeatureMetadata(
            name=name,
            display_name_en=clean_en,
            display_name_ru=clean_ru,
            group=group,
        )

    @classmethod
    def get_metadata(cls, feature_name: str) -> FeatureMetadata:
        """Retrieve metadata entry for given feature name, falling back to dynamic entry if new."""
        if feature_name in FEATURE_REGISTRY:
            return FEATURE_REGISTRY[feature_name]
        return cls._build_entry(feature_name)

    @classmethod
    def get_display_name(cls, feature_name: str, lang: Literal["en", "ru"] = "en") -> str:
        """Get clean human-readable display name in specified language."""
        meta = cls.get_metadata(feature_name)
        return meta.display_name_en if lang == "en" else meta.display_name_ru


FEATURE_REGISTRY: dict[str, FeatureMetadata] = {
    name: FeatureMetadataRegistry._build_entry(name) for name in _GROUP_ASSIGNMENTS
}

__all__ = [
    "FEATURE_REGISTRY",
    "FeatureMetadata",
    "FeatureMetadataRegistry",
]
