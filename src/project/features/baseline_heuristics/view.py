from typing import Any

import emoji
import numpy as np

from project.features.base import BaseFeatureView
from project.features.baseline_heuristics.schema import BaselineFeatures
from project.features.common.patterns import (
    RE_CODE_PATTERN,
    RE_CYRILLIC,
    RE_ECHO_PATTERN,
    RE_LETTERS,
    RE_MARKDOWN_PATTERN,
    RE_WORDS_BOUNDED,
    SPAM_TEMPLATES,
)
from project.features.store.context import DialogueContext


class BaselineHeuristicsView(BaseFeatureView):
    """Извлекает 32 базовых количественных и шаблонно-эвристических признака участника."""

    schema = BaselineFeatures

    @staticmethod
    def _extract_lexicon_and_overlap(all_self_text: str, partner_all_text: str) -> dict[str, float]:
        letters = RE_LETTERS.findall(all_self_text)
        total_letters = len(letters)
        upper_ratio = sum(1 for c in letters if c.isupper()) / float(total_letters) if total_letters > 0 else 0.0
        cyr_ratio = len(RE_CYRILLIC.findall(all_self_text)) / float(total_letters) if total_letters > 0 else 0.0

        all_words_lower = [w.lower() for w in RE_WORDS_BOUNDED.findall(all_self_text)]
        repeat_ratio = 1.0 - (len(set(all_words_lower)) / max(len(all_words_lower), 1))

        self_words = set(all_words_lower)
        partner_words = {w.lower() for w in RE_WORDS_BOUNDED.findall(partner_all_text)}
        jaccard = float(len(self_words & partner_words) / max(len(self_words | partner_words), 1))

        emoji_count = float(sum(1 for char in all_self_text if emoji.is_emoji(char)))
        return {
            "emoji_count": emoji_count,
            "has_emoji": 1.0 if emoji_count > 0 else 0.0,
            "has_markdown": 1.0 if bool(RE_MARKDOWN_PATTERN.search(all_self_text)) else 0.0,
            "upper_ratio": float(upper_ratio),
            "cyr_ratio": float(cyr_ratio),
            "repeat_ratio": float(repeat_ratio),
            "has_code_syntax": 1.0 if bool(RE_CODE_PATTERN.search(all_self_text)) else 0.0,
            "has_echo_marker": 1.0 if bool(RE_ECHO_PATTERN.search(all_self_text)) else 0.0,
            "has_spam_template": 1.0 if any(tmpl in all_self_text.lower() for tmpl in SPAM_TEMPLATES) else 0.0,
            "jaccard_overlap": jaccard,
        }

    def extract_participant_features(
        self,
        messages: list[dict[str, Any]],
        p: int,
        context: DialogueContext,
    ) -> dict[str, float]:
        """Расчет базовых статистик объёма, пунктуации и шаблонов для участника p."""
        partner_p = 1 - p
        total_msgs = context.total_msgs
        self_msgs = context.p_msgs[p]
        partner_msgs = context.p_msgs[partner_p]
        n_self, n_partner = len(self_msgs), len(partner_msgs)

        words_per_msg = [len(RE_WORDS_BOUNDED.findall(t)) for t in self_msgs]
        chars_per_msg = [len(t) for t in self_msgs]
        partner_words_per_msg = [len(RE_WORDS_BOUNDED.findall(t)) for t in partner_msgs]

        mean_words = float(np.mean(words_per_msg)) if words_per_msg else 0.0
        partner_mean_words = float(np.mean(partner_words_per_msg)) if partner_words_per_msg else 0.0
        total_newlines = float(sum(t.count("\n") for t in self_msgs))
        dot_end_count = float(sum(1 for t in self_msgs if t.rstrip().endswith(".") and not t.rstrip().endswith("...")))
        total_questions = float(sum(t.count("?") for t in self_msgs))
        lex = self._extract_lexicon_and_overlap(context.p_text[p], context.p_text[partner_p])

        features = BaselineFeatures(
            is_last_sender=1.0 if context.last_sender == p else 0.0,
            n_messages=float(n_self),
            dialog_total_messages=float(total_msgs),
            msg_share=float(n_self / max(total_msgs, 1)),
            max_consecutive_msgs=float(context.consec.get(p, 0)),
            mean_words_per_msg=mean_words,
            max_words_per_msg=float(np.max(words_per_msg)) if words_per_msg else 0.0,
            total_words=float(np.sum(words_per_msg)) if words_per_msg else 0.0,
            mean_chars_per_msg=float(np.mean(chars_per_msg)) if chars_per_msg else 0.0,
            max_chars_per_msg=float(np.max(chars_per_msg)) if chars_per_msg else 0.0,
            total_chars=float(np.sum(chars_per_msg)) if chars_per_msg else 0.0,
            has_newline=1.0 if total_newlines > 0 else 0.0,
            newline_count=total_newlines,
            has_dot_end=1.0 if dot_end_count > 0 else 0.0,
            dot_end_count=dot_end_count,
            has_question=1.0 if total_questions > 0 else 0.0,
            question_count=total_questions,
            has_emoji=lex["has_emoji"],
            emoji_count=lex["emoji_count"],
            has_markdown=lex["has_markdown"],
            upper_ratio=lex["upper_ratio"],
            cyr_ratio=lex["cyr_ratio"],
            repeat_ratio=lex["repeat_ratio"],
            has_code_syntax=lex["has_code_syntax"],
            has_echo_marker=lex["has_echo_marker"],
            has_spam_template=lex["has_spam_template"],
            partner_mean_words=partner_mean_words,
            partner_n_messages=float(n_partner),
            mean_words_ratio=float(mean_words / (partner_mean_words + 1.0)),
            n_messages_ratio=float(n_self / (n_partner + 1.0)),
            jaccard_overlap=lex["jaccard_overlap"],
            partner_has_question=1.0 if any("?" in t for t in partner_msgs) else 0.0,
        )
        return features.to_dict()
