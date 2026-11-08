from typing import Any

from project.features.base import BaseFeatureView
from project.features.common.patterns import (
    RE_BRACKET_SMILE,
    RE_ELLIPSIS,
    RE_WORDS_BOUNDED,
    SLANG_MARKERS,
)
from project.features.store.context import DialogueContext
from project.features.typography.schema import TypographyFeatures


class ChatTypographyView(BaseFeatureView):
    """Извлекает 14 маркеров живого чат-общения: сленг рунета, скобочные смайлики и пунктуацию."""

    schema = TypographyFeatures

    def extract_participant_features(
        self,
        messages: list[dict[str, Any]],
        p: int,
        context: DialogueContext,
    ) -> dict[str, float]:
        """Расчет показателей неформальной орфографии и пунктуации для участника p."""
        partner_p = 1 - p
        p_msgs = context.p_msgs[p]
        partner_msgs = context.p_msgs[partner_p]
        n_self = len(p_msgs)
        n_partner = len(partner_msgs)

        p_text = context.p_text[p]
        partner_text = context.p_text[partner_p]

        words_self = [w.lower() for w in RE_WORDS_BOUNDED.findall(p_text)]
        words_partner = [w.lower() for w in RE_WORDS_BOUNDED.findall(partner_text)]

        slang_count_self = float(sum(1 for w in words_self if w in SLANG_MARKERS))
        slang_count_partner = float(sum(1 for w in words_partner if w in SLANG_MARKERS))
        slang_rate_self = slang_count_self / max(1, len(words_self))
        slang_rate_partner = slang_count_partner / max(1, len(words_partner))
        diff_slang = slang_rate_self - slang_rate_partner

        bracket_matches_self = [bool(RE_BRACKET_SMILE.search(m)) for m in p_msgs]
        bracket_matches_partner = [bool(RE_BRACKET_SMILE.search(m)) for m in partner_msgs]
        bracket_smile_count = float(len(RE_BRACKET_SMILE.findall(p_text)))
        bracket_smile_rate = sum(bracket_matches_self) / max(1, n_self)
        bracket_smile_rate_partner = sum(bracket_matches_partner) / max(1, n_partner)
        diff_bracket_smile_rate = bracket_smile_rate - bracket_smile_rate_partner

        trailing_dots_self = sum(1 for m in p_msgs if m.rstrip().endswith(".") and not m.rstrip().endswith(".."))
        trailing_dots_partner = sum(
            1 for m in partner_msgs if m.rstrip().endswith(".") and not m.rstrip().endswith("..")
        )
        trailing_dot_rate = trailing_dots_self / max(1, n_self)
        trailing_dot_rate_partner = trailing_dots_partner / max(1, n_partner)
        diff_trailing_dot_rate = trailing_dot_rate - trailing_dot_rate_partner

        comma_rate = p_text.count(",") / max(1, len(words_self))
        comma_rate_partner = partner_text.count(",") / max(1, len(words_partner))
        diff_comma_rate = comma_rate - comma_rate_partner

        exclamation_rate = p_text.count("!") / max(1, n_self)
        ellipsis_count = float(len(RE_ELLIPSIS.findall(p_text)))

        starts_lower_cnt = sum(1 for m in p_msgs if m.lstrip() and m.lstrip()[0].isalpha() and m.lstrip()[0].islower())
        starts_lower_rate = starts_lower_cnt / max(1, n_self)

        digit_cnt = sum(1 for c in p_text if c.isdigit())
        digit_ratio = digit_cnt / max(1, len(p_text))

        features = TypographyFeatures(
            slang_count_self=slang_count_self,
            slang_rate_self=slang_rate_self,
            diff_slang=diff_slang,
            bracket_smile_count=bracket_smile_count,
            bracket_smile_rate=bracket_smile_rate,
            diff_bracket_smile_rate=diff_bracket_smile_rate,
            trailing_dot_rate=trailing_dot_rate,
            diff_trailing_dot_rate=diff_trailing_dot_rate,
            comma_rate=comma_rate,
            diff_comma_rate=diff_comma_rate,
            exclamation_rate=exclamation_rate,
            ellipsis_count=ellipsis_count,
            starts_lower_rate=starts_lower_rate,
            digit_ratio=digit_ratio,
        )
        return features.to_dict()
