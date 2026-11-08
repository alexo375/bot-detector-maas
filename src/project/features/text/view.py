from typing import Any

import numpy as np

from project.features.base import BaseFeatureView
from project.features.common.patterns import (
    RE_LETTERS,
    RE_WORDS_BOUNDED,
    TextPatternAnalyzer,
)
from project.features.store.context import DialogueContext
from project.features.text.schema import TextFeatures


class TextMetricsView(BaseFeatureView):
    """Извлекает 13 признаков энтропии Шеннона, Type-Token Ratio и контрастов длин слов."""

    schema = TextFeatures

    def extract_participant_features(
        self,
        messages: list[dict[str, Any]],
        p: int,
        context: DialogueContext,
    ) -> dict[str, float]:
        """Расчет информационно-лексических метрик текста для участника p."""
        partner_p = 1 - p
        p_msgs = context.p_msgs[p]
        partner_msgs = context.p_msgs[partner_p]
        n_self = len(p_msgs)
        n_partner = len(partner_msgs)

        p_text = context.p_text[p]
        partner_text = context.p_text[partner_p]

        lower_msgs_self = sum(1 for m in p_msgs if m == m.lower() and any(c.isalpha() for c in m))
        lower_msgs_partner = sum(1 for m in partner_msgs if m == m.lower() and any(c.isalpha() for c in m))
        lower_rate_self = lower_msgs_self / max(1, n_self)
        lower_rate_partner = lower_msgs_partner / max(1, n_partner)
        diff_lower_rate = lower_rate_self - lower_rate_partner

        letters_self = RE_LETTERS.findall(p_text)
        letters_partner = RE_LETTERS.findall(partner_text)
        upper_self = sum(1 for c in letters_self if c.isupper()) / max(1, len(letters_self))
        upper_partner = sum(1 for c in letters_partner if c.isupper()) / max(1, len(letters_partner))
        diff_upper_ratio = float(upper_self - upper_partner)

        ent_self = TextPatternAnalyzer.shannon_entropy(p_text)
        ent_partner = TextPatternAnalyzer.shannon_entropy(partner_text)
        diff_ent = ent_self - ent_partner

        dot_self = p_text.count(".") / max(1, n_self)
        dot_partner = partner_text.count(".") / max(1, n_partner)
        diff_dot = dot_self - dot_partner

        words_self = RE_WORDS_BOUNDED.findall(p_text.lower())
        words_partner = RE_WORDS_BOUNDED.findall(partner_text.lower())
        ttr_self = len(set(words_self)) / max(1, len(words_self))
        ttr_partner = len(set(words_partner)) / max(1, len(words_partner))
        diff_ttr = ttr_self - ttr_partner

        mean_word_len_self = float(np.mean([len(w) for w in words_self])) if words_self else 0.0
        mean_word_len_partner = float(np.mean([len(w) for w in words_partner])) if words_partner else 0.0
        diff_mean_word_len = mean_word_len_self - mean_word_len_partner

        features = TextFeatures(
            lower_rate_self=lower_rate_self,
            lower_rate_partner=lower_rate_partner,
            diff_lower_rate=diff_lower_rate,
            ent_self=ent_self,
            ent_partner=ent_partner,
            diff_ent=diff_ent,
            ttr_self=ttr_self,
            ttr_partner=ttr_partner,
            diff_ttr=diff_ttr,
            diff_dot=diff_dot,
            diff_upper_ratio=diff_upper_ratio,
            mean_word_len_self=mean_word_len_self,
            diff_mean_word_len=diff_mean_word_len,
        )
        return features.to_dict()
