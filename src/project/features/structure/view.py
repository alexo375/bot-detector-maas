from typing import Any

import numpy as np

from project.features.base import BaseFeatureView
from project.features.common.patterns import RE_WORDS_BOUNDED
from project.features.store.context import DialogueContext
from project.features.structure.schema import StructureFeatures


class DialogueStructureView(BaseFeatureView):
    """Извлекает 18 признаков позиционной структуры, серий реплик и эластичности длин ответов."""

    schema = StructureFeatures

    @staticmethod
    def _compute_response_dynamics(
        messages: list[dict[str, Any]],
        p: int,
        partner_p: int,
    ) -> tuple[float, float, float]:
        prompt_lens: list[float] = []
        reply_lens: list[float] = []
        response_ratios: list[float] = []
        for i in range(1, len(messages)):
            prev, curr = messages[i - 1], messages[i]
            if curr.get("participant_index") == p and prev.get("participant_index") == partner_p:
                l_prev = float(len(prev.get("text") or ""))
                l_curr = float(len(curr.get("text") or ""))
                prompt_lens.append(l_prev)
                reply_lens.append(l_curr)
                response_ratios.append((l_curr + 1.0) / (l_prev + 1.0))

        ratio_mean = float(np.mean(response_ratios)) if response_ratios else 1.0
        ratio_max = float(np.max(response_ratios)) if response_ratios else 1.0
        corr = 0.0
        if len(prompt_lens) >= 2 and np.std(prompt_lens) > 1e-6 and np.std(reply_lens) > 1e-6:
            corr_val = float(np.corrcoef(prompt_lens, reply_lens)[0, 1])
            corr = 0.0 if np.isnan(corr_val) else corr_val
        return ratio_mean, ratio_max, corr

    def extract_participant_features(
        self,
        messages: list[dict[str, Any]],
        p: int,
        context: DialogueContext,
    ) -> dict[str, float]:
        """Расчет структурных и динамических признаков длин реплик для участника p."""
        total_msgs = context.total_msgs
        partner_p = 1 - p
        p_msgs, partner_msgs = context.p_msgs[p], context.p_msgs[partner_p]
        chars_self = [len(m) for m in p_msgs]
        chars_partner = [len(m) for m in partner_msgs]
        total_chars, partner_chars = float(sum(chars_self)), float(sum(chars_partner))
        words_self = sum(len(RE_WORDS_BOUNDED.findall(m)) for m in p_msgs)
        words_partner = sum(len(RE_WORDS_BOUNDED.findall(m)) for m in partner_msgs)
        ratio_mean, ratio_max, turn_corr = self._compute_response_dynamics(messages, p, partner_p)
        consec_self, consec_partner = float(context.consec[p]), float(context.consec[partner_p])

        features = StructureFeatures(
            is_first_sender=1.0 if (total_msgs > 0 and messages[0].get("participant_index", -1) == p) else 0.0,
            pos_p0=float(context.pos_val[0]) if p == 0 else 0.0,
            pos_p1=float(context.pos_val[1]) if p == 1 else 0.0,
            last_msg_len=context.last_msg_len,
            first_msg_len_self=float(chars_self[0]) if chars_self else 0.0,
            last_msg_len_self=float(chars_self[-1]) if chars_self else 0.0,
            std_chars_per_msg=float(np.std(chars_self)) if len(chars_self) > 1 else 0.0,
            consec_partner=consec_partner,
            has_double_self=1.0 if consec_self >= 2 else 0.0,
            has_double_partner=1.0 if consec_partner >= 2 else 0.0,
            diff_chars=total_chars - partner_chars,
            char_ratio=(total_chars + 1.0) / (partner_chars + 1.0),
            diff_words=float(words_self - words_partner),
            diff_mean_len=(total_chars / max(1, len(p_msgs))) - (partner_chars / max(1, len(partner_msgs))),
            response_len_ratio_mean=ratio_mean,
            response_len_ratio_max=ratio_max,
            turn_length_correlation=turn_corr,
            is_even_dialogue=1.0 if (total_msgs > 0 and total_msgs % 2 == 0) else 0.0,
        )
        return features.to_dict()
