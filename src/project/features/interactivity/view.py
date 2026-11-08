from typing import Any

import numpy as np

from project.features.base import BaseFeatureView
from project.features.common.patterns import RE_PROBING_PATTERN
from project.features.interactivity.schema import InteractivityFeatures
from project.features.store.context import DialogueContext

_GREETING_PHRASES = frozenset(
    {"привет", "здравствуйте", "добрый день", "добрый вечер", "доброе утро", "хай", "прив", "ку", "hi", "hello", "hey"}
)


class InteractivityView(BaseFeatureView):
    """Извлекает 11 признаков эхо-цитирования, связности длин и реакции на проверки на бота."""

    schema = InteractivityFeatures

    def extract_participant_features(
        self,
        messages: list[dict[str, Any]],
        p: int,
        context: DialogueContext,
    ) -> dict[str, float]:
        """Расчет динамики парных реакций, эха и ответов на провокации для участника p."""
        partner_p = 1 - p
        total_msgs = context.total_msgs

        echo_all = 0.0
        echo_non_greet = 0.0
        fuzzy_echo = 0.0
        coupling_ratios: list[float] = []
        probing_expansions: list[float] = []
        partner_q_turns = 0
        answered_q_turns = 0

        for i in range(1, total_msgs):
            prev = messages[i - 1]
            curr = messages[i]
            pt = (prev.get("text") or "").strip().lower()
            ct = (curr.get("text") or "").strip().lower()

            if curr.get("participant_index") == p and prev.get("participant_index") == partner_p:
                if pt == ct and pt:
                    echo_all += 1.0
                    if len(pt) >= 2 and pt not in _GREETING_PHRASES:
                        echo_non_greet += 1.0

                s_p = {pt[j : j + 3] for j in range(len(pt) - 2)}
                s_c = {ct[j : j + 3] for j in range(len(ct) - 2)}
                if ((s_p or s_c) and (len(s_p & s_c) / max(1, len(s_p | s_c)) >= 0.6)) or (pt and ct and pt == ct):
                    fuzzy_echo += 1.0

                ratio = len(ct) / (len(pt) + 1.0)
                coupling_ratios.append(ratio)

                if RE_PROBING_PATTERN.search(pt):
                    probing_expansions.append(ratio)

                if "?" in pt:
                    partner_q_turns += 1
                    if "?" not in ct and len(ct) > 0:
                        answered_q_turns += 1

        has_echo_non_greet = 1.0 if echo_non_greet > 0.0 else 0.0
        mean_coupling = float(np.mean(coupling_ratios)) if coupling_ratios else 1.0
        std_coupling = float(np.std(coupling_ratios)) if coupling_ratios else 0.0

        p_msgs = context.p_msgs[p]
        partner_msgs = context.p_msgs[partner_p]

        self_probing_count = float(sum(1 for m in p_msgs if RE_PROBING_PATTERN.search(m)))
        partner_probing_count = float(sum(1 for m in partner_msgs if RE_PROBING_PATTERN.search(m)))
        diff_probing_count = self_probing_count - partner_probing_count

        probing_response_expansion = float(np.mean(probing_expansions)) if probing_expansions else 1.0
        question_answer_rate = float(answered_q_turns / max(1, partner_q_turns))

        features = InteractivityFeatures(
            echo_all=echo_all,
            echo_non_greet=echo_non_greet,
            has_echo_non_greet=has_echo_non_greet,
            fuzzy_echo=fuzzy_echo,
            mean_coupling=mean_coupling,
            std_coupling=std_coupling,
            partner_probing_count=partner_probing_count,
            self_probing_count=self_probing_count,
            diff_probing_count=diff_probing_count,
            probing_response_expansion=probing_response_expansion,
            question_answer_rate=question_answer_rate,
        )
        return features.to_dict()
