from project.drift.feature_importance_diff.schemas import (
    FeatureImportanceDiffReport,
    FeatureImportanceRecord,
    FeatureShiftStatus,
)


class FeatureImportanceDiffAnalyzer:
    """Анализатор смещения важности признаков между версиями моделей."""

    def __init__(
        self,
        reference_importance: dict[str, float] | None = None,
    ) -> None:
        self.reference_importance = reference_importance or {}

    @staticmethod
    def _normalize_map(imp_map: dict[str, float]) -> dict[str, float]:
        total = sum(abs(float(v)) for v in imp_map.values())
        if total <= 0.0:
            return dict(imp_map)
        return {k: float(v) / total for k, v in imp_map.items()}

    def analyze(
        self,
        current_importance: dict[str, float],
        top_n: int = 52,
        normalize: bool = False,
    ) -> FeatureImportanceDiffReport:
        """Сравнение важности текущих признаков с эталоном."""
        ref_importance: dict[str, float] = (
            self._normalize_map(self.reference_importance) if normalize else dict(self.reference_importance)
        )
        cur_importance: dict[str, float] = (
            self._normalize_map(current_importance) if normalize else dict(current_importance)
        )

        sorted_current = sorted(cur_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        sorted_ref = sorted(ref_importance.items(), key=lambda x: x[1], reverse=True)
        ref_ranks = {feat: idx + 1 for idx, (feat, _) in enumerate(sorted_ref)}

        records: list[FeatureImportanceRecord] = []
        for cur_rank, (feat, cur_val) in enumerate(sorted_current, start=1):
            prev_rank = ref_ranks.get(feat)
            prev_val = ref_importance.get(feat, 0.0)
            val_delta = cur_val - prev_val

            if prev_rank is None:
                rank_shift = 0
                status = FeatureShiftStatus.NEW
            else:
                rank_shift = prev_rank - cur_rank
                if rank_shift >= 3:
                    status = FeatureShiftStatus.SURGING
                elif rank_shift <= -3:
                    status = FeatureShiftStatus.DROPPING
                else:
                    status = FeatureShiftStatus.STABLE

            records.append(
                FeatureImportanceRecord(
                    feature=feat,
                    current_rank=cur_rank,
                    previous_rank=prev_rank,
                    rank_shift=rank_shift,
                    current_importance=round(cur_val, 4),
                    previous_importance=round(prev_val, 4),
                    importance_delta=round(val_delta, 4),
                    status=status,
                )
            )

        current_features_set = {f for f, _ in sorted_current}
        dropped_features = [
            {
                "feature": f,
                "previous_rank": r,
                "previous_importance": ref_importance[f],
                "status": "DROPPED",
            }
            for f, r in ref_ranks.items()
            if r <= top_n and f not in current_features_set
        ]

        return FeatureImportanceDiffReport.create(
            top_n=top_n,
            features=records,
            dropped_features=dropped_features,
        )
