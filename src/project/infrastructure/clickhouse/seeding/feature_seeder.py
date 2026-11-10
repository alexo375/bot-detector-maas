"""Модуль генерации и сохранения срезов признаков для ClickHouse Feature Store."""

import logging
from typing import TYPE_CHECKING

import polars as pl

from project.features import (
    BaselineHeuristicsView,
    BayesianTemplateView,
    CharTFIDFConfig,
    CharTFIDFView,
    ChatTypographyView,
    DialogueStructureView,
    FeatureStore,
    InteractivityView,
    TextMetricsView,
    TFIDFConfig,
    TFIDFView,
)

if TYPE_CHECKING:
    from project.infrastructure.clickhouse.dwh import FeatureDWH

logger = logging.getLogger(__name__)


class FeatureSnapshotSeeder:
    """Сервис вычисления и сохранения срезов baseline_features и candidate_features."""

    def __init__(self, dwh: "FeatureDWH") -> None:
        self.dwh = dwh

    def seed_snapshots(self) -> None:
        """Расчет и сохранение срезов baseline_features и candidate_features в DWH."""
        ytrain = self.dwh.raw.load_labels("train")
        ytest = self.dwh.raw.load_labels("test")
        train_dlgs = self.dwh.raw.load_dialogues_dict("train")
        test_dlgs = self.dwh.raw.load_dialogues_dict("test")

        train_dialogue_list = [train_dlgs[d_id] for d_id in ytrain["dialog_id"].unique(maintain_order=True)]
        test_dialogue_list = [test_dlgs[d_id] for d_id in ytest["dialog_id"].unique(maintain_order=True)]

        # 1. Базовые эвристики ('baseline_features')
        if not self.dwh.features.has_snapshot("baseline_features", "train"):
            logger.info("Формирование среза 'baseline_features'...")
            fs_base = FeatureStore(views=[BaselineHeuristicsView()])
            base_cols = list(fs_base.default_features)
            x_tr_b = fs_base.transform(train_dialogue_list)
            x_te_b = fs_base.transform(test_dialogue_list)

            df_tr_b = pl.DataFrame(
                {
                    "dialog_id": ytrain["dialog_id"].to_list(),
                    "participant_index": ytrain["participant_index"].to_list(),
                    **{col: x_tr_b[:, i].tolist() for i, col in enumerate(base_cols)},
                }
            )
            df_te_b = pl.DataFrame(
                {
                    "dialog_id": ytest["dialog_id"].to_list(),
                    "participant_index": ytest["participant_index"].to_list(),
                    **{col: x_te_b[:, i].tolist() for i, col in enumerate(base_cols)},
                }
            )
            self.dwh.features.save_snapshot(
                snapshot_tag="baseline_features",
                features_df=df_tr_b,
                dataset_type="train",
                description="32 базовые эвристики реплик и диалогов",
            )
            self.dwh.features.save_snapshot(
                snapshot_tag="baseline_features",
                features_df=df_te_b,
                dataset_type="test",
                description="32 базовые эвристики реплик и диалогов",
            )

        # 2. Полный пул кандидатов признаков ('candidate_features')
        if not self.dwh.features.has_snapshot("candidate_features", "train"):
            logger.info("Формирование среза 'candidate_features'...")
            feature_store = FeatureStore(
                views=[
                    BaselineHeuristicsView(),
                    DialogueStructureView(),
                    TextMetricsView(),
                    ChatTypographyView(),
                    InteractivityView(),
                    BayesianTemplateView(alpha=3.0, min_len=4, min_count=2, n_splits=5),
                    TFIDFView(
                        config=TFIDFConfig(max_features=120, ngram_min=1, ngram_max=2, min_df=2, svd_components=15)
                    ),
                    CharTFIDFView(
                        config=CharTFIDFConfig(
                            max_features=15000,
                            ngram_min=3,
                            ngram_max=5,
                            min_df=2,
                            n_components=35,
                        )
                    ),
                ]
            )
            cand_names = list(feature_store.default_features)
            x_full_tr = feature_store.fit_transform(
                train_dialogue_list,
                y=ytrain["is_bot"].to_numpy().astype(int),
                groups=ytrain["dialog_id"].to_numpy(),
            )
            x_full_te = feature_store.transform(test_dialogue_list)

            df_tr_cand = pl.DataFrame(
                {
                    "dialog_id": ytrain["dialog_id"].to_list(),
                    "participant_index": ytrain["participant_index"].to_list(),
                    **{col: x_full_tr[:, i].tolist() for i, col in enumerate(cand_names)},
                }
            )
            df_te_cand = pl.DataFrame(
                {
                    "dialog_id": ytest["dialog_id"].to_list(),
                    "participant_index": ytest["participant_index"].to_list(),
                    **{col: x_full_te[:, i].tolist() for i, col in enumerate(cand_names)},
                }
            )

            self.dwh.features.save_snapshot(
                snapshot_tag="candidate_features",
                features_df=df_tr_cand,
                dataset_type="train",
                description="Единый пул кандидатов признаков",
            )
            self.dwh.features.save_snapshot(
                snapshot_tag="candidate_features",
                features_df=df_te_cand,
                dataset_type="test",
                description="Единый пул кандидатов признаков",
            )


__all__ = ["FeatureSnapshotSeeder"]
