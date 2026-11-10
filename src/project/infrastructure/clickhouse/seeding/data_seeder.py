"""Сервис начальной инициализации и наполнения ClickHouse DWH."""

import json
import logging
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from project import paths
from project.core.config import config
from project.core.logging import LoggingConfigurator
from project.infrastructure.clickhouse.dwh import FeatureDWH
from project.infrastructure.clickhouse.seeding.feature_seeder import FeatureSnapshotSeeder
from project.models import PairwisePostprocessor

logger = logging.getLogger(__name__)


class ClickHouseDataSeeder:
    """Сервис начальной инициализации и наполнения ClickHouse DWH."""

    def __init__(self, dwh: FeatureDWH | None = None) -> None:
        self.dwh = dwh or FeatureDWH()
        self.feature_seeder = FeatureSnapshotSeeder(self.dwh)

    def run_full_seed(self) -> dict[str, Any]:
        """Выполняет полный цикл инициализации DWH."""
        logger.info("--- Этап 1: Применение миграций схемы DWH ---")
        applied = self.dwh.init_schema()
        logger.info("Миграции: %s", applied if applied else "схема актуальна")

        logger.info("--- Этап 2: Загрузка сырых датасетов (ODS/Bronze) ---")
        raw_counts = self.dwh.seed_raw_datasets(
            train_json_path=paths.raw.train_json,
            test_json_path=paths.raw.test_json,
            ytrain_csv_path=paths.raw.ytrain_csv,
            ytest_csv_path=paths.raw.ytest_csv,
        )
        logger.info("Сырые диалоги в DWH: %s", raw_counts)

        logger.info("--- Этап 3: Расчет и версионирование срезов признаков (Feature Store) ---")
        self.seed_feature_snapshots()

        logger.info("--- Этап 4: Генерация и сохранение OOF/Test предсказаний (Predictions Layer) ---")
        self.seed_model_predictions()

        status = self.dwh.status()
        logger.info("Статус наполнения DWH: %s", status)
        return status

    def seed_feature_snapshots(self) -> None:
        """Расчет и сохранение срезов baseline_features и candidate_features."""
        self.feature_seeder.seed_snapshots()

    def _compute_and_save_run_predictions(
        self,
        client: Any,
        name: str,
        info: dict[str, str],
        df_feats_tr: Any,
        df_feats_te: Any,
        ytrain: Any,
        ytest: Any,
        postprocessor: Any,
    ) -> None:
        import catboost as cb
        import lightgbm as lgb
        import xgboost as xgb

        run_id = str(info["run_id"])
        logger.info("Расчет OOF-предсказаний для %s (run_id=%s)...", name, run_id)
        art_path = client.download_artifacts(run_id, "feature_tree.json")
        with open(art_path, encoding="utf-8") as f:
            selected_feats = [item["name"] for item in json.load(f)["features"]]

        y_train = ytrain["is_bot"].to_numpy()
        x_tr, x_te = df_feats_tr.select(selected_feats).to_numpy(), df_feats_te.select(selected_feats).to_numpy()
        oof_raw = np.zeros(len(y_train), dtype=np.float32)
        test_preds_folds = np.zeros(len(ytest), dtype=np.float32)
        fold_indices = np.zeros(len(y_train), dtype=np.uint8)
        params = client.get_run(run_id).data.params
        cv = GroupKFold(n_splits=5)

        for fold_idx, (idx_tr, idx_va) in enumerate(cv.split(x_tr, y_train, groups=ytrain["dialog_id"].to_numpy())):
            fold_indices[idx_va] = fold_idx
            x_fold_tr, y_fold_tr, x_fold_va = x_tr[idx_tr], y_train[idx_tr], x_tr[idx_va]
            if info["family"] == "CatBoost":
                model: Any = cb.CatBoostClassifier(
                    iterations=int(params.get("iterations", 300)),
                    learning_rate=float(params.get("learning_rate", 0.05)),
                    depth=int(params.get("depth", 6)),
                    verbose=0,
                    random_seed=42,
                    allow_writing_files=False,
                )
            elif info["family"] == "LightGBM":
                model = lgb.LGBMClassifier(
                    n_estimators=int(params.get("n_estimators", 300)),
                    learning_rate=float(params.get("learning_rate", 0.05)),
                    max_depth=int(params.get("max_depth", 6)),
                    random_state=42,
                    verbose=-1,
                )
            else:
                model = xgb.XGBClassifier(
                    n_estimators=int(params.get("n_estimators", 300)),
                    learning_rate=float(params.get("learning_rate", 0.05)),
                    max_depth=int(params.get("max_depth", 6)),
                    random_state=42,
                    eval_metric="logloss",
                )
            model.fit(x_fold_tr, y_fold_tr)
            oof_raw[idx_va] = model.predict_proba(x_fold_va)[:, 1]
            test_preds_folds += model.predict_proba(x_te)[:, 1] / cv.n_splits

        d_ids_tr, p_idxs_tr = ytrain["dialog_id"].to_numpy(), ytrain["participant_index"].to_numpy()
        d_ids_te, p_idxs_te = ytest["dialog_id"].to_numpy(), ytest["participant_index"].to_numpy()
        oof_cal = postprocessor.transform(oof_raw, dialog_ids=d_ids_tr, participant_indices=p_idxs_tr)
        test_cal = postprocessor.transform(test_preds_folds, dialog_ids=d_ids_te, participant_indices=p_idxs_te)
        self.dwh.predictions.save_predictions(
            run_id=run_id,
            run_name=name,
            dataset_split="train_oof",
            dialog_ids=d_ids_tr,
            participant_indices=p_idxs_tr,
            prob_bot=oof_raw,
            prob_bot_calibrated=oof_cal,
            fold_indices=fold_indices,
            is_bot=y_train,
        )
        self.dwh.predictions.save_predictions(
            run_id=run_id,
            run_name=name,
            dataset_split="test",
            dialog_ids=d_ids_te,
            participant_indices=p_idxs_te,
            prob_bot=test_preds_folds,
            prob_bot_calibrated=test_cal,
        )

    def seed_model_predictions(self) -> None:
        """Обучение CV фолдов и строгое сохранение OOF/Test предсказаний из MLflow в ClickHouse."""
        import mlflow

        tracking_uri = config.mlflow_tracking_uri
        client = mlflow.tracking.MlflowClient(tracking_uri)
        experiments = client.search_experiments()

        target_specs = {
            "catboost_top_pruned": "CatBoost",
            "lightgbm_top_pruned": "LightGBM",
            "xgboost_top_pruned": "XGBoost",
        }
        exp_ids = [e.experiment_id for e in experiments]
        model_runs: dict[str, dict[str, str]] = {}
        if exp_ids:
            for run_name, family in target_specs.items():
                found = client.search_runs(
                    experiment_ids=exp_ids,
                    filter_string=f"attributes.run_name = '{run_name}'",
                    order_by=["attributes.start_time DESC"],
                    max_results=1,
                )
                if found:
                    model_runs[run_name] = {"run_id": found[0].info.run_id, "family": family}

        if not model_runs:
            raise RuntimeError(
                f"Expected MLflow runs ({', '.join(target_specs)}) not found at {tracking_uri} — "
                "seed failed without starter ONNX fallback."
            )

        runs_to_compute = [
            (name, info)
            for name, info in model_runs.items()
            if not self.dwh.predictions.has_predictions(str(info["run_id"]), "train_oof")
        ]
        if not runs_to_compute:
            logger.info("Все OOF-предсказания уже присутствуют в DWH, пропуск пересчета.")
            return

        ytrain, ytest = self.dwh.raw.load_labels("train"), self.dwh.raw.load_labels("test")
        df_feats_tr, _ = self.dwh.features.load_snapshot("candidate_features", dataset_type="train")
        df_feats_te, _ = self.dwh.features.load_snapshot("candidate_features", dataset_type="test")
        postprocessor = PairwisePostprocessor()
        for name, info in runs_to_compute:
            self._compute_and_save_run_predictions(
                client, name, info, df_feats_tr, df_feats_te, ytrain, ytest, postprocessor
            )


__all__ = ["ClickHouseDataSeeder"]

if __name__ == "__main__":
    LoggingConfigurator.configure()
    ClickHouseDataSeeder().run_full_seed()
