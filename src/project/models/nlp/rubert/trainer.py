"""Сервис дообучения и извлечения эмбеддингов RuBERT-tiny2."""

import logging
from typing import Any

import mlflow.artifacts
import mlflow.transformers
import numpy as np
import polars as pl
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from project.core.exceptions import DomainValidationError
from project.models.nlp.rubert.dataset import DialogueTextDataset
from project.models.nlp.rubert.engine import RuBERTTrainingEngine
from project.tracking.experiment_tracker.builder import ExperimentTrackerBuilder
from project.tracking.experiment_tracker.schemas import ExperimentEnvironment

logger = logging.getLogger(__name__)


class RuBERTBackboneTrainer:
    """Сервис дообучения и версионирования RuBERT-tiny2 через MLflow. Принимает чистые Polars-объекты."""

    def __init__(
        self,
        model_name: str = "cointegrated/rubert-tiny2",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _resolve_checkpoint(self) -> tuple[Any, Any, float]:
        """Загрузка обученного чекпоинта из MLflow Model Registry или базовой модели."""
        for target_uri in ("models:/rubert_detector@champion", "models:/rubert_detector/latest"):
            try:
                loaded = mlflow.transformers.load_model(target_uri)
                tok = loaded.tokenizer
                mod = loaded.model.to(self.device)
                logger.info("Loaded fine-tuned RuBERT checkpoint from MLflow Model Registry (%s)", target_uri)
                return tok, mod, 1e-5
            except Exception as exc:  # noqa: BLE001
                logger.debug("MLflow Model Registry checkpoint lookup note for %s: %s", target_uri, exc)

        logger.info("No checkpoint in registry. Initializing RuBERT from base pretrained %s", self.model_name)
        tok = AutoTokenizer.from_pretrained(self.model_name)
        mod = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2).to(self.device)
        return tok, mod, 3e-5

    def extract_embeddings(
        self,
        model: Any,
        tokenizer: Any,
        texts: pl.DataFrame | pl.Series | list[str],
        batch_size: int = 64,
        max_length: int = 128,
    ) -> np.ndarray:
        """Извлечение 312-мерных эмбеддингов последнего скрытого слоя (CLS-токен)."""
        model.eval()
        if isinstance(texts, pl.DataFrame):
            raw_texts = texts["text"].to_list()
        elif isinstance(texts, pl.Series):
            raw_texts = texts.to_list()
        else:
            raw_texts = list(texts)

        embeddings_list = []
        for i in range(0, len(raw_texts), batch_size):
            batch_texts = raw_texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = model.bert(**enc)
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings_list.append(cls_emb)

        return np.vstack(embeddings_list)

    @staticmethod
    def _validate_train_df(train_data: pl.DataFrame) -> tuple[str, str]:
        if not isinstance(train_data, pl.DataFrame):
            raise TypeError(f"train_data должен быть экземпляром polars.DataFrame, получено: {type(train_data)}")
        target_col = (
            "is_bot" if "is_bot" in train_data.columns else ("label" if "label" in train_data.columns else None)
        )
        if target_col is None:
            raise DomainValidationError(
                f"В train_data не найдена колонка целевой метки ('is_bot' или 'label'). Доступные: {train_data.columns}"
            )
        group_col = (
            "dialog_id" if "dialog_id" in train_data.columns else ("group" if "group" in train_data.columns else None)
        )
        if group_col is None:
            raise DomainValidationError(
                f"В train_data не найдена колонка группы диалога ('dialog_id' или 'group'). Доступные: {train_data.columns}"
            )
        if "text" not in train_data.columns:
            raise DomainValidationError(
                f"В train_data не найдена колонка текстов реплик 'text'. Доступные: {train_data.columns}"
            )
        return target_col, group_col

    def _run_training_epoch_loop(
        self,
        tokenizer: Any,
        model: Any,
        train_texts: list[str],
        y_train: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        max_length: int,
        batch_size: int,
        actual_lr: float,
        weight_decay: float,
        max_epochs: int,
        patience: int,
        log_to_mlflow: bool,
    ) -> dict[str, Any]:
        logger.info("Запуск обучения RuBERT на %d примерах (lr=%.2e)...", len(train_idx), actual_lr)
        enc_train = tokenizer(
            [train_texts[int(i)] for i in train_idx],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        train_ds = DialogueTextDataset(
            enc_train["input_ids"], enc_train["attention_mask"], torch.tensor(y_train[train_idx], dtype=torch.long)
        )
        enc_val = tokenizer(
            [train_texts[int(i)] for i in val_idx],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        val_ds = DialogueTextDataset(
            enc_val["input_ids"], enc_val["attention_mask"], torch.tensor(y_train[val_idx], dtype=torch.long)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=weight_decay)
        total_steps = max(1, len(train_loader) * max_epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
        )
        eval_metrics, _ = RuBERTTrainingEngine.train_and_validate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=max_epochs,
            patience=patience,
            device=self.device,
            log_to_mlflow=log_to_mlflow,
        )
        self.model_, self.tokenizer_ = model, tokenizer
        return eval_metrics

    def fit(
        self,
        train_data: pl.DataFrame,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 64,
        lr: float = 3e-5,
        weight_decay: float = 0.01,
        max_length: int = 128,
        seed: int = 42,
        experiment_name: str = "bot-detection",
        registered_model_name: str | None = "rubert_detector",
        environment: ExperimentEnvironment = ExperimentEnvironment.OFFLINE,
        val_split_indices: tuple[np.ndarray, np.ndarray] | None = None,
        log_to_mlflow: bool = True,
    ) -> dict[str, Any]:
        """Дообучение RuBERT-tiny2 на Polars DataFrame с валидацией и логированием в MLflow."""
        from project.models.blueprints import RUBERT_BLUEPRINT

        target_col, group_col = self._validate_train_df(train_data)
        train_texts = train_data["text"].to_list()
        y_train, groups = train_data[target_col].to_numpy().astype(int), train_data[group_col].to_numpy()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        tokenizer, model, resolved_lr = self._resolve_checkpoint()
        actual_lr = lr if lr != 3e-5 else resolved_lr
        if val_split_indices is not None:
            train_idx, val_idx = val_split_indices
        else:
            train_idx, val_idx = next(
                StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
                    train_texts, y_train, groups=groups
                )
            )

        if not log_to_mlflow:
            eval_metrics = self._run_training_epoch_loop(
                tokenizer,
                model,
                train_texts,
                y_train,
                train_idx,
                val_idx,
                max_length,
                batch_size,
                actual_lr,
                weight_decay,
                max_epochs,
                patience,
                False,
            )
            return {
                "stage": "RuBERT-tiny2 Fine-Tuning (Fold)",
                "run_id": "",
                "best_epoch": eval_metrics["best_epoch"],
                "final_val_logloss": eval_metrics["final_val_loss"],
                "final_val_auc": eval_metrics["final_val_auc"],
                "final_val_acc": eval_metrics["final_val_acc"],
                "final_brier_score": eval_metrics["final_brier"],
            }

        tracker = (
            ExperimentTrackerBuilder(
                run_name=RUBERT_BLUEPRINT.run_name,
                experiment_name=experiment_name,
                tags=RUBERT_BLUEPRINT.tags,
                description=RUBERT_BLUEPRINT.description,
            )
            .set_environment(environment)
            .set_dataset(data=train_data.to_pandas(), name="rubert_dialogue_texts", targets=y_train)
            .set_params(
                model_family="Transformer (RuBERT-tiny2)",
                pretrained_model=self.model_name,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                lr=actual_lr,
                weight_decay=weight_decay,
                max_length=max_length,
                seed=seed,
            )
            .defer_metrics()
            .require_graphics(False)
            .require_feature_tree(False)
            .require_submission(False)
            .build()
        )
        with tracker:
            eval_metrics = self._run_training_epoch_loop(
                tokenizer,
                model,
                train_texts,
                y_train,
                train_idx,
                val_idx,
                max_length,
                batch_size,
                actual_lr,
                weight_decay,
                max_epochs,
                patience,
                True,
            )
            mlflow_model_kwargs: dict[str, Any] = {
                "transformers_model": {"model": model, "tokenizer": tokenizer},
                "artifact_path": "model",
                "task": "text-classification",
            }
            if registered_model_name:
                mlflow_model_kwargs["registered_model_name"] = registered_model_name
            mlflow.transformers.log_model(**mlflow_model_kwargs)
            tracker.log_metrics(
                val_logloss=eval_metrics["final_val_loss"],
                val_roc_auc=eval_metrics["final_val_auc"],
                val_accuracy=eval_metrics["final_val_acc"],
            )
            active_run = mlflow.active_run()
            return {
                "stage": "RuBERT-tiny2 Fine-Tuning",
                "run_id": active_run.info.run_id if active_run is not None else "",
                "best_epoch": eval_metrics["best_epoch"],
                "final_val_logloss": eval_metrics["final_val_loss"],
                "final_val_auc": eval_metrics["final_val_auc"],
                "final_val_acc": eval_metrics["final_val_acc"],
                "final_brier_score": eval_metrics["final_brier"],
            }

    def predict_proba(self, texts: list[str], batch_size: int = 64, max_length: int = 128) -> np.ndarray:
        """Инференс вероятностей класса 1 (bot) для списка текстов (без локального Triton-фолбэка)."""
        if not texts:
            return np.empty(0, dtype=np.float32)

        if getattr(self, "model_", None) is None or getattr(self, "tokenizer_", None) is None:
            self.tokenizer_, self.model_, _ = self._resolve_checkpoint()

        self.model_.to(self.device)
        self.model_.eval()

        probs_list: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                enc = self.tokenizer_(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model_(**enc).logits
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                probs_list.append(probs)

        return np.concatenate(probs_list).astype(np.float32) if probs_list else np.empty(0, dtype=np.float32)


__all__ = ["RuBERTBackboneTrainer"]
