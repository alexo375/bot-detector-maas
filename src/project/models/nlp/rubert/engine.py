"""Движок эпохового обучения и валидации моделей RuBERT на PyTorch."""

import logging
from typing import Any

import mlflow
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class RuBERTTrainingEngine:
    """Инкапсулирует тренировочный цикл, расчет градиентов и валидацию для RuBERT."""

    @staticmethod
    def train_epoch(
        model: Any,
        train_loader: DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
    ) -> float:
        """Обучение модели на одной эпохе с отсечением градиентов."""
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for batch in train_loader:
            optimizer.zero_grad()
            b_ids = batch["input_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)

            outputs = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss_sum += float(loss.item()) * len(b_labels)
            train_samples += len(b_labels)

        return train_loss_sum / max(1, train_samples)

    @staticmethod
    def evaluate(
        model: Any,
        val_loader: DataLoader[Any],
        device: torch.device,
    ) -> tuple[float, float, float, float]:
        """Валидация модели на отложенной выборке: logloss, roc_auc, accuracy, brier."""
        model.eval()
        val_preds: list[float] = []
        val_targets: list[int] = []
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                b_ids = batch["input_ids"].to(device)
                b_mask = batch["attention_mask"].to(device)
                b_labels = batch["labels"].to(device)

                outputs = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
                val_loss_sum += float(outputs.loss.item()) * len(b_labels)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(b_labels.cpu().numpy())

        cur_val_loss = val_loss_sum / max(1, len(val_targets))
        cur_val_auc = float(roc_auc_score(val_targets, val_preds))
        cur_val_acc = float(accuracy_score(val_targets, [p > 0.5 for p in val_preds]))
        cur_brier = float(brier_score_loss(val_targets, val_preds))

        return cur_val_loss, cur_val_auc, cur_val_acc, cur_brier

    @classmethod
    def train_and_validate(
        cls,
        model: Any,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        max_epochs: int,
        patience: int,
        device: torch.device,
        log_to_mlflow: bool = True,
    ) -> tuple[dict[str, Any], Any]:
        """Полный цикл обучения с ранней остановкой и сохранением лучших весов."""
        best_val_loss = float("inf")
        best_epoch = 0
        no_improve = 0
        final_val_loss = 0.0
        final_val_auc = 0.0
        final_val_acc = 0.0
        final_brier = 0.0
        best_model_state = None

        for epoch in range(1, max_epochs + 1):
            train_loss_epoch = cls.train_epoch(model, train_loader, optimizer, scheduler, device)
            cur_val_loss, cur_val_auc, cur_val_acc, cur_brier = cls.evaluate(model, val_loader, device)

            if log_to_mlflow and mlflow.active_run() is not None:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss_epoch,
                        "val_logloss": cur_val_loss,
                        "val_roc_auc": cur_val_auc,
                        "val_accuracy": cur_val_acc,
                        "brier_score": cur_brier,
                    },
                    step=epoch,
                )

            if cur_val_loss < best_val_loss:
                best_val_loss = cur_val_loss
                best_epoch = epoch
                no_improve = 0
                final_val_loss = cur_val_loss
                final_val_auc = cur_val_auc
                final_val_acc = cur_val_acc
                final_brier = cur_brier
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Ранняя остановка на эпохе %d. Лучшая: %d", epoch, best_epoch)
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        metrics = {
            "best_epoch": best_epoch,
            "final_val_loss": final_val_loss,
            "final_val_auc": final_val_auc,
            "final_val_acc": final_val_acc,
            "final_brier": final_brier,
        }
        return metrics, best_model_state


__all__ = ["RuBERTTrainingEngine"]
