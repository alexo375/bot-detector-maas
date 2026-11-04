import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, brier_score_loss, confusion_matrix, roc_curve


class GraphicsLogger:
    """Построение и логирование канонических графиков оценки классификатора в MLflow."""

    @staticmethod
    def save_and_log(fig: Figure, artifact_path: str, filename: str) -> None:
        """Сохранение фигуры в артефакты MLflow и закрытие объекта Matplotlib."""
        fig.tight_layout()
        mlflow.log_figure(fig, f"{artifact_path}/{filename}")
        plt.close(fig)

    @classmethod
    def log_confusion_matrix(
        cls,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        artifact_path: str = "graphics",
    ) -> None:
        """Построение и сохранение нормализованной матрицы ошибок."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Human (0)", "Bot (1)"],
            yticklabels=["Human (0)", "Bot (1)"],
        )
        for i in range(2):
            for j in range(2):
                ax.text(
                    j + 0.5,
                    i + 0.7,
                    f"({cm_norm[i, j]:.1%})",
                    ha="center",
                    va="center",
                    color="gray" if cm[i, j] < cm.max() / 2 else "white",
                    fontsize=9,
                )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        cls.save_and_log(fig, artifact_path=artifact_path, filename="confusion_matrix.png")

    @classmethod
    def log_roc_curve(
        cls,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        artifact_path: str = "graphics",
    ) -> None:
        """Построение и сохранение ROC-кривой."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")
        cls.save_and_log(fig, artifact_path=artifact_path, filename="roc_curve.png")

    @classmethod
    def log_calibration_curve(
        cls,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
        artifact_path: str = "graphics",
    ) -> float:
        """Построение кривой калибровки (Reliability Diagram)."""
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy="uniform")
        brier = float(brier_score_loss(y_true, y_pred_proba))

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(prob_pred, prob_true, "s-", label=f"Model (Brier = {brier:.4f})")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve (Reliability Diagram)")
        ax.legend(loc="lower right")
        cls.save_and_log(fig, artifact_path=artifact_path, filename="calibration_curve.png")
        return brier

    @classmethod
    def log_prediction_distribution(
        cls,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        artifact_path: str = "graphics",
    ) -> None:
        """Построение и сохранение распределения предсказанных вероятностей по классам."""
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = list(np.linspace(0, 1, 30))
        ax.hist(
            y_pred_proba[y_true == 0],
            bins=bins,
            alpha=0.5,
            color="blue",
            label="Human (0)",
            density=True,
        )
        ax.hist(
            y_pred_proba[y_true == 1],
            bins=bins,
            alpha=0.5,
            color="red",
            label="Bot (1)",
            density=True,
        )
        ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.5, label="Default threshold (0.5)")
        ax.set_xlabel("Predicted Probability of Bot")
        ax.set_ylabel("Density")
        ax.set_title("Prediction Probability Distribution by Class")
        ax.legend(loc="upper center")
        cls.save_and_log(fig, artifact_path=artifact_path, filename="prediction_distribution.png")


__all__ = ["GraphicsLogger"]
