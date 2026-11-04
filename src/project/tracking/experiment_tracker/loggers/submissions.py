import mlflow
import pandas as pd
import polars as pl


class SubmissionLogger:
    """Логер CSV-файла предсказаний (submission.csv) напрямую в артефакты MLflow."""

    @staticmethod
    def log(
        submission: pl.DataFrame | pd.DataFrame,
        filename: str = "submission.csv",
        artifact_path: str | None = None,
    ) -> None:
        """Сохранение таблицы сабмишена в текстовые артефакты MLflow без промежуточных файлов."""
        csv_content = submission.write_csv() if isinstance(submission, pl.DataFrame) else submission.to_csv(index=False)
        target_path = f"{artifact_path}/{filename}" if artifact_path else filename
        mlflow.log_text(csv_content, target_path)


__all__ = ["SubmissionLogger"]
