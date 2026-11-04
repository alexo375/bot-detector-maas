"""Валидатор состояния и утилиты извлечения параметров для ExperimentTrackerBuilder."""

from typing import Any

from project.core.exceptions import DomainValidationError


class BuilderValidator:
    """Проверка инвариантов и валидация конфигурации билдера перед сборкой трекера."""

    @staticmethod
    def validate(
        run_name: str,
        require_description: bool,
        description: str | None,
        require_dataset: bool,
        dataset: Any,
        params: dict[str, Any] | None,
        require_metrics: bool,
        metrics: dict[str, Any] | None,
    ) -> None:
        """Строгая валидация обязательных полей MLOps рана."""
        errors: list[str] = []

        if require_description and (not description or not str(description).strip()):
            errors.append(
                "Не задано обязательное понятное описание эксперимента на русском языке: "
                "передайте описание в .set_description('...') или отключите через .require_description(False)"
            )

        if require_dataset:
            if dataset is None:
                errors.append("Не задан датасет: вызовите .set_dataset(data, name, targets) или .defer_dataset()")
            elif not dataset[1] or not str(dataset[1]).strip():
                errors.append("Имя датасета не может быть пустым: передайте непустое имя в .set_dataset(name=...)")

        if params is None:
            errors.append("Не заданы параметры модели: вызовите .set_params(model_family, ...)")
        elif not params.get("model_family") or not str(params.get("model_family")).strip():
            errors.append("Не указано обязательное семейство модели model_family в .set_params()")

        if require_metrics:
            if metrics is None:
                errors.append(
                    "Не заданы метрики: вызовите .set_metrics(val_logloss, ...) или отключите через .require_metrics(False)"
                )
            elif metrics.get("val_logloss") is None:
                errors.append("Не задана обязательная целевая метрика val_logloss в .set_metrics()")

        if errors:
            raise DomainValidationError(
                f"Ошибка валидации ExperimentTrackerBuilder для рана '{run_name}':\n"
                + "\n".join(f"  * {e}" for e in errors)
                + "\nТребования трекинга не выполнены."
            )


__all__ = ["BuilderValidator"]
