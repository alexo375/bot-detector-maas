"""Интерактивный диалог выбора версии модели для отката через Prefect UI."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, cast

from prefect.flow_runs import pause_flow_run
from prefect.input import RunInput
from pydantic import Field, create_model

from project.pipeline.model_rollback.schemas import ModelVersionCandidate


class RollbackPromptUI:
    """Интерактивный выбор совместимой версии модели через Prefect UI RunInput."""

    @staticmethod
    def prompt_target_version(
        candidates: Sequence[ModelVersionCandidate],
        timeout_seconds: int = 900,
    ) -> str:
        """Отображение формы выбора версии в UI Prefect с валидацией ввода."""
        if not candidates:
            raise ValueError("В MLflow Model Registry отсутствуют совместимые версии для отката.")

        option_members = {f"v{c.version}": c.label for c in candidates}
        dynamic_enum = cast("type[StrEnum]", StrEnum("RollbackVersionOption", option_members))
        default_option = dynamic_enum(candidates[min(1, len(candidates) - 1)].label)

        input_model = cast(
            "type[RunInput]",
            create_model(
                "SelectRollbackVersionInput",
                __base__=RunInput,
                target_version=(
                    dynamic_enum,
                    Field(default=default_option, description="Выберите целевую версию модели из MLflow"),
                ),
            ),
        )

        selection: Any = pause_flow_run(wait_for_input=input_model, timeout=timeout_seconds)
        if selection is None:
            raise RuntimeError("Откат отменен: не выбран целевой вариант в форме Resume.")

        raw_val = str(selection.target_version)
        for candidate in candidates:
            if candidate.label == raw_val or f"v{candidate.version}" == raw_val:
                return candidate.version

        return raw_val.split("|", 1)[0].strip().lstrip("v")


__all__ = ["RollbackPromptUI"]
