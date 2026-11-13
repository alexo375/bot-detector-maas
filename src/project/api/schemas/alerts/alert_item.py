from pydantic import BaseModel, Field


class AlertItem(BaseModel):
    """Схема единичного оповещения от Prometheus Alertmanager."""

    status: str = Field(..., description="Статус алерта (firing или resolved)")
    labels: dict[str, str] = Field(
        default_factory=dict, description="Метки алерта (alertname, severity, service, job, env, instance)"
    )
    annotations: dict[str, str] = Field(
        default_factory=dict, description="Аннотации алерта (summary, description, runbook_url)"
    )
    startsAt: str | None = Field(None, description="Время начала срабатывания алерта в формате ISO 8601")
    endsAt: str | None = Field(None, description="Время окончания/резолва алерта в формате ISO 8601")
    generatorURL: str | None = Field(None, description="URL источника правила в Prometheus")
    fingerprint: str | None = Field(None, description="Уникальный хеш-идентификатор алерта")
