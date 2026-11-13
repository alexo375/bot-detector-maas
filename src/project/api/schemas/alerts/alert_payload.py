from pydantic import BaseModel, Field

from project.api.schemas.alerts.alert_item import AlertItem


class AlertmanagerWebhookPayload(BaseModel):
    """Модель входящего вебхука от Prometheus Alertmanager."""

    receiver: str = Field(..., description="Имя ресивера Alertmanager (например, ntfy)")
    status: str = Field(..., description="Глобальный статус группы алертов (firing / resolved)")
    alerts: list[AlertItem] = Field(default_factory=list, description="Список сработавших алертов")
    groupLabels: dict[str, str] = Field(default_factory=dict, description="Метки группы алертов")
    commonLabels: dict[str, str] = Field(default_factory=dict, description="Общие метки для всей группы алертов")
    commonAnnotations: dict[str, str] = Field(
        default_factory=dict, description="Общие аннотации для всей группы алертов"
    )
    externalURL: str | None = Field(None, description="Внешний URL Alertmanager")
    version: str | None = Field(None, description="Версия формата вебхука Alertmanager (например, 4)")
    groupKey: str | None = Field(None, description="Ключ группировки алертов")
    truncatedAlerts: int = Field(0, description="Количество усеченных алертов в пейлоаде")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "receiver": "ntfy",
                    "status": "firing",
                    "alerts": [
                        {
                            "status": "firing",
                            "labels": {
                                "alertname": "HighDataDriftDetected",
                                "severity": "warning",
                                "service": "bot-detector-gateway",
                                "env": "production",
                                "instance": "api-gateway:8080",
                                "job": "api-gateway",
                            },
                            "annotations": {
                                "summary": "Feature Data Drift share exceeds 20%",
                                "description": "Statistical data drift detected across input features (share: 64.21%). Model retraining recommended.",
                            },
                            "startsAt": "2026-11-14T16:27:41.073Z",
                            "endsAt": "0001-01-01T00:00:00Z",
                            "generatorURL": "http://prometheus:9090/graph?g0.expr=bot_detector_data_drift_share+%3E+0.2",
                            "fingerprint": "4e3e5b6f02c4c134",
                        }
                    ],
                    "groupLabels": {
                        "alertname": "HighDataDriftDetected",
                        "severity": "warning",
                    },
                    "commonLabels": {
                        "alertname": "HighDataDriftDetected",
                        "severity": "warning",
                        "service": "bot-detector-gateway",
                        "env": "production",
                        "instance": "api-gateway:8080",
                        "job": "api-gateway",
                    },
                    "commonAnnotations": {
                        "summary": "Feature Data Drift share exceeds 20%",
                        "description": "Statistical data drift detected across input features (share: 64.21%). Model retraining recommended.",
                    },
                    "externalURL": "http://alertmanager:9093",
                    "version": "4",
                    "groupKey": '{}:{alertname="HighDataDriftDetected", severity="warning"}',
                    "truncatedAlerts": 0,
                }
            ]
        }
    }
