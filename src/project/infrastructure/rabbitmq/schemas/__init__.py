from project.infrastructure.rabbitmq.schemas.dialogue_verdict import DialogueVerdict
from project.infrastructure.rabbitmq.schemas.event import DialogueEvent
from project.infrastructure.rabbitmq.schemas.task import InferenceTask

__all__ = [
    "DialogueEvent",
    "DialogueVerdict",
    "InferenceTask",
]
