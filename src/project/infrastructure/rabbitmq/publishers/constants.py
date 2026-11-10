from project.core.config import config

QUEUE_AUDIT_EVENTS: str = config.rabbitmq.queue_audit_events
QUEUE_INFERENCE_TASKS: str = config.rabbitmq.queue_inference_tasks
