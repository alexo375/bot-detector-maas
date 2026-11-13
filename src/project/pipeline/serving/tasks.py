"""Prefect-задачи для публикации и перезагрузки моделей в Triton."""

from prefect import task

from project.core.logging import ProjectLogger
from project.pipeline.serving.publisher import ModelPublisher
from project.pipeline.serving.reloader import TritonReloader

logger = ProjectLogger.get_logger(__name__)


@task(name="export_onnx_and_sync_triton")
def export_onnx_task() -> dict[str, int]:
    """Экспорт переобученных моделей в ONNX и синхронизация репозитория Triton."""
    logger.info("Syncing @champion models from MLflow into Triton repository...")
    publisher = ModelPublisher()
    return publisher.publish_models()


@task(name="triton_hot_reload")
def hot_reload_triton_task() -> dict[str, bool]:
    """Отправка команды Hot Reload в Triton Server."""
    logger.info("Triggering Zero-Downtime Hot Reload in Triton...")
    results = TritonReloader().reload_all()
    logger.info("Triton Hot Reload response: %s", results)
    return results
