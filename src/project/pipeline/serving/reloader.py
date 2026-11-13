import logging

import httpx

from project.core.config import config
from project.infrastructure.triton.version_manager import TritonModelVersionManager

logger = logging.getLogger(__name__)


class TritonReloader:
    """Клиент для выполнения перезагрузки моделей в NVIDIA Triton Inference Server без простоя."""

    DEFAULT_MODELS: tuple[str, ...] = (
        "catboost_detector",
        "lightgbm_detector",
        "xgboost_detector",
        "meta_stacker",
        "bot_ensemble",
    )

    def __init__(
        self,
        triton_url: str | None = None,
        timeout_sec: float = 10.0,
        version_manager: TritonModelVersionManager | None = None,
    ) -> None:
        self.base_url = triton_url or config.triton.url
        self.timeout_sec = timeout_sec
        self.version_manager = version_manager or TritonModelVersionManager()

    def reload_model(self, model_name: str) -> bool:
        """Отправка POST запроса в Triton для перезагрузки конкретной модели с валидацией версии."""
        active_version = self.version_manager.get_latest_version(model_name)
        logger.info(
            "Triggering Triton reload for '%s' (active version in repository: %s)...",
            model_name,
            active_version,
        )
        base = self.base_url
        if not base.startswith(("http://", "https://")):
            base = f"http://{base}"
        reload_url = f"{base}/v2/repository/models/{model_name}/load"
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.post(reload_url)
        except Exception as e:
            raise RuntimeError(f"Could not reach Triton at {reload_url} to reload '{model_name}': {e}") from e

        if resp.status_code == 200:
            logger.info("Successfully reloaded Triton model '%s' (200 OK)", model_name)
            return True
        if resp.status_code == 400 and "polling is enabled" in resp.text:
            logger.info(
                "Triton polling mode is active; model '%s' (version %s) is automatically hot-reloaded by server",
                model_name,
                active_version,
            )
            return True
        raise RuntimeError(f"Triton returned status {resp.status_code} when reloading '{model_name}': {resp.text}")

    def reload_all(self, model_names: list[str] | None = None) -> dict[str, bool]:
        """Перезагрузка списка продуктовых моделей в Triton."""
        models = model_names or list(self.DEFAULT_MODELS)
        return {m: self.reload_model(m) for m in models}
