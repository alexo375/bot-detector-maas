"""HTTP REST KServe v2 транспорт к Triton Inference Server."""

from typing import Any

import httpx
import numpy as np

from project.core.config import config
from project.core.exceptions import TritonServerError
from project.core.logging import ProjectLogger
from project.infrastructure.triton.transport.base import BaseTritonTransport

logger = ProjectLogger.get_logger(__name__)


class HttpTritonTransport(BaseTritonTransport):
    """Сетевой HTTP REST KServe v2 транспорт к нативному ансамблю Triton."""

    def __init__(
        self,
        base_url: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
    ) -> None:
        resolved_url = (base_url or config.triton.url).rstrip("/")
        if not resolved_url.startswith("http://") and not resolved_url.startswith("https://"):
            resolved_url = f"http://{resolved_url}"
        self.base_url = resolved_url
        self.client = client or httpx.Client(timeout=timeout)

    def infer(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Отправка KServe v2 HTTP JSON запроса в ансамбль `bot_ensemble`."""
        n_rows, n_cols = features.shape
        url = f"{self.base_url}/v2/models/bot_ensemble/infer"
        features_f32 = np.asarray(features, dtype=np.float32)

        payload: dict[str, Any] = {
            "inputs": [
                {
                    "name": "features",
                    "shape": [n_rows, n_cols],
                    "datatype": "FP32",
                    "data": features_f32.flatten().tolist(),
                }
            ]
        }

        try:
            resp = self.client.post(url, json=payload)
            resp.raise_for_status()
            res_json = resp.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            raise TritonServerError(f"Сбой HTTP-запроса к Triton ({url}): {exc}") from exc

        outputs_map: dict[str, np.ndarray] = {}
        for out in res_json.get("outputs", []):
            name = out.get("name")
            out_shape = out.get("shape", [n_rows, 2])
            arr = np.array(out.get("data", []), dtype=np.float32).reshape(out_shape)
            outputs_map[name] = arr

        meta_probs = outputs_map.get("probabilities")
        if meta_probs is None:
            raise ValueError("Output 'probabilities' not found in Triton ensemble response")

        cb_probs = outputs_map.get("cb_probabilities")
        if cb_probs is None:
            raise ValueError("Output 'cb_probabilities' not found in Triton ensemble response")
        lgb_probs = outputs_map.get("lgb_probabilities")
        if lgb_probs is None:
            raise ValueError("Output 'lgb_probabilities' not found in Triton ensemble response")
        xgb_probs = outputs_map.get("xgb_probabilities")
        if xgb_probs is None:
            raise ValueError("Output 'xgb_probabilities' not found in Triton ensemble response")

        p_meta = meta_probs[:, 1].astype(np.float32)
        p_cb = cb_probs[:, 1].astype(np.float32)
        p_lgb = lgb_probs[:, 1].astype(np.float32)
        p_xgb = xgb_probs[:, 1].astype(np.float32)

        return p_meta, p_cb, p_lgb, p_xgb

    def is_healthy(self) -> bool:
        """Проверка готовности сервера и ансамблевой модели bot_ensemble через KServe v2 HTTP."""
        try:
            resp = self.client.get(f"{self.base_url}/v2/health/ready")
            if resp.status_code != 200:
                return False
            model_resp = self.client.get(f"{self.base_url}/v2/models/bot_ensemble/ready")
            return model_resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def close(self) -> None:
        """Закрытие HTTP сессии."""
        self.client.close()


__all__ = ["HttpTritonTransport"]
