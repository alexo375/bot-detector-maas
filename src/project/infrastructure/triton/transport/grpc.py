"""gRPC KServe v2 транспорт к Triton Inference Server (бинарный Zero-Copy)."""

from typing import Any

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from project.core.config import config
from project.core.exceptions import TritonServerError
from project.core.logging import ProjectLogger
from project.infrastructure.triton.transport.base import BaseTritonTransport

logger = ProjectLogger.get_logger(__name__)


class GrpcTritonTransport(BaseTritonTransport):
    """Сетевой gRPC транспорт к нативному ансамблю Triton (Zero-Copy тензоры)."""

    def __init__(
        self,
        server_url: str | None = None,
        grpc_client: Any | None = None,
        timeout: float = 10.0,
    ) -> None:
        resolved_url = server_url or config.triton.grpc_url
        clean_url = resolved_url.replace("http://", "").replace("https://", "").replace("grpc://", "")
        self.server_url = clean_url
        self.timeout = timeout
        self.client = grpc_client or grpcclient.InferenceServerClient(
            url=self.server_url,
            verbose=False,
        )

    def infer(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Отправка бинарного KServe v2 gRPC запроса в ансамбль `bot_ensemble`."""
        n_rows, n_cols = features.shape

        inputs = [grpcclient.InferInput("features", [n_rows, n_cols], "FP32")]
        inputs[0].set_data_from_numpy(features.astype(np.float32))

        outputs = [
            grpcclient.InferRequestedOutput("probabilities"),
            grpcclient.InferRequestedOutput("cb_probabilities"),
            grpcclient.InferRequestedOutput("lgb_probabilities"),
            grpcclient.InferRequestedOutput("xgb_probabilities"),
        ]

        try:
            response = self.client.infer(
                model_name="bot_ensemble",
                inputs=inputs,
                outputs=outputs,
                client_timeout=self.timeout,
            )
        except (InferenceServerException, Exception) as exc:
            raise TritonServerError(f"Сбой gRPC-запроса к Triton ({self.server_url}): {exc}") from exc

        meta_probs = response.as_numpy("probabilities")
        if meta_probs is None:
            raise ValueError("Output 'probabilities' not found in Triton ensemble response")

        cb_probs = response.as_numpy("cb_probabilities")
        if cb_probs is None:
            raise ValueError("Output 'cb_probabilities' not found in Triton ensemble response")
        lgb_probs = response.as_numpy("lgb_probabilities")
        if lgb_probs is None:
            raise ValueError("Output 'lgb_probabilities' not found in Triton ensemble response")
        xgb_probs = response.as_numpy("xgb_probabilities")
        if xgb_probs is None:
            raise ValueError("Output 'xgb_probabilities' not found in Triton ensemble response")

        p_meta = meta_probs[:, 1].astype(np.float32)
        p_cb = cb_probs[:, 1].astype(np.float32)
        p_lgb = lgb_probs[:, 1].astype(np.float32)
        p_xgb = xgb_probs[:, 1].astype(np.float32)

        return p_meta, p_cb, p_lgb, p_xgb

    def is_healthy(self) -> bool:
        """Проверка доступности gRPC сервера и готовности ансамблевой модели bot_ensemble."""
        try:
            return bool(
                self.client.is_server_ready(client_timeout=self.timeout)
                and self.client.is_model_ready("bot_ensemble", client_timeout=self.timeout)
            )
        except (InferenceServerException, OSError):
            return False

    def close(self) -> None:
        """Закрытие сетевых ресурсов gRPC канала."""
        if hasattr(self.client, "close"):
            self.client.close()


__all__ = ["GrpcTritonTransport"]
