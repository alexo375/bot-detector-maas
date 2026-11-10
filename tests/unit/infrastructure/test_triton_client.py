"""Модульные тесты для Triton HTTP-клиента и схем результатов инференса."""

from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from project.core.schemas import DialogueResultItem, ParticipantVerdict
from project.infrastructure.triton.client import TritonEnsembleClient
from project.infrastructure.triton.transport.http import HttpTritonTransport


def test_triton_client_url_formatting() -> None:
    """Проверка нормализации URL-адреса Triton сервера при инициализации."""
    c1 = TritonEnsembleClient(triton_url="localhost:8000/", protocol="http")
    assert isinstance(c1.transport, HttpTritonTransport)
    assert c1.transport.base_url == "http://localhost:8000"

    c2 = TritonEnsembleClient(triton_url="https://triton.prod:8000", protocol="http")
    assert isinstance(c2.transport, HttpTritonTransport)
    assert c2.transport.base_url == "https://triton.prod:8000"


def test_triton_client_infer_success() -> None:
    """Проверка успешного выполнения сетевого запроса KServe v2 к Triton."""
    mock_http = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "outputs": [
            {
                "name": "probabilities",
                "shape": [2, 2],
                "data": [0.8, 0.2, 0.1, 0.9],
            },
            {
                "name": "cb_probabilities",
                "shape": [2, 2],
                "data": [0.75, 0.25, 0.15, 0.85],
            },
            {
                "name": "lgb_probabilities",
                "shape": [2, 2],
                "data": [0.70, 0.30, 0.20, 0.80],
            },
            {
                "name": "xgb_probabilities",
                "shape": [2, 2],
                "data": [0.85, 0.15, 0.05, 0.95],
            },
        ]
    }
    mock_http.post.return_value = mock_resp

    client = TritonEnsembleClient(triton_url="http://test-triton:8000", client=mock_http)
    dummy_feats = np.ones((2, 10), dtype=np.float32)

    p_meta, p_cb, p_lgb, p_xgb = client.infer(dummy_feats)

    assert p_meta.shape == (2,)
    assert np.isclose(p_meta[0], 0.2, atol=1e-5)
    assert np.isclose(p_meta[1], 0.9, atol=1e-5)
    assert np.isclose(p_cb[1], 0.85, atol=1e-5)
    assert np.isclose(p_lgb[1], 0.80, atol=1e-5)
    assert np.isclose(p_xgb[1], 0.95, atol=1e-5)
    mock_resp.raise_for_status.assert_called_once()


def test_triton_client_infer_missing_probabilities_raises() -> None:
    """Проверка генерации исключения при отсутствии тензора probabilities в ответе."""
    mock_http = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"outputs": []}
    mock_http.post.return_value = mock_resp

    client = TritonEnsembleClient(triton_url="http://test-triton:8000", client=mock_http)
    dummy_feats = np.ones((2, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="Output 'probabilities' not found"):
        client.infer(dummy_feats)


def test_triton_client_infer_fails_on_partial_outputs() -> None:
    """Проверка строгого fail-fast при отсутствии выходов базовых моделей в ответе Triton."""
    mock_http = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "outputs": [
            {
                "name": "probabilities",
                "shape": [1, 2],
                "data": [0.3, 0.7],
            }
        ]
    }
    mock_http.post.return_value = mock_resp

    client = TritonEnsembleClient(triton_url="http://test-triton:8000", client=mock_http)
    dummy_feats = np.ones((1, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="Output 'cb_probabilities' not found"):
        client.infer(dummy_feats)


def test_core_schemas_dialogue_result_and_verdict() -> None:
    """Проверка сериализации и фабрики DialogueResultItem и ParticipantVerdict из core.schemas."""
    verdict = ParticipantVerdict(
        participant_index=1,
        bot_probability=0.88,
        is_bot=True,
    )
    result = DialogueResultItem.from_inference(
        dialog_id="dlg-1",
        dialogue_length=5,
        verdicts=[verdict],
        cached=False,
        latency_ms=12.3456,
    )

    d = result.model_dump()
    assert d["dialogue_length"] == 5
    assert len(d["verdicts"]) == 1
    assert d["verdicts"][0]["bot_probability"] == 0.88
    assert d["latency_ms"] == 12.346

    reconstructed = DialogueResultItem(**d)
    assert reconstructed.dialogue_length == 5
    assert reconstructed.verdicts[0].participant_index == 1
    assert reconstructed.verdicts[0].is_bot is True


def test_triton_client_auto_chunks_large_batches() -> None:
    """Клиент Triton прозрачно режет крупные батчи по max_batch_size и склеивает результат."""
    mock_transport = MagicMock()
    mock_transport.infer.side_effect = lambda features, **_kwargs: (
        np.full(len(features), 0.5, dtype=np.float32),
        np.full(len(features), 0.1, dtype=np.float32),
        np.full(len(features), 0.2, dtype=np.float32),
        np.full(len(features), 0.3, dtype=np.float32),
    )

    client = TritonEnsembleClient(transport=mock_transport, max_batch_size=3)
    feats = np.ones((7, 10), dtype=np.float32)

    p_meta, p_cb, p_lgb, p_xgb = client.infer(feats)

    assert mock_transport.infer.call_count == 3
    assert [len(call.kwargs["features"]) for call in mock_transport.infer.call_args_list] == [3, 3, 1]
    assert p_meta.shape == (7,)
    assert p_cb.shape == (7,)
    assert p_lgb.shape == (7,)
    assert p_xgb.shape == (7,)
