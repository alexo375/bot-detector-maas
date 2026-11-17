"""Интеграционные тесты конвейера инференса: FeatureStore -> Triton -> PairwisePostprocessor -> Cache & Audit."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from project.api.schemas import DialogueResultItem, PredictResponse
from project.features.store import FeatureStore
from project.infrastructure.rabbitmq.schemas.task import InferenceTask
from project.infrastructure.workers.inference_worker import InferenceWorker
from project.models.postprocessors.pairwise import PairwisePostprocessor
from tests.fixtures.mock_fixtures import FakeDialogueCache


@pytest.fixture
def feature_store() -> FeatureStore:
    """Экземпляр FeatureStore с реальными экстракторами признаков."""
    return FeatureStore()


@pytest.fixture
def pairwise_postprocessor() -> PairwisePostprocessor:
    """Реальный постпроцессор парной калибровки вероятностей."""
    return PairwisePostprocessor(max_pair_sum=1.0)


def test_feature_store_end_to_end_extraction(feature_store: FeatureStore) -> None:
    """Интеграционный тест: извлечение векторов признаков из реальных разнородных диалогов."""
    dialogues = [
        [
            {"participant_index": 0, "text": "Привет! Ты бот? Скажи что-нибудь смешное 😂"},
            {"participant_index": 1, "text": "Привет! Я обычный человек, пью кофе и пишу тесты."},
            {"participant_index": 0, "text": "Ага, все боты так говорят! http://example.com/check"},
            {"participant_index": 1, "text": "Ну ладно, думай как знаешь))"},
        ],
        [
            {"participant_index": 0, "text": "Fast msg 1"},
            {"participant_index": 1, "text": "Fast reply 2"},
        ],
    ]

    features = feature_store.transform(dialogues)

    # 2 диалога * 2 участника = 4 строки матрицы признаков
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 4
    assert features.shape[1] > 0
    # Признаки не должны содержать NaN или Inf
    assert not np.isnan(features).any(), "Матрица признаков содержит NaN"
    assert not np.isinf(features).any(), "Матрица признаков содержит Inf"


def test_pairwise_postprocessor_physical_constraint(
    pairwise_postprocessor: PairwisePostprocessor,
) -> None:
    """Интеграционный тест: проверка физического ограничения P(Bot0) + P(Bot1) <= 1.0."""
    # Ситуация, когда обе модели бустинга независимо выдали высокую вероятность бота
    raw_probs = np.array([0.85, 0.75, 0.20, 0.30], dtype=np.float32)
    dialog_ids = np.array(["d1", "d1", "d2", "d2"])
    participant_indices = np.array([0, 1, 0, 1])

    calibrated = pairwise_postprocessor.transform(
        raw_probs,
        dialog_ids=dialog_ids,
        participant_indices=participant_indices,
    )

    assert len(calibrated) == 4
    # Для первого диалога сумма вероятностей не должна превышать 1.0
    p0_d1, p1_d1 = float(calibrated[0]), float(calibrated[1])
    assert p0_d1 + p1_d1 <= 1.0 + 1e-5, f"Сумма вероятностей {p0_d1 + p1_d1} превышает 1.0"
    # Относительный порядок участников должен сохраняться (0.85 > 0.75)
    assert p0_d1 > p1_d1

    # Для второго диалога сумма 0.2 + 0.3 = 0.5 <= 1.0, калибровка не занижает
    p0_d2, p1_d2 = float(calibrated[2]), float(calibrated[3])
    assert pytest.approx(p0_d2, abs=1e-3) == 0.20
    assert pytest.approx(p1_d2, abs=1e-3) == 0.30


def test_inference_worker_full_pipeline_sync_execution(
    feature_store: FeatureStore,
    pairwise_postprocessor: PairwisePostprocessor,
) -> None:
    """Интеграционный тест полного синхронного конвейера обработки задачи в InferenceWorker."""
    cache = FakeDialogueCache()
    mock_triton = MagicMock()

    # Имитируем предсказания ансамбля Triton: meta, catboost, lightgbm, xgboost
    def mock_infer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = features.shape[0]
        # Четные (участник 0) - человек (0.1), нечетные (участник 1) - бот (0.9)
        p_meta = np.array([0.10 if i % 2 == 0 else 0.90 for i in range(n)], dtype=np.float32)
        p_cb = np.array([0.15 if i % 2 == 0 else 0.85 for i in range(n)], dtype=np.float32)
        p_lgb = np.array([0.12 if i % 2 == 0 else 0.88 for i in range(n)], dtype=np.float32)
        p_xgb = np.array([0.14 if i % 2 == 0 else 0.86 for i in range(n)], dtype=np.float32)
        return p_meta, p_cb, p_lgb, p_xgb

    mock_triton.infer.side_effect = mock_infer

    worker = InferenceWorker(
        triton_client=mock_triton,
        feature_store=feature_store,
        postprocessor=pairwise_postprocessor,
        cache=cache,  # type: ignore[arg-type]
    )

    task = InferenceTask(
        task_id="pipeline-task-001",
        dialogues=[
            {
                "dialog_id": "dlg_integration_1",
                "messages": [
                    {"participant_index": 0, "text": "Здравствуйте, как оформить возврат заказа?"},
                    {"participant_index": 1, "text": "Для возврата перейдите по ссылке в личном кабинете."},
                ],
            },
            {
                "dialog_id": "dlg_integration_2",
                "messages": [
                    {"participant_index": 0, "text": "Привет! Че делаешь?)"},
                    {"participant_index": 1, "text": "Да вот сижу отдыхаю."},
                ],
            },
        ],
    )

    response, events = worker.process_task_sync(task)

    # 1. Проверка ответа клиенту
    assert isinstance(response, PredictResponse)
    assert response.total == 2
    assert len(response.results) == 2

    res0 = response.results[0]
    assert isinstance(res0, DialogueResultItem)
    assert res0.dialog_id == "dlg_integration_1"
    assert res0.cached is False
    assert len(res0.verdicts) == 2
    assert res0.verdicts[0].participant_index == 0
    assert res0.verdicts[0].is_bot is False
    assert res0.verdicts[1].participant_index == 1
    assert res0.verdicts[1].is_bot is True

    # 2. Проверка событий аудита для ClickHouse
    assert len(events) == 2
    ev0 = events[0]
    assert ev0.dialog_id == "dlg_integration_1"
    assert ev0.verdict.value == "p1_bot"
    assert len(ev0.features_p0) > 0
    assert len(ev0.features_p1) > 0
    assert ev0.is_bot_p0 == 0
    assert ev0.is_bot_p1 == 1

    # 3. Проверка записи в кэш диалогов
    assert len(cache.dialogues) == 2
