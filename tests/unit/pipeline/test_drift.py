from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from project.drift.data_drift import DataDriftDetector, DWHFeatureFetcher
from project.drift.feature_importance_diff import (
    FeatureImportanceDiffAnalyzer,
    FeatureImportanceDiffPrometheusExporter,
    FeatureShiftStatus,
)
from project.features.metadata import FeatureMetadataRegistry


def test_data_drift_detector_statistical_check() -> None:
    """Проверка статистического анализа дрифта через двухвыборочный KS-тест и экспорт в Prometheus с Polars."""
    feature_names = [f"f_{i}" for i in range(10)]

    np.random.seed(42)
    ref_data = pl.DataFrame(np.random.normal(0, 1, size=(100, 10)), schema=feature_names, orient="row")
    cur_data = pl.DataFrame(np.random.normal(0, 1, size=(100, 10)), schema=feature_names, orient="row")

    detector = DataDriftDetector()
    summary = detector.detect_drift(
        reference_df=ref_data,
        current_df=cur_data,
        drift_share_threshold=0.30,
    )

    assert isinstance(summary.dataset_drift, bool)
    assert isinstance(summary.drifted_features, list)
    assert summary.drift_share < 0.30


def test_dwh_feature_fetcher_fail_fast_on_error_or_insufficient_data() -> None:
    """Проверка fail-fast: DWHFeatureFetcher обязан падать, а не подменять данные синтетикой."""
    mock_client = MagicMock()
    # 1. Ошибка ClickHouse
    mock_client.query.side_effect = RuntimeError("ClickHouse connection dropped")
    fetcher = DWHFeatureFetcher(client=mock_client)

    with pytest.raises(RuntimeError, match="ClickHouse reference_features fetch failed"):
        fetcher.load_or_create_reference_dataset()

    with pytest.raises(RuntimeError, match="ClickHouse chat_logs fetch failed"):
        fetcher.fetch_current_features(days=7)

    # 2. Недостаточно данных (< 10 строк)
    mock_result = MagicMock()
    mock_result.result_rows = [[0.1] * 52] * 5  # всего 5 строк
    mock_client.query.side_effect = None
    mock_client.query.return_value = mock_result

    with pytest.raises(RuntimeError, match="minimum 10 required"):
        fetcher.load_or_create_reference_dataset()

    with pytest.raises(RuntimeError, match="minimum 10 required"):
        fetcher.fetch_current_features(days=7)


def test_dwh_feature_fetcher_current_query_format() -> None:
    """Проверка формирования SQL-запроса выгрузки фичей за окно мониторинга без искусственного LIMIT."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    fetcher = DWHFeatureFetcher(client=mock_client)
    n_feats = len(fetcher.store.feature_names)
    mock_result.result_rows = [[[0.5] * n_feats, [0.5] * n_feats]] * 15
    mock_client.query.return_value = mock_result

    res_df = fetcher.fetch_current_features(days=7)

    assert isinstance(res_df, pl.DataFrame)
    assert len(res_df) == 30  # 15 диалогов * 2 участника

    called_query = mock_client.query.call_args[0][0]
    assert "WHERE timestamp >= now() - INTERVAL 7 DAY" in called_query
    assert "LIMIT" not in called_query


def test_feature_importance_diff_and_prometheus_export() -> None:
    """Проверка расчета смещения важности признаков и экспорта в Prometheus."""
    ref_importance = {
        "feat_a": 10.0,
        "feat_b": 8.0,
        "feat_d": 7.0,
        "feat_f": 6.0,
        "feat_c": 2.0,
    }
    cur_importance = {
        "feat_a": 9.0,
        "feat_b": 4.0,
        "feat_c": 12.0,
        "feat_e": 6.0,
    }

    analyzer = FeatureImportanceDiffAnalyzer(reference_importance=ref_importance)
    diff_report = analyzer.analyze(current_importance=cur_importance, top_n=4)

    assert diff_report.top_n == 4
    features_dict = {f.feature: f for f in diff_report.features}
    assert features_dict["feat_c"].status == FeatureShiftStatus.SURGING
    assert features_dict["feat_c"].status == "SURGING"
    assert features_dict["feat_e"].status == FeatureShiftStatus.NEW
    assert features_dict["feat_e"].status == "NEW"

    exporter = FeatureImportanceDiffPrometheusExporter()
    exporter.export(diff_report)
    dname_c = FeatureMetadataRegistry.get_display_name("feat_c")
    assert exporter.FEATURE_IMPORTANCE_CURRENT.labels(feature="feat_c", display_name=dname_c)._value.get() == 12.0


def test_feature_importance_diff_redis_sync_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверка записи FeatureImportanceDiff в Redis и синхронизации в API Gateway Prometheus Gauges."""
    import json
    from unittest.mock import patch

    from project.api.metrics import metrics_manager

    analyzer = FeatureImportanceDiffAnalyzer(reference_importance={"msg_share": 5.0, "n_messages": 10.0})
    report = analyzer.analyze({"msg_share": 12.0, "n_messages": 6.0}, top_n=2)

    fake_store: dict[str, str] = {}

    class _FakeRedis:
        def set(self, key: str, val: str, **_: object) -> None:
            fake_store[key] = val

        def mget(self, keys: list[str]) -> list[str | None]:
            return [fake_store.get(k) for k in keys]

        def close(self) -> None:
            pass

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(metrics_manager, "_redis_client", None)
    fake_redis = _FakeRedis()
    with patch("redis.Redis.from_url", return_value=fake_redis):
        FeatureImportanceDiffPrometheusExporter().export(report)
        assert "bot_detector:feature_importance_metrics" in fake_store
        payload = json.loads(fake_store["bot_detector:feature_importance_metrics"])
        assert len(payload["features"]) == 2

        metrics_manager.sync_from_redis()

    dname_msg = FeatureMetadataRegistry.get_display_name("msg_share")
    delta_val = FeatureImportanceDiffPrometheusExporter.FEATURE_IMPORTANCE_DELTA.labels(
        feature="msg_share", display_name=dname_msg
    )._value.get()
    shift_val = FeatureImportanceDiffPrometheusExporter.FEATURE_RANK_SHIFT.labels(
        feature="msg_share", display_name=dname_msg
    )._value.get()
    assert delta_val == 7.0
    assert shift_val == 1.0


def test_data_drift_detector_handles_nan_inf_and_disjoint_columns() -> None:
    """DataDriftDetector фильтрует NaN/Inf в Polars и не падает при непересекающихся колонках."""
    detector = DataDriftDetector()

    ref_df = pl.DataFrame({"f1": [1.0, 2.0, float("nan"), float("inf"), 3.0, 4.0]})
    cur_df = pl.DataFrame({"f1": [1.1, 2.1, float("-inf"), 3.1, 4.1, 5.0]})
    summary = detector.detect_drift(ref_df, cur_df)
    assert summary.number_of_columns == 1
    assert 0.0 <= summary.drift_share <= 1.0

    disjoint = detector.detect_drift(pl.DataFrame({"a": [1.0, 2.0]}), pl.DataFrame({"b": [1.0, 2.0]}))
    assert disjoint.number_of_columns == 0
    assert disjoint.dataset_drift is False
