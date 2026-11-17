"""Интеграционные тесты мониторинга: метрики Prometheus и экспортеры Data Drift / Feature Importance Shift."""

from prometheus_client import REGISTRY

from project.api.metrics import (
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    PREDICTIONS_TOTAL,
)
from project.drift.data_drift.prometheus_exporter import DataDriftPrometheusExporter
from project.drift.data_drift.summary import DataDriftSummary
from project.drift.feature_importance_diff.prometheus_exporter import FeatureImportanceDiffPrometheusExporter
from project.drift.feature_importance_diff.schemas.record import FeatureImportanceRecord
from project.drift.feature_importance_diff.schemas.report import FeatureImportanceDiffReport
from project.drift.feature_importance_diff.schemas.shift_status import FeatureShiftStatus
from project.features.metadata import FeatureMetadataRegistry


def test_data_drift_prometheus_exporter_integration():
    """Интеграционный тест экспортера метрик Data Drift в реестр Prometheus."""
    exporter = DataDriftPrometheusExporter()

    summary = DataDriftSummary(
        dataset_drift=True,
        drift_share=0.2,
        number_of_drifted_columns=2,
        number_of_columns=10,
        drift_threshold=0.1,
        retrain_recommended=True,
        drifted_features=["msg_share", "max_consecutive_msgs"],
    )
    p_values = {
        "msg_share": 0.001,
        "max_consecutive_msgs": 0.003,
        "is_last_sender": 0.45,
    }

    exporter.export(summary, column_p_values=p_values)

    # Проверяем значения в реестре Prometheus
    assert exporter.DATA_DRIFT_SHARE._value.get() == 0.2
    assert exporter.DATA_DRIFT_DETECTED._value.get() == 1.0

    # Проверяем метрики для конкретных признаков
    dname_msg = FeatureMetadataRegistry.get_display_name("msg_share")
    dname_last = FeatureMetadataRegistry.get_display_name("is_last_sender")
    val_p_msg_share = exporter.FEATURE_DRIFT_P_VALUE.labels(feature="msg_share", display_name=dname_msg)._value.get()
    assert abs(val_p_msg_share - 0.001) < 1e-5
    val_status_msg_share = exporter.FEATURE_DRIFT_STATUS.labels(
        feature="msg_share", display_name=dname_msg
    )._value.get()
    assert val_status_msg_share == 1.0

    val_status_normal = exporter.FEATURE_DRIFT_STATUS.labels(
        feature="is_last_sender", display_name=dname_last
    )._value.get()
    assert val_status_normal == 0.0


def test_feature_importance_diff_prometheus_exporter_integration():
    """Интеграционный тест экспортера метрик Feature Importance Drift в Prometheus."""
    exporter = FeatureImportanceDiffPrometheusExporter()

    records = [
        FeatureImportanceRecord(
            feature="is_last_sender",
            current_importance=0.15,
            previous_importance=0.10,
            importance_delta=0.05,
            current_rank=1,
            previous_rank=2,
            rank_shift=-1,
            status=FeatureShiftStatus.SURGING,
        ),
        FeatureImportanceRecord(
            feature="total_words",
            current_importance=0.02,
            previous_importance=0.08,
            importance_delta=-0.06,
            current_rank=10,
            previous_rank=4,
            rank_shift=6,
            status=FeatureShiftStatus.DROPPING,
        ),
    ]

    report = FeatureImportanceDiffReport.create(
        top_n=2,
        features=records,
        dropped_features=[],
    )

    exporter.export(report)

    # Проверяем экспорт в метрики
    dname_last = FeatureMetadataRegistry.get_display_name("is_last_sender")
    dname_words = FeatureMetadataRegistry.get_display_name("total_words")
    curr_imp = exporter.FEATURE_IMPORTANCE_CURRENT.labels(
        feature="is_last_sender", display_name=dname_last
    )._value.get()
    assert abs(curr_imp - 0.15) < 1e-4

    prev_imp = exporter.FEATURE_IMPORTANCE_REFERENCE.labels(
        feature="is_last_sender", display_name=dname_last
    )._value.get()
    assert abs(prev_imp - 0.10) < 1e-4

    delta = exporter.FEATURE_IMPORTANCE_DELTA.labels(feature="is_last_sender", display_name=dname_last)._value.get()
    assert abs(delta - 0.05) < 1e-4

    rank_shift = exporter.FEATURE_RANK_SHIFT.labels(feature="total_words", display_name=dname_words)._value.get()
    assert rank_shift == 6.0


def test_api_business_metrics_registry_integration():
    """Интеграционный тест регистрации и инкремента счетчиков кэша и вердиктов API."""
    initial_hits = CACHE_HITS_TOTAL._value.get()
    initial_misses = CACHE_MISSES_TOTAL._value.get()

    CACHE_HITS_TOTAL.inc()
    CACHE_MISSES_TOTAL.inc(2)

    assert CACHE_HITS_TOTAL._value.get() == initial_hits + 1
    assert CACHE_MISSES_TOTAL._value.get() == initial_misses + 2

    # Проверка многоклассового счетчика вердиктов
    PREDICTIONS_TOTAL.labels(verdict="both_humans").inc()
    PREDICTIONS_TOTAL.labels(verdict="both_bots").inc()

    metric_names = [m.name for m in REGISTRY.collect()]
    assert "bot_detector_cache_hits" in metric_names
    assert "bot_detector_cache_misses" in metric_names
    assert "bot_detector_predictions" in metric_names
