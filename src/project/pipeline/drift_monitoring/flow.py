"""Prefect Flow для комплексного фонового мониторинга дрифта данных и моделей."""

import argparse
from typing import Any

from prefect import flow

from project.core.config import config
from project.core.logging import LoggingConfigurator, ProjectLogger
from project.drift.concept_drift import ConceptDriftDetector
from project.drift.data_drift import DWHFeatureFetcher
from project.drift.model_age import ChampionMetadataReader
from project.drift.prediction_drift import PredictionDriftDetector
from project.pipeline.drift_monitoring.decision import DriftDecisionEngine
from project.pipeline.drift_monitoring.schemas import DriftMonitoringResult
from project.pipeline.drift_monitoring.tasks import (
    detect_feature_drift_task,
    fetch_dwh_data_task,
)
from project.pipeline.metrics_sync import DriftMetricsSyncService

_MIN_DRIFT_EVAL_DIALOGUES = 10

logger = ProjectLogger.get_logger(__name__)


class DriftMonitoringDeployer:
    """Координатор вспомогательных операций и деплоя Prefect Flow мониторинга дрифта."""

    @staticmethod
    def persist_decision(dialogues_count: int, model_age_days: float, status: int, reason: str) -> None:
        """Сохранение снапшота адаптивного решения в Redis и синхронизация `/metrics`."""
        DriftMetricsSyncService.persist_adaptive_decision(dialogues_count, model_age_days, status, reason)
        DriftMetricsSyncService.trigger_api_gateway_metrics()

    @staticmethod
    def build_healthy_reset_result(
        fetcher: DWHFeatureFetcher,
        *,
        effective_threshold: float,
        dialogues_count: int,
        model_age_days: float,
        reason: str,
    ) -> DriftMonitoringResult:
        """Сброс базовых метрик в здоровое состояние при малом трафике или смене чемпиона."""
        feat_names = fetcher.get_feature_names()
        pred_report, concept_report = DriftMetricsSyncService.reset_baseline(
            feat_names,
            drift_threshold=effective_threshold,
            dialogues_count=dialogues_count,
            model_age_days=model_age_days,
            retrain_status=0,
            reason=reason,
        )
        DriftMetricsSyncService.trigger_api_gateway_metrics()
        return DriftDecisionEngine.build_zero_monitoring_result(
            dialogues_count=dialogues_count,
            model_age_days=model_age_days,
            num_columns=len(feat_names),
            retrain_status=0,
            reason=reason,
            psi=pred_report.psi,
            entropy=pred_report.mean_entropy,
            disagreement_rate=concept_report.disagreement_rate,
            concept_drift_detected=concept_report.concept_drift_detected,
        )

    @staticmethod
    def trigger_retraining_deployment() -> None:
        """Асинхронная постановка задачи переобучения в очередь Prefect Server."""
        try:
            from prefect.deployments import run_deployment

            run_deployment(
                name="bot_detection_retrain_flow/bot-detection-retraining",
                timeout=0,
                parameters={"force_retrain": True},
            )
            logger.info("Successfully queued 'bot-detection-retraining' flow run via Prefect API")
        except Exception as trg_err:  # noqa: BLE001
            logger.warning("Could not auto-trigger retraining deployment: %s", trg_err)

    @staticmethod
    def extract_summary_field(drift_summary: Any, field_name: str, default: Any) -> Any:
        """Унифицированное чтение поля из `DataDriftSummary` или dict (после сериализации Prefect)."""
        if isinstance(drift_summary, dict):
            return drift_summary.get(field_name, default)
        return getattr(drift_summary, field_name, default)

    @staticmethod
    def deploy() -> str:
        """Регистрация расписания мониторинга дрифта в Prefect Server."""
        from prefect.client.schemas.objects import ConcurrencyLimitConfig, ConcurrencyLimitStrategy
        from prefect.deployments.runner import EntrypointType

        deployment: Any = drift_monitoring_flow.to_deployment(
            name="drift-monitor",
            work_pool_name="default-agent-pool",
            cron=config.schedule.drift_monitoring_cron,
            concurrency_limit=ConcurrencyLimitConfig(
                limit=1,
                collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW,
                grace_period_seconds=60,
            ),
            tags=["monitoring", "data-drift", "decision-engine"],
            description="Регулярный автоматический аудит распределений признаков, объема данных и адаптивного ретрейна",
            entrypoint_type=EntrypointType.MODULE_PATH,
        )
        deployment.entrypoint = f"project.pipeline.drift_monitoring.flow:{drift_monitoring_flow.name}"
        dep_id = deployment.apply()
        logger.info("Successfully registered deployment 'drift-monitor' in Prefect (ID: %s)", dep_id)
        return str(dep_id)


@flow(name="drift_monitoring_flow")
def drift_monitoring_flow(
    days: int = 7,
    drift_threshold: float | None = None,
    auto_trigger: bool | None = None,
) -> DriftMonitoringResult:
    """Регулярный фоновый мониторинг Data Drift, Prediction Drift и Concept Drift."""
    effective_threshold = config.schedule.drift_threshold if drift_threshold is None else float(drift_threshold)
    logger.info("Starting Comprehensive Drift Monitoring (days=%d, threshold=%.2f)", days, effective_threshold)

    champ_info = ChampionMetadataReader.read("bot_detector")
    model_age_days = float(champ_info.get("age_days", 0.0))
    since_ts_raw = champ_info.get("created_at_ts")
    since_ts = float(since_ts_raw) if since_ts_raw is not None else None

    fetcher = DWHFeatureFetcher()
    try:
        production_dialogues_count = fetcher.count_recent_dialogues(days=days, since_ts=since_ts)
    except FileNotFoundError as exc:
        logger.warning("ADAPTIVE DECISION: Feature store state not initialized yet (%s). Skipping drift check.", exc)
        DriftMonitoringDeployer.persist_decision(0, model_age_days, 0, "awaiting_model_initialization")
        return DriftDecisionEngine.build_zero_monitoring_result(
            dialogues_count=0,
            model_age_days=model_age_days,
            num_columns=0,
            retrain_status=0,
            reason="awaiting_model_initialization",
        )

    if production_dialogues_count < _MIN_DRIFT_EVAL_DIALOGUES:
        logger.info(
            "ADAPTIVE DECISION: System healthy (awaiting production traffic: %d < %d dialogues).",
            production_dialogues_count,
            _MIN_DRIFT_EVAL_DIALOGUES,
        )
        return DriftMonitoringDeployer.build_healthy_reset_result(
            fetcher,
            effective_threshold=effective_threshold,
            dialogues_count=production_dialogues_count,
            model_age_days=model_age_days,
            reason="system_healthy",
        )

    if DriftDecisionEngine.is_retrain_in_progress():
        logger.info("ADAPTIVE DECISION: Retraining already in progress. Skipping duplicate trigger.")
        DriftMonitoringDeployer.persist_decision(
            production_dialogues_count, model_age_days, 3, "retraining_in_progress"
        )
        return DriftDecisionEngine.build_zero_monitoring_result(
            dialogues_count=production_dialogues_count,
            model_age_days=model_age_days,
            num_columns=len(fetcher.get_feature_names()),
            retrain_status=3,
            reason="retraining_in_progress",
        )

    data = fetch_dwh_data_task(days=days, since_ts=since_ts) if since_ts is not None else fetch_dwh_data_task(days=days)
    drift_summary = detect_feature_drift_task(data=data, drift_threshold=effective_threshold)

    champ_run_id = str(champ_info.get("run_id") or "")
    ref_probs, cur_probs, cur_verdicts = (
        fetcher.fetch_prediction_data(days=days, since_ts=since_ts, reference_run_id=champ_run_id)
        if champ_run_id
        else fetcher.fetch_prediction_data(days=days, since_ts=since_ts)
    )
    pred_report = PredictionDriftDetector().detect_drift(
        reference_probs=ref_probs,
        current_probs=cur_probs,
        current_verdicts=cur_verdicts,
    )
    concept_probs = fetcher.fetch_concept_drift_probs(days=days, since_ts=since_ts, limit=200)
    concept_report = ConceptDriftDetector().detect_drift(model_probs=concept_probs or {"ensemble": cur_probs})

    drift_share = float(DriftMonitoringDeployer.extract_summary_field(drift_summary, "drift_share", 0.0))
    dataset_drift = bool(DriftMonitoringDeployer.extract_summary_field(drift_summary, "dataset_drift", False))
    any_drift = bool(dataset_drift or pred_report.is_drifted or concept_report.concept_drift_detected)

    post_champ_info = ChampionMetadataReader.read("bot_detector")
    if post_champ_info.get("version") != champ_info.get("version") and not DriftDecisionEngine.is_retrain_in_progress():
        post_since_raw = post_champ_info.get("created_at_ts")
        post_since_ts = float(post_since_raw) if post_since_raw is not None else None
        post_count = fetcher.count_recent_dialogues(days=days, since_ts=post_since_ts)
        post_age = float(post_champ_info.get("age_days", 0.0))
        logger.info(
            "Champion promoted during drift check (%s -> %s); restoring clean baseline.",
            champ_info.get("version"),
            post_champ_info.get("version"),
        )
        return DriftMonitoringDeployer.build_healthy_reset_result(
            fetcher,
            effective_threshold=effective_threshold,
            dialogues_count=post_count,
            model_age_days=post_age,
            reason="promoted",
        )

    if DriftDecisionEngine.is_retrain_in_progress():
        retrain_status, decision_reason, retrain_recommended = 3, "retraining_in_progress", False
        logger.info("ADAPTIVE DECISION: Retraining already in progress. Skipping duplicate trigger.")
    else:
        should_auto_trigger = config.schedule.auto_trigger_retrain if auto_trigger is None else auto_trigger
        retrain_status, decision_reason, retrain_recommended = DriftDecisionEngine.evaluate(
            any_drift=any_drift,
            is_stale=model_age_days >= config.schedule.max_model_staleness_days,
            has_enough_data=production_dialogues_count >= config.schedule.min_retrain_samples,
            should_auto_trigger=should_auto_trigger,
            is_in_cooldown=DriftDecisionEngine.is_retrain_in_cooldown(),
            production_dialogues_count=production_dialogues_count,
            min_retrain_samples=config.schedule.min_retrain_samples,
            drift_share=drift_share,
            model_age_days=model_age_days,
        )
        if retrain_recommended and should_auto_trigger:
            DriftMonitoringDeployer.trigger_retraining_deployment()

    DriftMonitoringDeployer.persist_decision(
        production_dialogues_count, model_age_days, retrain_status, decision_reason
    )
    return {
        "drift_share": drift_share,
        "dataset_drift": dataset_drift,
        "prediction_drift_psi": pred_report.psi,
        "prediction_entropy": pred_report.mean_entropy,
        "ensemble_disagreement_rate": concept_report.disagreement_rate,
        "concept_drift_detected": concept_report.concept_drift_detected,
        "retrain_recommended": retrain_recommended,
        "retrain_status": retrain_status,
        "decision_reason": decision_reason,
        "production_dialogues_count": production_dialogues_count,
        "model_age_days": model_age_days,
        "drifted_features": DriftMonitoringDeployer.extract_summary_field(drift_summary, "drifted_features", []),
        "number_of_drifted_columns": DriftMonitoringDeployer.extract_summary_field(
            drift_summary, "number_of_drifted_columns", 0
        ),
        "number_of_columns": DriftMonitoringDeployer.extract_summary_field(drift_summary, "number_of_columns", 0),
    }


def main() -> None:
    """CLI интерфейс для ручного запуска и регистрации расписания."""
    parser = argparse.ArgumentParser(description="Drift Monitoring Flow")
    parser.add_argument("--days", type=int, default=7, help="Дни истории для выборки из ClickHouse")
    parser.add_argument("--threshold", type=float, default=None, help="Порог допустимого дрифта")
    parser.add_argument("--deploy", action="store_true", help="Зарегистрировать расписание в Prefect UI")
    parser.add_argument(
        "--no-auto-trigger",
        action="store_true",
        help="Не ставить задачу переобучения в очередь автоматически при детекции дрифта",
    )
    args = parser.parse_args()

    LoggingConfigurator.configure()
    if args.deploy:
        DriftMonitoringDeployer.deploy()
        return

    res = drift_monitoring_flow(
        days=args.days,
        drift_threshold=args.threshold,
        auto_trigger=False if args.no_auto_trigger else None,
    )
    logger.info("=== Drift Monitoring Summary ===")
    logger.info("Retrain Recommended: %s", res["retrain_recommended"])
    logger.info("Data Drift Share: %.2f%%", res["drift_share"] * 100)
    logger.info("Prediction PSI: %.4f", res["prediction_drift_psi"])


if __name__ == "__main__":
    main()
