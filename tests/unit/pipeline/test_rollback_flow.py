"""Модульные тесты для ручного отката моделей через Prefect (MLflow -> Triton -> Redis)."""

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import onnx
import pytest
from onnx import TensorProto, helper

from project.drift.model_age import ChampionMetadataReader
from project.pipeline.model_rollback import (
    ModelVersionCandidate,
    RollbackFlowDeployer,
    RollbackMode,
    bot_detection_rollback_flow,
)
from project.pipeline.model_rollback.inspector import ModelVersionInspector
from project.pipeline.model_rollback.interactive import RollbackPromptUI
from project.pipeline.model_rollback.tasks import (
    invalidate_prediction_cache_task,
    resolve_rollback_target_task,
    switch_mlflow_champion_task,
)

pytestmark = [pytest.mark.xdist_group("prefect"), pytest.mark.usefixtures("prefect_isolated_server")]


def _make_dummy_onnx(path: Path, input_dim: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    inp = helper.make_tensor_value_info("meta_input", TensorProto.FLOAT, [None, input_dim])
    out = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 2])
    node = helper.make_node("Identity", ["meta_input"], ["probabilities"])
    graph = helper.make_graph([node], "dummy", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx.save(model, str(path))


def test_resolve_rollback_target_steps_and_exact() -> None:
    """Выбор цели отката для steps_back=1/2, exact_version и граничных ошибок."""
    candidates = [
        ModelVersionCandidate("5", "run_5", 500.0, 0.14, ["@champion"], "v5 | 2026 | loss=0.1400 | @champion"),
        ModelVersionCandidate("4", "run_4", 400.0, 0.15, [], "v4 | 2026 | loss=0.1500"),
        ModelVersionCandidate("2", "run_2", 200.0, 0.16, ["@rolled_back"], "v2 | 2026 | loss=0.1600 | @rolled_back"),
    ]

    picked_1 = resolve_rollback_target_task.fn(RollbackMode.STEPS_BACK, steps_back=1, candidates=candidates)
    assert picked_1.version == "4"

    picked_2 = resolve_rollback_target_task.fn(RollbackMode.STEPS_BACK, steps_back=2, candidates=candidates)
    assert picked_2.version == "2"

    exact = resolve_rollback_target_task.fn(RollbackMode.EXACT_VERSION, target_version="v2", candidates=candidates)
    assert exact.run_id == "run_2"

    with pytest.raises(ValueError, match="Недостаточно прошлых версий"):
        resolve_rollback_target_task.fn(RollbackMode.STEPS_BACK, steps_back=3, candidates=candidates)

    with pytest.raises(ValueError, match="не найдена"):
        resolve_rollback_target_task.fn(RollbackMode.EXACT_VERSION, target_version="99", candidates=candidates)


def test_prompt_target_version_interactively_parses_selection() -> None:
    """Проверка работы интерактивной формы выбора версии в UI Prefect."""
    candidates = [
        ModelVersionCandidate("4", "run_4", 400.0, 0.14, ["@champion"], "v4 | 2026-11-18 | loss=0.1400 | @champion"),
        ModelVersionCandidate("3", "run_3", 300.0, 0.15, [], "v3 | 2026-11-17 | loss=0.1500"),
    ]

    with patch(
        "project.pipeline.model_rollback.interactive.pause_flow_run",
        return_value=SimpleNamespace(target_version=candidates[1].label),
    ):
        chosen = RollbackPromptUI.prompt_target_version(candidates)
        assert chosen == "3"

    with (
        patch("project.pipeline.model_rollback.interactive.pause_flow_run", return_value=None),
        pytest.raises(RuntimeError, match="Откат отменен"),
    ):
        RollbackPromptUI.prompt_target_version(candidates)


def test_list_rollback_candidates_filters_incomplete_runs() -> None:
    """Список версий сортируется по убыванию и отсеивает раны без 4 ONNX-артефактов."""
    mock_client = MagicMock()
    mv1 = SimpleNamespace(version="1", run_id="r1", creation_timestamp=1000, aliases=[])
    mv2 = SimpleNamespace(version="2", run_id="r2", creation_timestamp=2000, aliases=[])
    mv3 = SimpleNamespace(version="3", run_id="r3", creation_timestamp=3000, aliases=[])
    mock_client.search_model_versions.return_value = [mv1, mv3, mv2]
    mock_client.get_model_version_by_alias.side_effect = [
        SimpleNamespace(version="3"),  # champion
        SimpleNamespace(version="2"),  # challenger
        None,  # rolled_back
    ]

    def _list_artifacts(rid: str, path: str | None = None):
        if rid == "r1":
            return [SimpleNamespace(path="catboost_detector")]  # incomplete
        return [
            SimpleNamespace(path="catboost_detector"),
            SimpleNamespace(path="lightgbm_detector"),
            SimpleNamespace(path="xgboost_detector"),
            SimpleNamespace(path="meta_stacker"),
        ]

    mock_client.list_artifacts.side_effect = _list_artifacts
    mock_client.get_run.side_effect = lambda rid: SimpleNamespace(
        outputs=None,
        data=SimpleNamespace(metrics={"val_logloss": 0.12 if rid == "r3" else 0.14}),
    )

    with (
        patch("project.pipeline.model_rollback.inspector.MlflowClient", return_value=mock_client),
        patch("project.tracking.experiment_tracker.loggers.onnx_resolver.MlflowClient", return_value=mock_client),
    ):
        res = ModelVersionInspector.list_candidates()

    assert [c.version for c in res] == ["3", "2"]
    assert "@champion" in res[0].aliases
    assert "@challenger" in res[1].aliases
    assert "loss=0.1200" in res[0].label


def test_switch_mlflow_champion_validates_meta_dim_and_updates_aliases(tmp_path: Path) -> None:
    """Переключение @champion отклоняет несовместимую ONNX-размерность и ставит @rolled_back прежнему."""
    bad_onnx = tmp_path / "bad" / "meta_stacker" / "model.onnx"
    _make_dummy_onnx(bad_onnx, input_dim=4)
    good_onnx = tmp_path / "good" / "meta_stacker" / "model.onnx"
    _make_dummy_onnx(good_onnx, input_dim=3)

    mock_client = MagicMock()
    mock_client.get_run.return_value = SimpleNamespace(outputs=None)
    mock_client.list_artifacts.return_value = [
        SimpleNamespace(path=p) for p in ("catboost_detector", "lightgbm_detector", "xgboost_detector", "meta_stacker")
    ]
    mock_client.get_model_version.side_effect = [
        SimpleNamespace(version="1", run_id="run_bad"),
        SimpleNamespace(version="2", run_id="run_good"),
    ]
    mock_client.download_artifacts.side_effect = [str(bad_onnx), str(good_onnx)]
    mock_client.get_model_version_by_alias.return_value = SimpleNamespace(version="3")

    with (
        patch("project.pipeline.model_rollback.tasks.MlflowClient", return_value=mock_client),
        patch("project.pipeline.model_rollback.inspector.MlflowClient", return_value=mock_client),
        patch("project.tracking.experiment_tracker.loggers.onnx_resolver.MlflowClient", return_value=mock_client),
    ):
        with pytest.raises(ValueError, match="Несовместимая размерность meta_stacker"):
            switch_mlflow_champion_task.fn(target_version="1")

        out = switch_mlflow_champion_task.fn(target_version="2", reason="regression_fix")

    assert out == {"previous_version": "3", "target_version": "2", "run_id": "run_good"}
    mock_client.set_registered_model_alias.assert_any_call("bot_detector", "rolled_back", "3")
    mock_client.set_registered_model_alias.assert_any_call("bot_detector", "champion", "2")


def test_invalidate_prediction_cache_only_clears_dialog_scores() -> None:
    """Очистка Redis кэша удаляет только dialog_score:* и не трогает async_task:*."""

    class _FakeRedis:
        def __init__(self) -> None:
            self.store = {
                "dialog_score:abc": "1",
                "dialog_score:def": "2",
                "async_task:t1": "keep",
            }

        def scan(self, cursor: int = 0, match: str = "*", count: int = 100):
            prefix = match.removesuffix("*")
            return 0, [k for k in list(self.store) if k.startswith(prefix)]

        def delete(self, *keys: str) -> int:
            removed = 0
            for k in keys:
                if k in self.store:
                    del self.store[k]
                    removed += 1
            return removed

        def close(self) -> None:
            pass

    fake_r = _FakeRedis()
    with patch("redis.Redis.from_url", return_value=fake_r):
        removed = invalidate_prediction_cache_task.fn()

    assert removed == 2
    assert list(fake_r.store) == ["async_task:t1"]


def test_rollback_flow_interactive_pick_and_window_reset() -> None:
    """Интерактивный режим через prompt_target_version_interactively, публикация, очистка кэша и ресет окна."""
    candidates = [
        ModelVersionCandidate("4", "run_4", 400.0, 0.14, ["@champion"], "v4 | 2026-11-18 | loss=0.1400 | @champion"),
        ModelVersionCandidate("3", "run_3", 300.0, 0.15, [], "v3 | 2026-11-17 | loss=0.1500"),
    ]
    fake_store: dict[str, str] = {"dialog_score:h1": "{}"}

    class _FakeRedis:
        def get(self, key: str) -> str | None:
            return fake_store.get(key)

        def set(self, key: str, val: str, **_kwargs: object) -> None:
            fake_store[key] = val

        def scan(self, cursor: int = 0, match: str = "*", count: int = 100):
            prefix = match.removesuffix("*")
            return 0, [k for k in list(fake_store) if k.startswith(prefix)]

        def delete(self, *keys: str) -> int:
            cnt = 0
            for k in keys:
                if k in fake_store:
                    del fake_store[k]
                    cnt += 1
            return cnt

        def close(self) -> None:
            pass

    fake_r = _FakeRedis()
    with (
        patch(
            "project.pipeline.model_rollback.inspector.ModelVersionInspector.list_candidates", return_value=candidates
        ),
        patch(
            "project.pipeline.model_rollback.interactive.RollbackPromptUI.prompt_target_version",
            return_value="3",
        ) as mock_prompt,
        patch(
            "project.pipeline.model_rollback.flow.switch_mlflow_champion_task",
            return_value={"previous_version": "4", "target_version": "3", "run_id": "run_3"},
        ),
        patch("project.pipeline.model_rollback.flow.export_onnx_task") as mock_export,
        patch(
            "project.pipeline.model_rollback.flow.hot_reload_triton_task",
            return_value={"meta_stacker": True},
        ) as mock_reload,
        patch("project.pipeline.model_rollback.flow.verify_triton_health_task", return_value=True),
        patch("project.pipeline.model_rollback.tasks.DWHFeatureFetcher") as mock_fetcher_cls,
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.reset_baseline") as mock_reset_baseline,
        patch("project.pipeline.metrics_sync.DriftMetricsSyncService.trigger_api_gateway_metrics") as mock_metrics_sync,
        patch("redis.Redis.from_url", return_value=fake_r),
    ):
        mock_fetcher_cls.return_value.get_feature_names.return_value = ["f1", "f2"]

        res = bot_detection_rollback_flow(mode=RollbackMode.INTERACTIVE_PICK, reason="drift_regression")

    assert res.status == "completed"
    assert res.previous_version == "4"
    assert res.rolled_back_to_version == "3"
    assert res.target_run_id == "run_3"
    assert res.cache_invalidated_keys == 1
    assert res.drift_window_reset is True
    assert "dialog_score:h1" not in fake_store
    assert json.loads(fake_store["bot_detector:champion_effective_since_ts"])["version"] == "3"
    mock_prompt.assert_called_once()
    mock_export.assert_called_once()
    mock_reload.assert_called_once()
    mock_reset_baseline.assert_called_once()
    mock_metrics_sync.assert_called_once()


def test_get_champion_model_info_respects_rollback_effective_since() -> None:
    """ChampionMetadataReader.read использует timestamp отката вместо старого creation_timestamp версии MLflow."""
    old_created_ms = (time.time() - 10 * 86400) * 1000.0
    rollback_ts = time.time() - 60.0

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = SimpleNamespace(
        version="2",
        creation_timestamp=old_created_ms,
        run_id="",
    )
    mock_redis = MagicMock()
    mock_redis.get.return_value = json.dumps({"version": "2", "ts": rollback_ts})

    with (
        patch("project.drift.model_age.MlflowClient", return_value=mock_client),
        patch("redis.Redis.from_url", return_value=mock_redis),
    ):
        info = ChampionMetadataReader.read("bot_detector")

    assert info["version"] == "2"
    assert abs(info["created_at_ts"] - rollback_ts) < 1e-3
    assert info["age_days"] < 0.05


def test_deploy_rollback_flow_applies_deployment() -> None:
    """Проверка регистрации deployment bot-detection-rollback в Prefect."""
    mock_dep = MagicMock()
    mock_dep.apply.return_value = "rollback-dep-id"

    with patch.object(bot_detection_rollback_flow, "to_deployment", return_value=mock_dep) as mock_to_dep:
        dep_id = RollbackFlowDeployer.deploy()

    assert dep_id == "rollback-dep-id"
    assert mock_to_dep.call_args.kwargs["name"] == "bot-detection-rollback"
    assert mock_dep.entrypoint == "project.pipeline.model_rollback.flow:bot_detection_rollback_flow"
    mock_dep.apply.assert_called_once()


def test_switch_mlflow_champion_clears_or_shifts_stale_challenger(tmp_path: Path) -> None:
    """Откат снимает залипший @challenger с цели/rolled_back либо переносит его на третьего кандидата."""
    good_onnx = tmp_path / "good" / "meta_stacker" / "model.onnx"
    _make_dummy_onnx(good_onnx, input_dim=3)

    # Кейс A: есть только v12 (@champion) и v10 (@challenger + @rolled_back) -> откат к v10 удаляет залипший @challenger
    mock_client_a = MagicMock()
    mock_client_a.get_run.return_value = SimpleNamespace(outputs=None)
    mock_client_a.list_artifacts.return_value = [
        SimpleNamespace(path=p) for p in ("catboost_detector", "lightgbm_detector", "xgboost_detector", "meta_stacker")
    ]
    mock_client_a.get_model_version.return_value = SimpleNamespace(version="10", run_id="run_v10")
    mock_client_a.download_artifacts.return_value = str(good_onnx)
    aliases_a = {"champion": "12", "challenger": "10", "rolled_back": "10"}
    mock_client_a.get_model_version_by_alias.side_effect = lambda _m, a: SimpleNamespace(version=aliases_a[a])

    with (
        patch("project.pipeline.model_rollback.tasks.MlflowClient", return_value=mock_client_a),
        patch("project.pipeline.model_rollback.inspector.MlflowClient", return_value=mock_client_a),
        patch("project.tracking.experiment_tracker.loggers.onnx_resolver.MlflowClient", return_value=mock_client_a),
        patch(
            "project.pipeline.model_rollback.inspector.ModelVersionInspector.list_candidates",
            return_value=[SimpleNamespace(version="12"), SimpleNamespace(version="10")],
        ),
    ):
        out_a = switch_mlflow_champion_task.fn(target_version="10", reason="bad_v12")

    assert out_a == {"previous_version": "12", "target_version": "10", "run_id": "run_v10"}
    mock_client_a.set_registered_model_alias.assert_any_call("bot_detector", "rolled_back", "12")
    mock_client_a.set_registered_model_alias.assert_any_call("bot_detector", "champion", "10")
    mock_client_a.delete_registered_model_alias.assert_called_once_with("bot_detector", "challenger")

    # Кейс B: при наличии третьего кандидата v8 @challenger сдвигается на него
    mock_client_b = MagicMock()
    mock_client_b.get_run.return_value = SimpleNamespace(outputs=None)
    mock_client_b.list_artifacts.return_value = mock_client_a.list_artifacts.return_value
    mock_client_b.get_model_version.return_value = SimpleNamespace(version="10", run_id="run_v10")
    mock_client_b.download_artifacts.return_value = str(good_onnx)
    aliases_b = {"champion": "12", "challenger": "10", "rolled_back": "9"}
    mock_client_b.get_model_version_by_alias.side_effect = lambda _m, a: SimpleNamespace(version=aliases_b[a])

    with (
        patch("project.pipeline.model_rollback.tasks.MlflowClient", return_value=mock_client_b),
        patch("project.pipeline.model_rollback.inspector.MlflowClient", return_value=mock_client_b),
        patch("project.tracking.experiment_tracker.loggers.onnx_resolver.MlflowClient", return_value=mock_client_b),
        patch(
            "project.pipeline.model_rollback.inspector.ModelVersionInspector.list_candidates",
            return_value=[SimpleNamespace(version="12"), SimpleNamespace(version="10"), SimpleNamespace(version="8")],
        ),
    ):
        switch_mlflow_champion_task.fn(target_version="10", reason="bad_v12")

    mock_client_b.set_registered_model_alias.assert_any_call("bot_detector", "challenger", "8")
    mock_client_b.delete_registered_model_alias.assert_not_called()
