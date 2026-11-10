"""Модульные тесты менеджера версионирования моделей Triton."""

from pathlib import Path

import pytest

from project.infrastructure.triton.version_manager import TritonModelVersionManager


def test_version_manager_empty_repo(tmp_path: Path) -> None:
    """Проверка поведения на пустом репозитории."""
    vm = TritonModelVersionManager(repository_path=tmp_path)

    assert vm.get_versions("non_existent") == []
    assert vm.get_latest_version("non_existent") is None
    assert vm.get_next_version("non_existent") == 1
    assert vm.get_model_path("non_existent") == tmp_path / "non_existent" / "1" / "model.onnx"
    assert vm.prune_old_versions("non_existent") == []


def test_version_manager_numerical_sorting_and_filtering(tmp_path: Path) -> None:
    """Проверка численной сортировки версий и игнорирования нечисловых папок."""
    model_dir = tmp_path / "catboost_detector"
    model_dir.mkdir(parents=True)

    # Создаем версии 1, 2, 10, а также нечисловые элементы
    (model_dir / "1").mkdir()
    (model_dir / "10").mkdir()
    (model_dir / "2").mkdir()
    (model_dir / "temp_folder").mkdir()
    (model_dir / "config.pbtxt").write_text("dummy")

    vm = TritonModelVersionManager(repository_path=tmp_path)

    assert vm.get_versions("catboost_detector") == [1, 2, 10]
    assert vm.get_latest_version("catboost_detector") == 10
    assert vm.get_next_version("catboost_detector") == 11
    assert vm.get_model_path("catboost_detector") == model_dir / "10" / "model.onnx"
    assert vm.get_model_path("catboost_detector", version=2) == model_dir / "2" / "model.onnx"


def test_version_manager_prepare_version_dir(tmp_path: Path) -> None:
    """Проверка создания директории версии."""
    vm = TritonModelVersionManager(repository_path=tmp_path)

    ver1_dir = vm.prepare_version_dir("model_x")
    assert ver1_dir == tmp_path / "model_x" / "1"
    assert ver1_dir.is_dir()

    ver5_dir = vm.prepare_version_dir("model_x", version=5)
    assert ver5_dir == tmp_path / "model_x" / "5"
    assert ver5_dir.is_dir()


def test_version_manager_publish_and_rotation(tmp_path: Path) -> None:
    """Проверка публикации новых версий и ротации старых до keep_versions=3."""
    src_model = tmp_path / "source_model.onnx"
    src_model.write_bytes(b"model_binary_payload")

    vm = TritonModelVersionManager(repository_path=tmp_path, keep_versions=3)

    # Публикация версий 1, 2, 3
    v1 = vm.publish_model("detector", src_model)
    assert v1 == 1
    assert (tmp_path / "detector" / "1" / "model.onnx").read_bytes() == b"model_binary_payload"

    v2 = vm.publish_model("detector", src_model)
    assert v2 == 2

    v3 = vm.publish_model("detector", src_model)
    assert v3 == 3
    assert vm.get_versions("detector") == [1, 2, 3]

    # Публикация версии 4 -> версия 1 должна удалиться, остаются 2, 3, 4
    v4 = vm.publish_model("detector", src_model)
    assert v4 == 4
    assert vm.get_versions("detector") == [2, 3, 4]
    assert not (tmp_path / "detector" / "1").exists()
    assert (tmp_path / "detector" / "2").exists()

    # Публикация версии 5 -> версия 2 должна удалиться, остаются 3, 4, 5
    v5 = vm.publish_model("detector", src_model)
    assert v5 == 5
    assert vm.get_versions("detector") == [3, 4, 5]
    assert not (tmp_path / "detector" / "2").exists()


def test_version_manager_publish_missing_source_file(tmp_path: Path) -> None:
    """Проверка ошибки при попытке опубликовать несуществующий файл."""
    vm = TritonModelVersionManager(repository_path=tmp_path)
    with pytest.raises(FileNotFoundError, match="Source model file does not exist"):
        vm.publish_model("detector", tmp_path / "does_not_exist.onnx")
