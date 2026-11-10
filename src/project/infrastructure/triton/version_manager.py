import logging
import re
import shutil
from pathlib import Path

from project.core.config import config

logger = logging.getLogger(__name__)


class TritonModelVersionManager:
    """Управление версиями моделей Triton: создание числовых версий, ротация и резолв путей."""

    def __init__(
        self,
        repository_path: str | Path | None = None,
        keep_versions: int = 3,
    ) -> None:
        """Инициализация менеджера версий.

        Args:
            repository_path: Корневой путь к Triton model repository.
            keep_versions: Количество последних числовых версий, сохраняемых на диске.

        """
        self.repository_path = (
            Path(repository_path) if repository_path is not None else Path(config.triton.model_repository)
        )
        self.keep_versions = max(1, keep_versions)

    def get_model_dir(self, model_name: str) -> Path:
        """Получить путь к директории конкретной модели в репозитории."""
        return self.repository_path / model_name

    def get_versions(self, model_name: str) -> list[int]:
        """Получить отсортированный список всех существующих числовых версий модели."""
        model_dir = self.get_model_dir(model_name)
        if not model_dir.exists():
            return []
        versions = [int(p.name) for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        return sorted(versions)

    def get_latest_version(self, model_name: str) -> int | None:
        """Получить номер самой последней активной версии модели."""
        versions = self.get_versions(model_name)
        return versions[-1] if versions else None

    def get_model_path(
        self,
        model_name: str,
        version: int | None = None,
        filename: str = "model.onnx",
    ) -> Path:
        """Получить путь к бинарному файлу модели.

        Если версия не указана, возвращается путь к файлу последней активной версии.
        """
        target_version = version if version is not None else self.get_latest_version(model_name)
        ver_str = str(target_version) if target_version is not None else "1"
        return self.get_model_dir(model_name) / ver_str / filename

    def get_next_version(self, model_name: str) -> int:
        """Вычислить номер следующей инкрементальной версии."""
        versions = self.get_versions(model_name)
        return (max(versions) + 1) if versions else 1

    def prepare_version_dir(self, model_name: str, version: int | None = None) -> Path:
        """Создать директорию для новой версии модели."""
        ver = version if version is not None else self.get_next_version(model_name)
        ver_dir = self.get_model_dir(model_name) / str(ver)
        ver_dir.mkdir(parents=True, exist_ok=True)
        return ver_dir

    def publish_model(
        self,
        model_name: str,
        src_model_path: Path | str,
        target_filename: str = "model.onnx",
        extra_files: dict[str, Path] | None = None,
    ) -> int:
        """Опубликовать новую версию модели из исходного файла и выполнить ротацию устаревших версий.

        Args:
            model_name: Имя модели Triton (например, 'catboost_detector').
            src_model_path: Путь к скачанному или экспортированному файлу модели.
            target_filename: Имя файла в целевой директории версии (по умолчанию 'model.onnx').
            extra_files: Дополнительные файлы версии (например, файлы токенизатора):
                имя_в_версии -> путь_к_исходнику.

        Returns:
            Номер созданной числовой версии.

        """
        src_path = Path(src_model_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Source model file does not exist: {src_path}")

        next_version = self.get_next_version(model_name)
        version_dir = self.prepare_version_dir(model_name, version=next_version)
        target_file = version_dir / target_filename

        shutil.copy2(src_path, target_file)
        # Если рядом с model.onnx существует файл внешних весов (например, model.onnx.data)
        src_data = src_path.parent / f"{src_path.name}.data"
        if src_data.is_file():
            shutil.copy2(src_data, version_dir / f"{target_filename}.data")

        for filename, src_extra in (extra_files or {}).items():
            shutil.copy2(src_extra, version_dir / filename)

        logger.info(
            "Published model '%s' version %d -> %s (%d extra files)",
            model_name,
            next_version,
            target_file,
            len(extra_files or {}),
        )

        self.prune_old_versions(model_name)
        return next_version

    def prune_old_versions(self, model_name: str) -> list[int]:
        """Удалить старые версии модели, превышающие лимит keep_versions.

        Returns:
            Список удаленных номеров версий.

        """
        versions = self.get_versions(model_name)
        if len(versions) <= self.keep_versions:
            return []

        to_prune = versions[: -self.keep_versions]
        model_dir = self.get_model_dir(model_name)
        for ver in to_prune:
            prune_path = model_dir / str(ver)
            if prune_path.exists():
                shutil.rmtree(prune_path)
                logger.info(
                    "Pruned obsolete version %d of model '%s' (%s)",
                    ver,
                    model_name,
                    prune_path,
                )
        return to_prune

    def update_model_input_dim(
        self,
        model_name: str,
        feature_dim: int,
        input_name: str = "features",
    ) -> bool:
        """Обновляет размерность входного тензора (`dims: [ N ]`) в `config.pbtxt` Triton."""
        config_path = self.get_model_dir(model_name) / "config.pbtxt"
        if not config_path.exists():
            raise FileNotFoundError(f"Triton config.pbtxt not found for model '{model_name}': {config_path}")

        raw_text = config_path.read_text(encoding="utf-8")
        pattern = re.compile(
            rf'(name:\s*"{re.escape(input_name)}"[^}}]*dims:\s*\[\s*)\d+(\s*\])',
            re.DOTALL,
        )
        updated_text, count = pattern.subn(rf"\g<1>{int(feature_dim)}\g<2>", raw_text, count=1)
        if count == 0:
            raise ValueError(f"Could not locate input '{input_name}' dims in {config_path}")
        if updated_text != raw_text:
            config_path.write_text(updated_text, encoding="utf-8")
            logger.info("Updated Triton config '%s' input '%s' dims -> [%d]", model_name, input_name, feature_dim)
        return True
