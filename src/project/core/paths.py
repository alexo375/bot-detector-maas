from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawDataPaths:
    """Пути к исходным соревновательным файлам (используются исключительно в слое сидирования DWH)."""

    train_json: Path
    test_json: Path
    ytrain_csv: Path
    ytest_csv: Path

    @classmethod
    def from_root(cls, root: Path) -> "RawDataPaths":
        """Создает набор путей к raw JSON/CSV файлам от корня проекта."""
        data = root / "data"
        return cls(
            train_json=data / "train.json",
            test_json=data / "test.json",
            ytrain_csv=data / "ytrain.csv",
            ytest_csv=data / "ytest.csv",
        )


@dataclass(frozen=True)
class ProjectPaths:
    """Определение путей к директориям проекта, конфигураций и артефактов."""

    root: Path = Path(__file__).resolve().parents[3]
    env_file: Path = root / ".env"

    data: Path = root / "data"
    deploy: Path = root / "deploy"
    triton_models: Path = deploy / "triton" / "model_repository"

    @property
    def raw(self) -> RawDataPaths:
        """Локализованные пути к исходным соревновательным файлам (для сидера DWH)."""
        return RawDataPaths.from_root(self.root)


paths = ProjectPaths()
