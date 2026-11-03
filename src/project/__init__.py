from importlib.metadata import PackageNotFoundError, version

from project.core.paths import paths
from project.core.utils import Reproducibility

try:
    __version__ = version("project")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "Reproducibility",
    "__version__",
    "paths",
]
