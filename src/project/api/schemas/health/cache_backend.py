from enum import StrEnum


class CacheBackend(StrEnum):
    """Тип активного хранилища кэша."""

    REDIS = "redis"
    DISABLED = "disabled"
