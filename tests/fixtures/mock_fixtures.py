"""Модульные фикстуры тестовых дублёров (test doubles, mocks) для Redis и кэша диалогов."""

from typing import Any


class FakeRedisClient:
    """Изолированный in-memory мок клиента Redis для модульных тестов API."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def mget(self, keys: list[str]) -> list[str | None]:
        return [self.store.get(k) for k in keys]

    def pipeline(self, transaction: bool = False) -> Any:
        del transaction

        class FakePipeline:
            def __init__(self, parent: FakeRedisClient) -> None:
                self.parent = parent
                self.actions: list[tuple[str, str]] = []

            def setex(self, key: str, ttl: int, value: str) -> "FakePipeline":
                del ttl
                self.actions.append((key, value))
                return self

            def execute(self) -> None:
                for k, v in self.actions:
                    self.parent.store[k] = v

        return FakePipeline(self)

    def setex(self, key: str, ttl: int, value: str) -> bool:
        del ttl
        self.store[key] = value
        return True

    def delete(self, key: str) -> int:
        return 1 if self.store.pop(key, None) is not None else 0

    def close(self) -> None:
        pass


class FakeDialogueCache:
    """In-memory мок DialogueCache для изоляции воркеров и API без Redis."""

    def __init__(self) -> None:
        self.dialogues: dict[str, Any] = {}
        self.tasks: dict[str, Any] = {}

    @property
    def is_connected(self) -> bool:
        return True

    def get(self, key: str) -> Any:
        return self.dialogues.get(key)

    def mget(self, keys: list[str]) -> list[Any]:
        return [self.dialogues.get(k) for k in keys]

    def set(self, key: str, val: Any) -> None:
        self.dialogues[key] = val.to_dict() if hasattr(val, "to_dict") else val

    def mset(self, items: list[tuple[str, Any]], ttl_seconds: int | None = None) -> None:
        del ttl_seconds
        for k, v in items:
            self.dialogues[k] = v.to_dict() if hasattr(v, "to_dict") else v

    def get_task(self, task_id: str) -> Any:
        return self.tasks.get(task_id)

    def set_task(self, task_id: str, val: Any) -> None:
        self.tasks[task_id] = val
