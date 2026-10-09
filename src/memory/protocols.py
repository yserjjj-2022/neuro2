"""Protocols for DI — memory access abstraction.

Allow energy/tm/wiring modules to depend on the abstraction
rather than a concrete MemoryStore. Analogous to SupportsWrite
from telemetry.
"""

from __future__ import annotations

from typing import Protocol

from .models import Episode
from .serialize import Vector


class SupportsStore(Protocol):
    """Protocol для записи в память.

    Raises:
        ValueError: При несовпадении размерности эмбеддинга.
        MemoryStoreError: При сбое I/O.
    """

    def store(self, episode: Episode) -> int: ...


class SupportsRecall(Protocol):
    """Protocol для чтения из памяти.

    Raises:
        ValueError: При несовпадении размерности эмбеддинга.
        MemoryStoreError: При сбое I/O.
    """

    def recall(self, query_embedding: Vector, limit: int = 5) -> list[Episode]: ...


class SupportsConsolidate(Protocol):
    """Protocol для консолидации памяти (S6).

    Читает все эпизоды, удаляет незначимые, сохраняет схемы.
    """

    def all_episodes(self) -> list[Episode]: ...

    def delete(self, ids: list[int]) -> int: ...

    def save_schema(self, centroid: Vector, member_count: int, summary: str) -> int: ...

    def count(self) -> int: ...  # S6 проход 2: ночной цикл


class SupportsMemory(SupportsStore, SupportsRecall, SupportsConsolidate, Protocol):
    """Комбинированный контракт хранилища памяти (store + recall + consolidate).

    Объединяет три протокола в один, чтобы ``MemoryRouter`` мог принять
    ``MemoryStore`` без синтаксиса пересечения типов (``A & B`` не поддерживается
    mypy в аннотациях).
    """
