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

    def recall(
        self, query_embedding: Vector, limit: int = 5
    ) -> list[Episode]: ...
