"""MemoryStoreError — custom exception wrapping sqlite3 errors."""

from __future__ import annotations


class MemoryStoreError(Exception):
    """Собственное исключение — оборачивает sqlite3 ошибки.

    В отличие от TelemetryLogger.log() (fire-and-forget, ничего не возвращает),
    store() возвращает id — значимое значение для вызывающего кода.
    Молчаливое проглатывание ошибки с None/-1 создаёт риск, что caller
    продолжит работу с несуществующей записью.

    Поэтому: ошибка логируется через logging.error(), затем пробрасывается
    как MemoryStoreError. Вызывающий код сам решает, ловить или нет.
    """
