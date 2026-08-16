"""Imperative Shell — file I/O management.

Owns the log file, delegates serialization to the functional core.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from .models import TelemetryEvent
from .serialize import serialize_event

logger = logging.getLogger(__name__)


class TelemetryWriter:
    """Imperative Shell: единственное место I/O в модуле.

    Делегирует сериализацию чистому ядру (serialize_event),
    управляет файлом на диске.

    Открывает файл один раз при __init__, закрывает при close().
    flush() после каждой записи для crash-safety.

    Attributes:
        log_path: Путь к JSONL-файлу для записи.
        _file: Открытый файловый объект (None до первого write или после close).
    """

    def __init__(self, log_path: Path) -> None:
        """Инициализация writer.

        Args:
            log_path: Путь к JSONL-файлу для записи.
        """
        self.log_path = log_path
        self._file: io.TextIOWrapper | None = None
        self._open_file()

    def _open_file(self) -> None:
        """Открыть файл для дозаписи (создать, если не существует)."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115 — файл держится открытым между write()

    def write(self, event: TelemetryEvent) -> None:
        """Записать одно событие в JSONL.

        1. Вызывает serialize_event(event) для получения строки
        2. Записывает строку + "\\n" в файл
        3. Вызывает self._file.flush() для crash-safety

        Args:
            event: Событие для записи.

        Raises:
            RuntimeError: Если writer закрыт.
        """
        if self._file is None:
            raise RuntimeError("Writer is closed, cannot write")

        line = serialize_event(event) + "\n"
        try:
            self._file.write(line)
            self._file.flush()
        except OSError as exc:
            logger.error(
                "Failed to write telemetry event to %s: %s", self.log_path, exc
            )
            raise

    def close(self) -> None:
        """Закрыть файл (опционально, для cleanup)."""
        if self._file is not None:
            try:
                self._file.close()
            except OSError as exc:
                logger.error(
                    "Failed to close telemetry log file %s: %s", self.log_path, exc
                )
            finally:
                self._file = None
