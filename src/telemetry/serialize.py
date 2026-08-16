"""Functional Core — pure serialization.

No I/O, no file system access. Input → output.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from .models import TelemetryEvent

logger = logging.getLogger(__name__)


def serialize_event(event: TelemetryEvent) -> str:
    """Сериализация TelemetryEvent в JSON-строку.

    Чистая функция: вход → выход, без I/O, без файлов.
    Тестируется без tmp_path, без mock — просто input → output.

    Args:
        event: Событие для сериализации.

    Returns:
        JSON-строка (валидный JSON по RFC 8259).

    Raises:
        ValueError: Если event содержит NaN/Infinity (невалидный JSON).
    """
    data = asdict(event)
    try:
        return json.dumps(data, allow_nan=False, ensure_ascii=False)
    except ValueError as exc:
        logger.error("Cannot serialize event with NaN/Infinity: %s", exc)
        raise ValueError(
            f"TelemetryEvent contains NaN or Infinity values: {data}"
        ) from exc
