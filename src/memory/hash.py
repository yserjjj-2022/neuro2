"""Functional Core — content hashing for deduplication.

Pure function, no I/O.
"""

from __future__ import annotations

import hashlib


def content_hash(content: str) -> str:
    """SHA-256 хеш контента для дедупликации.

    Чистая функция: str → str, без I/O.

    Args:
        content: Текст содержания эпизода.

    Returns:
        Hex-строка SHA-256 (64 символа).
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
