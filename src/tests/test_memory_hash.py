"""Unit tests for content_hash — pure core, no SQLite."""

from src.memory.hash import content_hash


def test_content_hash_deterministic() -> None:
    """Одинаковый content → одинаковый hash."""
    assert content_hash("same text") == content_hash("same text")


def test_content_hash_different() -> None:
    """Разный content → разный hash."""
    assert content_hash("one") != content_hash("two")


def test_content_hash_unicode() -> None:
    """Unicode content корректно хешируется."""
    h = content_hash("привет, мир 🌍")
    assert len(h) == 64  # SHA-256 hex = 64 символа
    assert h == content_hash("привет, мир 🌍")


def test_content_hash_empty_string() -> None:
    """Пустая строка — валидный вход (не исключение)."""
    assert len(content_hash("")) == 64
