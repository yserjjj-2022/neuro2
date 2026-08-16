"""Unit tests for MemoryStore — schema, store(), recall(), atomicity."""

import sqlite3
from pathlib import Path
from typing import Self

import numpy as np
import pytest

from src.memory.errors import MemoryStoreError
from src.memory.models import Episode
from src.memory.store import MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FailingConnection:
    """Прокси над реальным sqlite3.Connection: execute кидает операционную ошибку.

    sqlite3.Connection — immutable C-тип, patch.object не работает.
    Прокси делегирует транзакции (__enter__/__exit__) реальному соединению,
    а execute перехватывает и бросает sqlite3.OperationalError на указанном
    по счёту вызове (или на всех, если fail_on_call is None).
    """

    def __init__(self, real: sqlite3.Connection, fail_on_call: int | None = None):
        self._real = real
        self._call = 0
        self._fail_on_call = fail_on_call

    def __enter__(self) -> Self:
        self._real.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._real.__exit__(*exc)

    def close(self) -> None:
        """Делегировать закрытие реальному соединению."""
        self._real.close()

    def execute(self, sql, params=()):
        self._call += 1
        if self._fail_on_call is None or self._call == self._fail_on_call:
            raise sqlite3.OperationalError("simulated I/O failure")
        return self._real.execute(sql, params)


def _make_episode(
    content: str = "test",
    embedding: np.ndarray | None = None,
    id: int | None = None,
) -> Episode:
    if embedding is None:
        embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    return Episode(
        content=content,
        embedding=embedding,
        timestamp=1.0,
        valence=2.0,
        stress=3.0,
        free_energy=4.0,
        id=id,
    )


# ---------------------------------------------------------------------------
# Smoke tests (step 7)
# ---------------------------------------------------------------------------


def test_schema_creation() -> None:
    """Таблицы episodes и episode_vectors существуют после __init__."""
    store = MemoryStore(db_path=":memory:", embedding_dim=4)
    assert store._conn is not None

    tables = {
        row[0]
        for row in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "episodes" in tables
    assert "episode_vectors" in tables

    store.close()


def test_episodes_schema_columns() -> None:
    """Колонки таблицы episodes соответствуют Episode."""
    store = MemoryStore(db_path=":memory:", embedding_dim=4)
    assert store._conn is not None
    columns = {row[1] for row in store._conn.execute("PRAGMA table_info(episodes)")}
    assert columns == {
        "id",
        "content",
        "content_hash",
        "embedding",
        "timestamp",
        "valence",
        "stress",
        "free_energy",
    }
    store.close()


def test_sqlite_vec_loaded() -> None:
    """Расширение vec0 загружено (smoke на :memory:)."""
    store = MemoryStore(db_path=":memory:", embedding_dim=4)
    assert store._conn is not None
    version = store._conn.execute("SELECT vec_version()").fetchone()[0]
    assert version.startswith("v0.1.9")
    store.close()


def test_init_rejects_invalid_embedding_dim(tmp_path: Path) -> None:
    """0, отрицательное, не-int → ValueError ДО подключения к БД."""
    db_path = tmp_path / "should_not_exist.db"

    for bad in (0, -1, 3.14, "4", None):
        with pytest.raises(ValueError):
            MemoryStore(db_path=db_path, embedding_dim=bad)  # type: ignore[arg-type]

    # Некорректное значение не создаёт файл БД
    assert not db_path.exists()


def test_store_opens_file_based_db(tmp_path: Path) -> None:
    """Файловая БД: файл создаётся, схема работает."""
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=db_path, embedding_dim=4)

    assert db_path.exists()
    assert store._conn is not None
    store.close()


def test_close_twice_no_error() -> None:
    """close() дважды без ошибки (идемпотентен)."""
    store = MemoryStore(db_path=":memory:", embedding_dim=4)
    store.close()
    store.close()  # не должно упасть


# ---------------------------------------------------------------------------
# store() tests (step 8-9)
# ---------------------------------------------------------------------------


def test_store_inserts_episode() -> None:
    """store() → id, эпизод в БД."""
    store = MemoryStore(db_path=":memory:", embedding_dim=4)
    ep = _make_episode(content="hello world")

    ep_id = store.store(ep)

    assert isinstance(ep_id, int)
    assert ep_id > 0
    store.close()


def test_store_duplicate_returns_same_id() -> None:
    """Повторный store() того же content → тот же id, vec0 не затрагивается."""
    store = MemoryStore(db_path=":memory:", embedding_dim=4)
    ep = _make_episode(content="duplicate test")

    id1 = store.store(ep)
    id2 = store.store(ep)

    assert id1 == id2
    store.close()


def test_store_dim_mismatch() -> None:
    """ValueError при embedding.shape[0] != embedding_dim."""
    store = MemoryStore(db_path=":memory:", embedding_dim=4)
    ep = _make_episode(
        content="bad dim",
        embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),  # dim=3
    )

    with pytest.raises(ValueError):
        store.store(ep)
    store.close()


def test_store_memory_store_error_on_sqlite_failure() -> None:
    """sqlite3.Error при store() → MemoryStoreError (после логирования)."""
    store = MemoryStore(db_path=":memory:", embedding_dim=4)
    ep = _make_episode()

    assert store._conn is not None
    store._conn = _FailingConnection(store._conn)  # type: ignore[assignment]

    with pytest.raises(MemoryStoreError):
        store.store(ep)

    store.close()


def test_store_atomicity_rollback() -> None:
    """Ошибка на vec0-insert → episodes НЕ содержит orphan-строку.

    Прокси падает на втором execute (vec0-insert). Транзакция
    откатывается целиком — episodes не содержит записи с этим content_hash.
    """
    store = MemoryStore(db_path=":memory:", embedding_dim=4)
    ep = _make_episode(content="atomicity test")

    assert store._conn is not None
    store._conn = _FailingConnection(store._conn, fail_on_call=2)  # type: ignore[assignment]

    with pytest.raises(MemoryStoreError):
        store.store(ep)

    # Проверяем, что orphan-записи нет
    assert store._conn is not None
    from src.memory.hash import content_hash

    c_hash = content_hash("atomicity test")
    row = store._conn.execute(
        "SELECT id FROM episodes WHERE content_hash = ?", (c_hash,)
    ).fetchone()
    assert row is None

    store.close()


# ---------------------------------------------------------------------------
# recall() tests (step 10-11)
# ---------------------------------------------------------------------------


def test_recall_empty_db() -> None:
    """Пустая БД → [] (не ошибка, не исключение)."""
    store = MemoryStore(db_path=":memory:", embedding_dim=4)

    results = store.recall(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))

    assert results == []
    store.close()


def test_recall_top_k_order() -> None:
    """recall() возвращает top-k по близости, порядок строго проверяется.

    Создаём 3 эпизода с заведомо разной близостью к запросу:
    - ep1: близко к запросу
    - ep2: средне
    - ep3: далеко
    Проверяем [e.id for e in results] == [ep1_id, ep2_id, ep3_id]
    (не set, а список — порядок важен).
    """
    store = MemoryStore(db_path=":memory:", embedding_dim=2)

    query = np.array([1.0, 0.0], dtype=np.float32)

    ep1 = _make_episode(
        content="closest",
        embedding=np.array([0.9, 0.1], dtype=np.float32),
    )
    ep2 = _make_episode(
        content="medium",
        embedding=np.array([0.5, 0.5], dtype=np.float32),
    )
    ep3 = _make_episode(
        content="farthest",
        embedding=np.array([0.0, 1.0], dtype=np.float32),
    )

    id1 = store.store(ep1)
    id2 = store.store(ep2)
    id3 = store.store(ep3)

    results = store.recall(query, limit=3)

    # Строгая проверка ПОРЯДКА, не только состава
    assert [r.id for r in results] == [id1, id2, id3]
    store.close()


def test_recall_dim_mismatch() -> None:
    """ValueError при query_embedding.shape[0] != embedding_dim."""
    store = MemoryStore(db_path=":memory:", embedding_dim=4)

    with pytest.raises(ValueError):
        store.recall(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    store.close()


def test_recall_memory_store_error() -> None:
    """sqlite3.Error при recall() → MemoryStoreError (НЕ [] как сигнал)."""
    store = MemoryStore(db_path=":memory:", embedding_dim=4)

    assert store._conn is not None
    store._conn = _FailingConnection(store._conn)  # type: ignore[assignment]

    with pytest.raises(MemoryStoreError):
        store.recall(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))

    store.close()


# ---------------------------------------------------------------------------
# Persistence test (step 12)
# ---------------------------------------------------------------------------


def test_persistence_across_reopen(tmp_path: Path) -> None:
    """store → close → reopen (тот же db_path) → recall находит."""
    db_path = tmp_path / "persistence.db"
    embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    # Записываем
    store1 = MemoryStore(db_path=db_path, embedding_dim=4)
    ep = _make_episode(content="persistent episode", embedding=embedding)
    store1.store(ep)
    store1.close()

    # Переоткрываем
    store2 = MemoryStore(db_path=db_path, embedding_dim=4)
    results = store2.recall(embedding, limit=1)

    assert len(results) == 1
    assert results[0].content == "persistent episode"
    store2.close()
