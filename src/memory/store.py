"""MemoryStore — imperative shell owning the SQLite connection.

The only place in the memory module that touches SQLite. Pure core
(cosine_similarity, serialize_embedding, content_hash) stays separate.

Opens connection in __init__, loads sqlite-vec extension, creates schema.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import sqlite_vec  # type: ignore[import-not-found]  # пакет без стабов

from .errors import MemoryStoreError
from .hash import content_hash
from .models import Episode
from .serialize import Vector, deserialize_embedding, serialize_embedding

logger = logging.getLogger(__name__)

_SCHEMA_EPISODES = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    embedding BLOB NOT NULL,
    timestamp REAL NOT NULL,
    valence REAL NOT NULL,
    stress REAL NOT NULL,
    free_energy REAL NOT NULL
)
"""

_INSERT_EPISODE = """
INSERT OR IGNORE INTO episodes
    (content, content_hash, embedding, timestamp, valence, stress, free_energy)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_ID_BY_HASH = "SELECT id FROM episodes WHERE content_hash = ?"

_INSERT_VECTOR = "INSERT INTO episode_vectors (rowid, embedding) VALUES (?, ?)"

_RECALL_SQL = """
SELECT e.id, e.content, e.embedding, e.timestamp,
       e.valence, e.stress, e.free_energy, v.distance
FROM episodes e
JOIN (SELECT rowid, distance FROM episode_vectors
      WHERE embedding MATCH ? ORDER BY distance LIMIT ?) v
    ON e.id = v.rowid
ORDER BY v.distance
"""


class MemoryStore:
    """Imperative Shell: единственное место SQLite I/O в модуле.

    Открывает соединение при __init__, закрывает при close().
    Загружает расширение sqlite-vec для векторного поиска.
    Создает схему при первом открытии (CREATE TABLE IF NOT EXISTS).

    Attributes:
        db_path: Путь к SQLite-файлу.
        embedding_dim: Размерность векторов эмбеддинга.
        _conn: SQLite-соединение.
    """

    def __init__(
        self,
        db_path: Path | str,
        embedding_dim: int,
    ) -> None:
        """Инициализация хранилища памяти.

        Args:
            db_path: Путь к SQLite-файлу (":memory:" для тестов).
            embedding_dim: Размерность векторов эмбеддинга.
                Должна совпадать с размерностью, передаваемой в store/recall.

        Raises:
            ValueError: Если embedding_dim не положительное int.
            MemoryStoreError: Если не удалось загрузить sqlite-vec
                или создать схему.

        Note:
            embedding_dim валидируется ДО подключения к БД — некорректное
            значение не оставит после себя даже пустого файла БД.
            Причина: vec0 НЕ поддерживает bind-параметры (?) для типа
            колонки — размерность встраивается в DDL как
            f"float[{embedding_dim}]". Это единственное место в модуле
            без ?-placeholder'а, поэтому валидация обязательна.
        """
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be a positive int, got {embedding_dim!r}"
            )

        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self._conn: sqlite3.Connection | None = None
        self._open()

    def _open(self) -> None:
        """Открыть соединение, загрузить sqlite-vec, создать схему."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.execute(_SCHEMA_EPISODES)
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS episode_vectors "
                f"USING vec0(embedding float[{self.embedding_dim}])"
            )
        except (sqlite3.Error, OSError) as exc:
            logger.error(
                "Failed to initialize MemoryStore at %s: %s",
                self.db_path,
                exc,
            )
            raise MemoryStoreError(
                f"Failed to initialize MemoryStore at {self.db_path}: {exc}"
            ) from exc
        self._conn = conn

    def close(self) -> None:
        """Закрыть соединение (опционально, для cleanup)."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                logger.error(
                    "Failed to close MemoryStore connection %s: %s",
                    self.db_path,
                    exc,
                )
            finally:
                self._conn = None

    def store(self, episode: Episode) -> int:
        """Записать эпизод в память.

        1. Проверяет embedding.shape[0] == self.embedding_dim (fail-fast)
        2. Вычисляет content_hash(episode.content) через чистое ядро
        3. Сериализует embedding через serialize_embedding()
        4. INSERT OR IGNORE по content_hash (UNIQUE-конфликт не выбрасывает
           IntegrityError — просто игнорирует вставку)
        5. Проверяет cursor.rowcount:
           - rowcount == 1 → новая запись: id = cursor.lastrowid, шаг 6
           - rowcount == 0 → дубликат: SELECT id WHERE content_hash = ?,
             шаг 6 пропускается (вектор уже записан)
        6. Записывает вектор в vec0 virtual table (только для новых записей)

        Args:
            episode: Эпизод для записи (id игнорируется, назначается БД).

        Returns:
            id записи (новый или существующий при дубликате).

        Raises:
            ValueError: Если episode.embedding.shape[0] != self.embedding_dim.
            MemoryStoreError: При сбое I/O (после логирования через
                logging.error). Ошибка не проглатывается молча — id
                семантически важен для caller-а.

        Note:
            Дубликат — точное совпадение content (по SHA-256).
            Похожесть по эмбеддингу — НЕ дубликат (хранятся отдельно).
            Шаг 6 (vec0 insert) выполняется только для новых записей —
            при дубликате векторная таблица не затрагивается.
            Механизм детекции: INSERT OR IGNORE + cursor.rowcount
            (не exception-based — IntegrityError не возникает при OR IGNORE).
            Шаги 4 и 6 обёрнуты в единую транзакцию (with self._conn:) —
            при сбое vec0-insert откатывается и episodes-insert.
        """
        if episode.embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"embedding dim {episode.embedding.shape[0]} "
                f"!= store dim {self.embedding_dim}"
            )

        c_hash = content_hash(episode.content)
        blob = serialize_embedding(episode.embedding)

        if self._conn is None:
            raise MemoryStoreError("MemoryStore is closed")

        try:
            with self._conn:  # auto-commit/rollback транзакция
                cur = self._conn.execute(
                    _INSERT_EPISODE,
                    (
                        episode.content,
                        c_hash,
                        blob,
                        episode.timestamp,
                        episode.valence,
                        episode.stress,
                        episode.free_energy,
                    ),
                )

                if cur.rowcount == 1:
                    # Новая запись — вставляем вектор
                    ep_id = cur.lastrowid
                    if ep_id is None:
                        raise MemoryStoreError(
                            "INSERT OR IGNORE reported a new row but lastrowid is None"
                        )
                    self._conn.execute(_INSERT_VECTOR, (ep_id, blob))
                else:
                    # Дубликат — берём существующий id, vec0 не трогаем
                    row = self._conn.execute(_SELECT_ID_BY_HASH, (c_hash,)).fetchone()
                    if row is None or row[0] is None:
                        # Невозможно при INSERT OR IGNORE + UNIQUE,
                        # но защищаем от race condition в будущем
                        raise MemoryStoreError(
                            f"rowcount=0 but no existing row for hash {c_hash}"
                        )
                    ep_id = row[0]
        except MemoryStoreError:
            raise
        except sqlite3.Error as exc:
            logger.error("store() failed for content_hash=%s: %s", c_hash, exc)
            raise MemoryStoreError(f"store() failed: {exc}") from exc

        return ep_id

    def recall(
        self,
        query_embedding: Vector,
        limit: int = 5,
    ) -> list[Episode]:
        """Найти ближайшие эпизоды по семантической близости.

        1. Проверяет query_embedding.shape[0] == self.embedding_dim (fail-fast)
        2. Сериализует query_embedding
        3. Запрос к vec0 virtual table: MATCH + ORDER BY distance + LIMIT
           (внутри подзапроса — см. Implementation Note 5)
        4. JOIN подзапроса с таблицей эпизодов для метаданных
           с внешним ORDER BY v.distance (см. Implementation Note 5)
        5. Десериализует эмбеддинги через deserialize_embedding()

        Args:
            query_embedding: Вектор запроса.
            limit: Максимум результатов (default=5).

        Returns:
            Список Episode, отсортированных по убыванию схожести.
            Пустой список, если БД пуста.

        Raises:
            ValueError: Если query_embedding.shape[0] != self.embedding_dim.
            MemoryStoreError: При сбое I/O (после логирования). Аналогично
                store() — пустой список НЕ используется как сигнал ошибки,
                чтобы [] однозначно означало "эпизодов не найдено".

        Note:
            Пустая БД → [] (не ошибка, не исключение).
        """
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"query_embedding dim {query_embedding.shape[0]} "
                f"!= store dim {self.embedding_dim}"
            )

        query_blob = serialize_embedding(query_embedding)

        if self._conn is None:
            raise MemoryStoreError("MemoryStore is closed")

        try:
            rows = self._conn.execute(_RECALL_SQL, (query_blob, limit)).fetchall()
        except sqlite3.Error as exc:
            logger.error("recall() failed: %s", exc)
            raise MemoryStoreError(f"recall() failed: {exc}") from exc

        return [
            Episode(
                id=row[0],
                content=row[1],
                embedding=deserialize_embedding(row[2], self.embedding_dim),
                timestamp=row[3],
                valence=row[4],
                stress=row[5],
                free_energy=row[6],
            )
            for row in rows
        ]
