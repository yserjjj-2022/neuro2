# PLAN.md — src/memory

## Файлы для создания/изменения

1. `src/memory/models.py` — `Episode` dataclass
2. `src/memory/similarity.py` — `cosine_similarity()` (чистое ядро)
3. `src/memory/serialize.py` — `serialize_embedding()`, `deserialize_embedding()` (чистое ядро)
4. `src/memory/hash.py` — `content_hash()` (чистое ядро)
5. `src/memory/errors.py` — `MemoryStoreError`
6. `src/memory/store.py` — `MemoryStore` (imperative shell, SQLite + sqlite-vec)
7. `src/memory/protocols.py` — `SupportsStore`, `SupportsRecall`
8. `src/memory/__init__.py` — Re-exports
9. `src/tests/test_memory_similarity.py` — тесты cosine_similarity
10. `src/tests/test_memory_serialize.py` — тесты serialize/deserialize
11. `src/tests/test_memory_hash.py` — тесты content_hash
12. `src/tests/test_memory_store.py` — тесты MemoryStore

## Зависимости

**Внешние:**
- `numpy` — векторные операции
- `sqlite-vec` — виртуальная таблица vec0 (уже в pyproject.toml)

**Стандартная библиотека:**
- `dataclasses` — frozen dataclass для Episode
- `typing` — Protocol для DI
- `pathlib` — Path для db_path
- `sqlite3` — SQLite-соединение
- `hashlib` — SHA-256 для content_hash
- `logging` — для логирования ошибок I/O

## Порядок реализации

### 1. Core: similarity.py (чистое ядро, без SQLite)

- Создать `cosine_similarity(a, b) -> float`
- **Порядок валидации:**
  1. Shape mismatch → ValueError
  2. Пустые векторы (size == 0) → ValueError
  3. ‖a‖ == 0 ИЛИ ‖b‖ == 0 → 0.0 (не NaN, не inf)
  4. `dot(a, b) / (||a|| · ||b||)`
- Использовать `np.linalg.norm` и `np.dot` — чистые numpy-операции

### 2. Тесты для similarity (`src/tests/test_memory_similarity.py`)

- `test_cosine_identical`: одинаковые векторы → 1.0
- `test_cosine_orthogonal`: ортогональные → 0.0
- `test_cosine_opposite`: противоположные → -1.0
- `test_cosine_shape_mismatch`: ValueError при разных размерностях
- `test_cosine_empty`: ValueError при пустых векторах
- `test_cosine_zero_vector`: ‖a‖ == 0 ИЛИ ‖b‖ == 0 → 0.0

### 3. Core: serialize.py (чистое ядро, без SQLite)

- Создать `serialize_embedding(embedding) -> bytes`:
  `np.asarray(embedding, dtype=np.float32).tobytes()`
- Создать `deserialize_embedding(blob, dim) -> np.ndarray`:
  `np.frombuffer(blob, dtype=np.float32).reshape(dim).copy()` — `.copy()`
  обязателен: frombuffer возвращает read-only view (WRITEABLE=False),
  что в Фазе 2 сломало бы in-place операции (`embedding += delta`) с
  неочевидным ValueError. Копия на типичных размерностях почти ничего
  не стоит по производительности.
- ValueError если `len(blob) != dim * 4`

### 4. Core: hash.py (чистое ядро, без SQLite)

- Создать `content_hash(content) -> str`:
  `hashlib.sha256(content.encode("utf-8")).hexdigest()`

### 5. Тесты для serialize/hash (`src/tests/test_memory_serialize.py`, `src/tests/test_memory_hash.py`)

- `test_serialize_deserialize_roundtrip`: serialize → deserialize = исходный (assert_allclose с допуском)
- `test_deserialize_bad_length`: ValueError при len(blob) != dim * 4
- `test_content_hash_deterministic`: одинаковый content → одинаковый hash
- `test_content_hash_different`: разный content → разный hash
- `test_content_hash_unicode`: unicode content корректно хешируется

### 6. models.py + errors.py (тривиальные типы, задают контракт)

- Создать `@dataclass(frozen=True) Episode`:
  - `content: str`, `embedding: np.ndarray`, `timestamp: float`
  - `valence: float`, `stress: float`, `free_energy: float`
  - `id: int | None = None`
- Создать `class MemoryStoreError(Exception)` с докстрингом о разнице от telemetry fire-and-forget

### 7. Store: __init__ + схема (+ smoke test расширения)

- Создать `MemoryStore.__init__(db_path, embedding_dim)`:
  1. **Валидация embedding_dim ПЕРЕД построением f-string со схемой**:
     ```python
     if not isinstance(embedding_dim, int) or embedding_dim <= 0:
         raise ValueError(f"embedding_dim must be a positive int, got {embedding_dim!r}")
     ```
     Причина: vec0 НЕ поддерживает bind-параметры (`?`) для типа колонки —
     размерность обязана быть встроена в DDL как `f"float[{embedding_dim}]"`.
     Это единственное место в модуле без `?`-placeholder'а. Без проверки
     float/отрицательный dim либо сломает DDL невнятной
     `sqlite3.OperationalError`, либо откроет инъекцию в SQL, если значение
     придёт из внешнего конфига.
  2. `sqlite3.connect(db_path)`
  3. `conn.enable_load_extension(True)` → `conn.load_extension("vec0")`
  4. CREATE TABLE IF NOT EXISTS `episodes` (id, content, content_hash UNIQUE, embedding BLOB, timestamp, valence, stress, free_energy)
  5. CREATE VIRTUAL TABLE IF NOT EXISTS `episode_vectors` USING vec0(embedding float[{embedding_dim}])
- **Smoke test** (`test_memory_store.py`):
  - `test_schema_creation`: :memory: → таблицы episodes и episode_vectors существуют
  - `test_sqlite_vec_loaded`: расширение загружено (smoke на :memory:)
  - `test_init_rejects_invalid_embedding_dim`: 0, отрицательное, не-int → ValueError ДО подключения к БД

### 8. Store: store() — с транзакцией и rowcount сразу (НЕ двумя заходами)

- Создать `store(episode) -> int`:
  1. Dimension validation: `episode.embedding.shape[0] != self.embedding_dim` → ValueError
  2. `content_hash(episode.content)` через чистое ядро
  3. `serialize_embedding(episode.embedding)` через чистое ядро
  4. `with self._conn:` (транзакция):
     - `INSERT OR IGNORE INTO episodes (...)`
     - `cursor.rowcount == 1` → `id = cursor.lastrowid`, INSERT в episode_vectors
     - `cursor.rowcount == 0` → дубликат, `SELECT id WHERE content_hash = ?`
  5. try/except `sqlite3.Error` → `logging.error()` + `MemoryStoreError`
  6. Вернуть id

### 9. Тесты для store() — ВКЛЮЧАЯ атомарность СРАЗУ (не в конце)

- `test_store_inserts_episode`: store() → id, эпизод в БД
- `test_store_duplicate_returns_same_id`: повторный store() → тот же id
- `test_store_dim_mismatch`: ValueError при embedding.shape[0] != embedding_dim
- `test_store_memory_store_error`: monkeypatch execute → sqlite3.OperationalError → MemoryStoreError
- **`test_store_atomicity_rollback`**: мок vec0-insert (второй execute) → MemoryStoreError → SELECT по content_hash пусто

### 10. Store: recall()

- Создать `recall(query_embedding, limit=5) -> list[Episode]`:
  1. Dimension validation → ValueError
  2. `serialize_embedding(query_embedding)`
  3. MATCH в vec0 внутри **подзапроса** (прямой JOIN с vec0 зависает — баг
     sqlite-vec 0.1.9, проверено на :memory: и файловой БД):
     ```
     SELECT e.*, v.distance FROM episodes e
       JOIN (SELECT rowid, distance FROM episode_vectors
             WHERE embedding MATCH ? ORDER BY distance LIMIT ?) v
         ON e.id = v.rowid
       ORDER BY v.distance
     ```
     Внешний ORDER BY v.distance обязателен: порядок строк после JOIN
     не гарантирован стандартом SQL, даже если подзапрос отсортирован.
  4. Десериализует эмбеддинги через `deserialize_embedding()`
  5. try/except `sqlite3.Error` → `logging.error()` + `MemoryStoreError`
  6. Пустая БД → []

### 11. Тесты для recall() (`src/tests/test_memory_store.py`)

- `test_recall_empty_db`: пустая БД → []
- `test_recall_top_k`: store несколько, recall → top-k по близости
- `test_recall_dim_mismatch`: ValueError при query_embedding.shape[0] != embedding_dim
- `test_recall_memory_store_error`: monkeypatch → MemoryStoreError (НЕ [] как сигнал ошибки)

### 12. Persistence-тест (интеграционный, tmp_path)

- `test_persistence_across_reopen` (в `test_memory_store.py`):
  store → close → reopen (тот же db_path) → recall находит
- `test_close_twice_no_error`: close() дважды без ошибки

### 13. protocols.py + __init__.py

- Создать `SupportsStore` (store -> int), `SupportsRecall` (recall -> list[Episode])
- Re-export в `__init__.py`: `Episode`, `cosine_similarity`, `serialize_embedding`, `deserialize_embedding`, `content_hash`, `MemoryStoreError`, `MemoryStore`, `SupportsStore`, `SupportsRecall`

### 14. Интеграция с src/host/wiring.py (опционально, отдельным PR)

- По аналогии с energy+telemetry: wiring, подключающий memory к host loop
- **НЕ входит в этот PR** — отдельный priority-пункт после одобрения store/recall

## План тестов

| Тест | Покрытие | Инвариант |
|------|----------|-----------|
| `test_cosine_identical` | Косинусная схожесть | cos(a,a) = 1.0 |
| `test_cosine_orthogonal` | Косинусная схожесть | cos(a,b) = 0.0 |
| `test_cosine_opposite` | Косинусная схожесть | cos(a,-a) = -1.0 |
| `test_cosine_shape_mismatch` | Валидация | ValueError |
| `test_cosine_empty` | Edge case | ValueError |
| `test_cosine_zero_vector` | Edge case | 0.0 при ‖a‖==0 ИЛИ ‖b‖==0 |
| `test_serialize_deserialize_roundtrip` | Сериализация | assert_allclose |
| `test_deserialize_bad_length` | Валидация | ValueError |
| `test_content_hash_deterministic` | Хеш | Детерминированность |
| `test_content_hash_different` | Хеш | Разный контент → разный hash |
| `test_content_hash_unicode` | Хеш | Unicode |
| `test_schema_creation` | Схема | Таблицы созданы |
| `test_sqlite_vec_loaded` | Расширение | vec0 загружено |
| `test_init_rejects_invalid_embedding_dim` | Валидация __init__ | ValueError при 0/отрицательном/не-int |
| `test_store_inserts_episode` | store() | id возвращён |
| `test_store_duplicate_returns_same_id` | Дедупликация | Тот же id |
| `test_store_dim_mismatch` | Валидация | ValueError |
| `test_store_memory_store_error` | Crash-safety | MemoryStoreError |
| `test_store_atomicity_rollback` | Атомарность | Откат транзакции |
| `test_recall_empty_db` | Edge case | [] |
| `test_recall_top_k` | Поиск | Top-k по близости |
| `test_recall_dim_mismatch` | Валидация | ValueError |
| `test_recall_memory_store_error` | Crash-safety | MemoryStoreError, не [] |
| `test_persistence_across_reopen` | Персистентность | tmp_path reopen |
| `test_close_twice_no_error` | Cleanup | close() идемпотентен |

## Заметки для реализации

- **Core-first**: шаги 1-5 не трогают SQLite — можно закоммитить отдельным PR/шагом
- **Atomicity**: тест rollback пишется СРАЗУ после store(), не в общем шаге тестов в конце
- **Smoke test sqlite-vec**: загрузка бинарного расширения — самая вероятная точка платформенных сюрпризов (macOS/Linux/CI). Проверить отдельно от логики store/recall
- **DDL без placeholder'а**: `embedding_dim` интерполируется в DDL как `f"float[{embedding_dim}]"` — vec0 не поддерживает `?` для типа колонки. Это единственное место без bind-параметра, поэтому валидация `isinstance(int) and > 0` обязательна ДО построения f-string
- **deserialize `.copy()`**: `np.frombuffer(...).reshape(dim).copy()` — иначе read-only view ломает in-place операции в Фазе 2
- **Прямой JOIN с vec0 зависает**: баг sqlite-vec 0.1.9. Только подзапрос: `JOIN (SELECT rowid, distance FROM episode_vectors WHERE embedding MATCH ? ORDER BY distance LIMIT ?) v ON e.id = v.rowid`. Проверено на :memory: и файловой БД
- **Транзакция**: `with self._conn:` — auto-commit/rollback. НЕ два отдельных execute без транзакции
- **Rowcount**: `cursor.rowcount == 1` → новая запись (lastrowid + vec0), `== 0` → дубликат (SELECT id). НЕ exception-based
- **Дубликат**: точное совпадение content (SHA-256). Похожесть по эмбеддингу — не дубликат
- **Dimension validation**: до любых SQL-операций (fail-fast), аналог energy
- **Порядок except**: специфичные sqlite3 исключения раньше общего `sqlite3.Error` (для будущих правок)
- **BLOB**: float32 little-endian, contiguous (`np.float32 .tobytes()`)
- **Повреждённый файл БД**: sqlite3.DatabaseError → MemoryStoreError (fail-fast), recovery — Фаза 2
- **Reference**: `FreeEnergyResult` (frozen dataclass), `TelemetryWriter` (shell owning resource), `SupportsWrite` (Protocol for DI), `wiring.py` (интеграция energy+telemetry)