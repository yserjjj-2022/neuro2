# SPEC.md — src/memory

## Назначение

Эпизодическая память хоста: хранение прецедентов (правок, ситуаций) с эмбеддингами в SQLite + sqlite-vec, поиск по семантической близости за доли миллисекунды. Модуль хранит и извлекает — не принимает решений.

S2 (`stages/S2_SPEC.md`) добавляет: эмбеддер (текст → вектор), детектор
значимых событий, приор памяти и `MemoryRouter` (recall → приор, запись).

S6 (`stages/S6_SPEC.md`) добавляет **консолидацию** (`consolidation.py`):
active pruning незначимых эпизодов + Structure Learning (схемы). Удаление
явное и логируемое — инвариант 6 («recall монотонен **кроме явной
консолидации**»). `MemoryStore.delete` / `all_episodes` / `save_schema`.

См. также:
- `FreeEnergyResult` из `src/core/energy/` — пример frozen dataclass для доменного объекта
- `TelemetryEvent` из `src/telemetry/` — пример frozen dataclass с плоской структурой
- `EnergyObserver` / `TelemetryLogger` — примеры Shell с DI через sink/Protocol

## S2: эмбеддер, события, приор, роутер

### Embedder (Protocol) + реализации

```python
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    def embed(self, text: str) -> Vector: ...


class EmbedderError(Exception): ...


@dataclass(frozen=True)
class FakeEmbedder:  # детерминированный bag-of-tokens hashing
    dim: int = 8
    seed: int = 0

    def embed(self, text: str) -> Vector: ...


@dataclass
class ApiEmbedder:  # OpenAI-совместимый (RouterAI), ленивый клиент
    model: str = "voyageai/voyage-4-lite"
    dim: int = 256
    base_url: str = "https://routerai.ru/api/v1"
    api_key: str | None = None
    normalize: bool = True


def build_embedder(
    mode="auto", dim=8, model=..., base_url=..., api_dim=256, api_key=None
) -> Embedder:
    """auto: ключ→api, иначе fake; fake; api (без ключа → ValueError)."""


def embedder_settings_from_env() -> dict[str, object]:
    """base_url/model/api_dim из EMBEDDER_* (с дефолтами)."""
```

Настройки API — в `.env` (`EMBEDDER_API_KEY`, `_BASE_URL`, `_MODEL`, `_DIM`),
CLI грузит `load_dotenv()`. Векторы L2-нормируются; `ApiEmbedder` кэширует
текст → вектор.

### Значимые события (Core)

```python
def is_significant_event(f, prev_f, reflex_tags, spike_threshold,
                         has_new_message=False) -> bool:
    """reflex ИЛИ (f - prev_f) > spike_threshold ИЛИ новое сообщение (S3)."""


def build_event_content(active_tags, reflex_tags, valence, stress) -> str:
    """Дескриптор ситуации для эпизода без текста."""
```

### Приор памяти (Core)

```python
MEMORY_PRIOR_DIM = 4


def encode_memory_prior(episode, query) -> Vector:
    """[cos(query, ep.embedding), tanh(valence), tanh(stress), tanh(f)].

    None episode/query → нули. Компоненты ∈ [-1, 1].
    """
```

### MemoryRouter (Shell)

```python
class MemoryRouter:
    def __init__(
        self, store, embedder, spike_threshold, recall_limit=1, prior_dim=4
    ) -> None: ...
    def context_embedding(self, text) -> Vector | None: ...  # кэш
    def recall_prior(self, query) -> Vector: ...  # top-1 → приор
    def maybe_store(
        self, *, text, query, f, prev_f, valence, stress, active_tags,
        reflex_tags, now, has_new_message=False
    ) -> int | None: ...
```

**Crash-safety:** сбой store/recall/embed → лог + безопасный дефолт (нули /
None); память не роняет тик. Это осознанное отличие от `MemoryStore`
(fail-fast для прямого caller'а).

## Публичный интерфейс

### Episode (dataclass)

```python
@dataclass(frozen=True)
class Episode:
    """Эпизод памяти — прецедент правки или ситуации.

    Аналог FreeEnergyResult (energy) и TelemetryEvent (telemetry) —
    frozen dataclass как доменный объект модуля.

    id=None для новых эпизодов (до записи в БД).
    После recall — id заполнен из БД.
    """

    content: str  # Текстовое содержание эпизода (правка, ситуация)
    embedding: np.ndarray  # Вектор эмбеддинга content
    timestamp: float  # Unix timestamp (time.time()) момента эпизода
    valence: float  # Валентность в момент эпизода (из energy)
    stress: float  # Аллостатический стресс в момент эпизода
    free_energy: float  # F(t) в момент эпизода
    id: int | None = None  # Назначается БД при insert; None для новых
```

### Functional Core — чистые функции (без I/O, без SQLite)

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусная схожесть двух векторов.

    Чистая функция: тестируется без SQLite, без файловой системы.
    Используется для тестов и потенциального re-ranking результатов recall.

    Args:
        a: Первый вектор.
        b: Второй вектор.

    Returns:
        Косинусная схожесть ∈ [-1.0, 1.0].

    Raises:
        ValueError: Если a.shape != b.shape.
        ValueError: Если векторы пустые (size == 0).

    Formula:
        cos(a, b) = dot(a, b) / (||a|| · ||b||)

    Edge cases:
        - ‖a‖ == 0 ИЛИ ‖b‖ == 0 → return 0.0 (деление на ноль в знаменателе)
        - Пустые векторы → ValueError
    """
    ...


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Сериализация эмбеддинга в BLOB для хранения в SQLite.

    Чистая функция: np.ndarray → bytes, без I/O.

    Args:
        embedding: Вектор эмбеддинга.

    Returns:
        BLOB (bytes) для записи в SQLite.

    Note:
        Формат: float32 little-endian, contiguous.
        np.asarray(embedding, dtype=np.float32).tobytes()
    """
    ...


def deserialize_embedding(blob: bytes, dim: int) -> np.ndarray:
    """Десериализация BLOB из SQLite в np.ndarray.

    Чистая функция: bytes → np.ndarray, без I/O.

    Args:
        blob: BLOB из SQLite.
        dim: Ожидаемая размерность вектора.

    Returns:
        Вектор эмбеддинга (np.float32).

    Raises:
        ValueError: Если len(blob) != dim * 4 (размер float32).
    """
    ...


def content_hash(content: str) -> str:
    """SHA-256 хеш контента для дедупликации.

    Чистая функция: str → str, без I/O.

    Args:
        content: Текст содержания эпизода.

    Returns:
        Hex-строка SHA-256 (64 символа).
    """
    ...
```

### MemoryStoreError (кастомное исключение)

```python
class MemoryStoreError(Exception):
    """Собственное исключение — оборачивает sqlite3 ошибки.

    В отличие от TelemetryLogger.log() (fire-and-forget, ничего не возвращает),
    store() возвращает id — значимое значение для вызывающего кода.
    Молчаливое проглатывание ошибки с None/-1 создаёт риск, что caller
    продолжит работу с несуществующей записью.

    Поэтому: ошибка логируется через logging.error(), затем пробрасывается
    как MemoryStoreError. Вызывающий код сам решает, ловить или нет.
    """

    ...
```

### Imperative Shell — владеет SQLite-соединением

```python
class MemoryStore:
    """Imperative Shell: единственное место SQLite I/O в модуле.

    Соответствует паттерну telemetry:
    - Ядро (cosine_similarity, serialize_embedding, ...) — чистые функции
    - Shell (MemoryStore) — владеет соединением, выполняет запросы
    - DI через Protocol (SupportsStore / SupportsRecall) — можно мокать

    Открывает соединение при __init__, закрывает при close().
    Загружает расширение sqlite-vec для векторного поиска.
    Создает схему при первом открытии (CREATE TABLE IF NOT EXISTS).
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

        Note:
            Создаёт таблицы и виртуальную таблицу vec0 при первом открытии.
            Загружает расширение sqlite-vec.
        """
        ...

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
        """
        ...

    def recall(
        self,
        query_embedding: np.ndarray,
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
        ...

    def close(self) -> None:
        """Закрыть соединение (опционально, для cleanup)."""
        ...
```

### Protocols для DI

```python
class SupportsStore(Protocol):
    """Protocol для записи в память.

    Позволяет energy/tm/wiring модулям зависеть от абстракции,
    а не от конкретного MemoryStore. Аналог SupportsWrite из telemetry.

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
        self, query_embedding: np.ndarray, limit: int = ...
    ) -> list[Episode]: ...


class SupportsConsolidate(Protocol):  # S6
    def all_episodes(self) -> list[Episode]: ...
    def delete(self, ids: list[int]) -> int: ...
    def save_schema(
        self, centroid: np.ndarray, member_count: int, summary: str
    ) -> int: ...
```

## Консолидация (S6)

```python
@dataclass(frozen=True)
class Schema:
    centroid: Vector
    member_count: int
    summary: str

@dataclass(frozen=True)
class ConsolidationPlan:
    prune_ids: tuple[int, ...]
    schemas: tuple[Schema, ...]
    kept: int
    reason: str

@dataclass(frozen=True)
class ConsolidationResult:
    pruned: int
    schemas_saved: int
    reason: str

def episode_weight(episode, *, now, recency_tau_s) -> float: ...
def plan_consolidation(episodes, *, min_weight, schema_threshold,
                       max_schemas, now, recency_tau_s=86400.0) -> ConsolidationPlan: ...
def consolidate(store, *, min_weight, schema_threshold, max_schemas,
                now, recency_tau_s=86400.0) -> ConsolidationResult: ...
```

`plan_consolidation` (Core, чистая): pruning по весу (`episode_weight` =
аффект × свежесть), Structure Learning — жадная кластеризация сохранённых по
косинусу. `consolidate` (Shell): исполнить план, логировать число удалённых.
Таблица `schemas` (centroid, member_count, summary, created_at).

## Инварианты

1. **cosine_similarity ∈ [-1, 1]**: результат всегда в диапазоне.
2. **Нулевой вектор → 0.0**: если ‖a‖ == 0 ИЛИ ‖b‖ == 0 → 0.0 (не NaN, не inf). Формула dot/(‖a‖·‖b‖) ломается при нулевом множителе в знаменателе.
3. **Пустая БД → recall() = []**: первый запуск без эпизодов не ошибка.
4. **Дедупликация по content_hash**: одинаковый content → одна запись (возвращает существующий id).
5. **embedding_dim фиксирована**: все векторы в хранилище имеют одну размерность (задаётся при __init__).
6. **Схема создаётся при открытии**: CREATE TABLE IF NOT EXISTS — первый запуск создаёт схему автоматически.
7. **Functional Core**: cosine_similarity, serialize_embedding, deserialize_embedding, content_hash — чистые функции, тестируются без SQLite.
8. **DI через Protocol**: MemoryStore реализует SupportsStore + SupportsRecall; другие модули зависят от Protocol.
9. **Ошибки I/O → MemoryStoreError**: store() и recall() логируют через logging.error() и пробрасывают MemoryStoreError. Ошибка не проглатывается молча — в отличие от telemetry (fire-and-forget), id для store() и пустой список для recall() семантически значимы. Вызывающий код сам решает, ловить исключение или нет, но не может неявно продолжить с несуществующей записью.
10. **BLOB-формат**: float32 little-endian, contiguous (np.float32 .tobytes()).
11. **Dimension validation (fail-fast)**: `episode.embedding.shape[0] == self.embedding_dim` в store() и `query_embedding.shape[0] == self.embedding_dim` в recall(). Иначе ValueError — до любых SQL-операций.
12. **Атомарность записи**: store() либо создаёт episode+vector вместе, либо не создаёт ничего — частичная запись невозможна. Шаги 4 и 6 обёрнуты в единую транзакцию; при сбое vec0-insert откатывается и episodes-insert.

## Критерии приёмки

- [ ] `Episode` — frozen dataclass, все поля типизированы
- [ ] `cosine_similarity()` — чистая функция, тестируется без SQLite
- [ ] `cosine_similarity()` raises ValueError при shape mismatch
- [ ] `cosine_similarity()` возвращает 0.0 если ‖a‖ == 0 ИЛИ ‖b‖ == 0 (не NaN)
- [ ] `serialize_embedding()` / `deserialize_embedding()` — round-trip: serialize → deserialize = исходный вектор (np.testing.assert_allclose с допуском; сериализация даункастит float64 → float32, строгое равенство некорректно для float64-входа)
- [ ] `content_hash()` — детерминированный, одинаковый content → одинаковый hash
- [ ] `MemoryStore` создаёт схему при первом открытии (CREATE TABLE IF NOT EXISTS)
- [ ] `MemoryStoreError` — кастомное исключение, оборачивает sqlite3 ошибки
- [ ] `MemoryStore.store()` — вставляет эпизод, возвращает id
- [ ] `MemoryStore.store()` — дубликат по content → возвращает существующий id (не создаёт новый, vec0 не затрагивается)
- [ ] `MemoryStore.store()` — raises ValueError при embedding.shape[0] != embedding_dim
- [ ] `MemoryStore.store()` — raises MemoryStoreError при сбое I/O (после логирования)
- [ ] `MemoryStore.store()` — атомарность: ошибка на vec0-insert → episodes НЕ содержит orphan-строку (транзакция откатилась целиком)
- [ ] `MemoryStore.recall()` — пустая БД → []
- [ ] `MemoryStore.recall()` — возвращает top-k по схожести
- [ ] `MemoryStore.recall()` — raises ValueError при query_embedding.shape[0] != embedding_dim
- [ ] `MemoryStore.recall()` — raises MemoryStoreError при сбое I/O (не возвращает [] как сигнал ошибки)
- [ ] `MemoryStore.close()` — закрывает соединение без ошибки при повторном вызове
- [ ] `SupportsStore` / `SupportsRecall` — Protocol для DI
- [ ] Минимум 13 тестов: 3 для core (cosine, serialize/deserialize, content_hash), 10 для shell (store, duplicate, dim mismatch store, store raises MemoryStoreError, atomicity rollback, recall empty, recall top-k, dim mismatch recall, recall raises MemoryStoreError, close)
- [ ] `ruff check` и `ruff format` проходят без ошибок
- [ ] mypy strict для `src/memory/` не ругается

## Стратегия тестирования

Следует CONSTITUTION.md §3.2 и прецеденту из energy/telemetry:

| Уровень | Инструмент | Что тестирует |
|---------|-----------|---------------|
| **Core (pure)** | Без SQLite вообще | `cosine_similarity`, `serialize_embedding`, `deserialize_embedding`, `content_hash` |
| **Shell (unit)** | `:memory:` | `store()`, `recall()` на пустой БД, дедупликация — быстро, без файловой системы |
| **Shell (integration)** | `tmp_path` | Персистентность: записать → закрыть → reopened → recall находит. Миграции схемы. |

**Баланс**: `tmp_path` — основной вариант (весь смысл модуля — персистентность между сессиями). `:memory:` — только для мелких тестов без проверки персистентности. Core-функции — вообще без SQLite.

**Симуляция сбоя I/O для тестов MemoryStoreError**: через мок/monkeypatch на уровне соединения (например, monkeypatch `cursor.execute` → бросает `sqlite3.OperationalError`), НЕ через реальное повреждение файла БД — это быстрее и не зависит от платформы. Оба теста (`store` и `recall`) пишутся на `:memory:`.

**Тест атомарности (rollback)**: мокаем vec0-insert (второй `cursor.execute` в `store()`) так, что он бросает `sqlite3.OperationalError`; после `MemoryStoreError` проверяем, что в `episodes` нет orphan-строки (SELECT по content_hash вернул пусто). На `:memory:`.

## Явно НЕ входит в скоуп (Phase 1)

- ~~**Ночной сон**~~ → S6: консолидация вызывается явно (`consolidate`).
- ~~**Structure Learning**~~ → S6: схемы (`plan_consolidation`).
- **EvolvingSteeringMemory**: нет накопления вектора характера — Фаза 2+
- ~~**Удаление эпизодов**~~ → S6: `delete()` (только явная консолидация).
- **Расписание ночного сна** — S6 проход 2 (в проходе 1 — явный вызов).
- **Ротация/сжатие БД**: нет VACUUM, нет архивации — Фаза 4+
- **Конкурентный доступ**: явно исключён — однопользовательский однопроцессный хост
- **Многомерный поиск**: нет фильтрации по времени/валентности — только по схожести
- **Re-ranking**: нет повторной сортировки результатов через cosine_similarity — только vec0 distance

## Open Questions

| Вопрос | Статус | Решение |
|--------|--------|---------|
| **sqlite-vec vs brute-force Python** | Решено | sqlite-vec с первого дня. Зависимость уже в `pyproject.toml`. Виртуальная таблица `vec0` для индексированного поиска. `cosine_similarity()` в ядре — для тестов и потенциального re-ranking в Фазе 2. |
| **Критерий дубликата** | Решено | Точное совпадение content по SHA-256 (`content_hash`). Похожесть по эмбеддингу — НЕ дубликат. |
| **embedding_dim** | Решено | Параметр `MemoryStore.__init__`. Фиксированная размерность на время жизни хранилища. Смена модели эмбеддингов → новая БД. |
| **metadata (произвольные поля)** | Отложен | Phase 1: структурированные поля (valence, stress, free_energy). Phase 2: JSON-колонка `metadata` для произвольных атрибутов. |
| **Размер BLOB и лимит limit** | Отложен | Phase 1: limit по умолчанию = 5. Phase 2: параметризуемый из config. |
| **Повреждённый файл БД** | Отложен | Phase 1: sqlite3.DatabaseError пробрасывается как MemoryStoreError (fail-fast при старте). Phase 2: recovery mode (бэкап + новая БД). |
| **Многопоточный доступ в Фазе 4** | Тех.долг | Phase 1: однопроцессный хост, INSERT OR IGNORE → SELECT безопасен. Phase 4: фоновая консолидация/сон (отдельный поток на тот же SQLite-файл) потребует пересмотра дедупликации — двухшаговый INSERT OR IGNORE → SELECT перестанет быть атомарным. Решение: WAL-mode + explicit transaction или UPSERT (INSERT ... ON CONFLICT). |

## Implementation Notes

1. **sqlite-vec loading**: `conn.enable_load_extension(True)` → `conn.load_extension("vec0")` в `__init__`.
2. **Schema**: две таблицы — `episodes` (метаданные + BLOB) и `episode_vectors` (vec0 virtual table). Связь по `id`.
3. **Content hash**: `hashlib.sha256(content.encode("utf-8")).hexdigest()`, колонка `UNIQUE`.
4. **Duplicate handling**: `INSERT OR IGNORE` по `content_hash`, затем проверка `cursor.rowcount`: `1` → новая запись (`lastrowid` + vec0 insert), `0` → дубликат (`SELECT id WHERE content_hash = ?`, vec0 не трогаем). Исключение-based вариант НЕ используется — `INSERT OR IGNORE` не выбрасывает `IntegrityError`. Без race condition (однопроцессный хост).
   **Атомарность**: оба INSERT (episodes + episode_vectors) выполняются в одной транзакции (`with conn:` — auto-commit/rollback в sqlite3). Если vec0-insert падает, откатывается и episodes-insert — иначе возможна orphan-запись: content_hash уже занят, но вектор так и не будет записан, и дедупликация больше никогда не даст повторить попытку (тихая потеря данных — эпизод невидим для recall()).
5. **vec0 query**: `SELECT e.*, v.distance FROM episodes e JOIN (SELECT rowid, distance FROM episode_vectors WHERE embedding MATCH ? ORDER BY distance LIMIT ?) v ON e.id = v.rowid ORDER BY v.distance`. **Важно 1**: подзапрос обязателен — прямой JOIN `episodes e JOIN episode_vectors v ON ... WHERE v.embedding MATCH ?` зависает (баг sqlite-vec 0.1.9). Подзапрос работает и на `:memory:`, и на файловой БД. **Важно 2**: внешний `ORDER BY v.distance` обязателен — порядок строк после JOIN не гарантирован стандартом SQL, даже если подзапрос отсортирован. Планировщик SQLite вправе переупорядочить строки при join, а поведение планировщика на стыке vec0 + JOIN заведомо нестандартное.
6. **Reference patterns**: `FreeEnergyResult` (frozen dataclass), `TelemetryWriter` (shell owning resource), `SupportsWrite` (Protocol for DI).
7. **Crash-safety**: `store()` и `recall()` оборачивают sqlite3 операции в try/except, логируют ошибку через `logging.error()`, затем пробрасывают `MemoryStoreError`. Отличие от `TelemetryLogger.log()` (fire-and-forget): здесь возвращаемое значение (id) и пустой список семантически значимы — caller не должен неявно продолжить с несуществующей записью или спутать "ошибка" с "не найдено". Порядок except неактуален для `store()` (дубликат через rowcount, не через IntegrityError), но для будущих правок: специфичные исключения sqlite3 всегда раньше общего `sqlite3.Error`.
8. **Dimension validation**: `episode.embedding.shape[0] != self.embedding_dim` → ValueError до любых SQL-операций. Аналог shape validation в `FreeEnergyCalculator.compute()`.
