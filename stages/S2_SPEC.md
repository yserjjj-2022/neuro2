# SPEC.md — S2: Непрерывность

Стадия S2 из `BUILD_ROADMAP.md`. Цель — **замкнуть контур во времени**:
память становится органом чувств, и поведение хоста зависит от прошлых
эпизодов (в том числе после перезапуска).

Предпосылка: `MemoryStore` (store/recall) уже реализован (`src/memory/`).
Блокер `[Phase1][memory-wiring]` — отсутствие источника `content`/`embedding`.
S2 его снимает: появляется **эмбеддер** и **реальный текст** коммуникативного
входа. Ворота S2 — в `VALIDATION.md` §4. Решения — ADR-0003 (API-first),
ADR-0004 (FC/IS), ADR-0006 (непрерывность).

## Область (входит)

1. **Эмбеддер** — текст → вектор; Protocol + детерминированный fake + API.
2. **Коммуникативный вход как текст** — `TextMessageProvider` эмбеддит реальный
   текст в шину (разблокирует и memory-wiring, и коммуникативный маршрут).
3. **Значимые события** — детектор (всплеск F / reflex) + построение `content`.
4. **Recall → приор** — похожий прошлый эпизод подаётся в шину как сегмент
   «память» (приор), влияя на F и, значит, на поведение.
5. **Запись эпизодов** в loop на значимых событиях (с дебаунсом).
6. **Персистентность** — БД из конфига; после перезапуска recall находит.

## Явно НЕ входит

- **Речь / Intent-Frame / LLM-ответ** — S3.
- **Консолидация во сне, Structure Learning** — S6.
- **EvolvingSteeringMemory** (вектор характера) — S5+.
- **Проекции колонок** (`reads`) — S4+; колонки по-прежнему читают всю шину.
- **Policy / действие** — S4.
- **Recall → точность γ** (второй маршрут влияния) — отложено; в S2 приор в шину.
- **Реальное локальное эмбеддер-модель (ONNX)** — позже; интерфейс готов.

## 1. Эмбеддер

Эмбеддер — Shell (сетевой/вычислительный I/O), инъекция через Protocol.

```python
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    def embed(self, text: str) -> Vector: ...

class EmbedderError(Exception):
    """Оборачивает сбои эмбеддера (сеть, API, размерность)."""

@dataclass(frozen=True)
class FakeEmbedder:
    """Детерминированный эмбеддер для тестов и replay.

    Bag-of-tokens hashing: токены → buckets через sha256, L2-нормировка.
    Похожие тексты (общие токены) → выше косинус. Пустой текст → нули.
    """
    dim: int = 8
    seed: int = 0
    def embed(self, text: str) -> Vector: ...

class ApiEmbedder:
    """OpenAI-совместимый эмбеддер (RouterAI по умолчанию), ленивый клиент.

    Клиент создаётся при первом вызове (ключ из env/аргумента). Вектор
    L2-нормируется (масштаб не зависит от модели/размерности). Кэширует
    текст → вектор (активное сообщение не меняется между тиками).
    Сбой сети → EmbedderError. Тестами НЕ покрывается (сеть) — только
    контракт и фабрика.
    """
    def __init__(self, model: str = "voyageai/voyage-4-lite", dim: int = 256,
                 base_url: str = "https://routerai.ru/api/v1",
                 api_key: str | None = None, normalize: bool = True) -> None: ...
    @property
    def dim(self) -> int: ...
    def embed(self, text: str) -> Vector: ...
```

### Конфигурация через окружение (.env)

API-настройки НЕ в коде, а в `.env` (в `.gitignore`; шаблон — `.env.example`):

| Переменная | Дефолт | Смысл |
|---|---|---|
| `EMBEDDER_API_KEY` | — | Ключ RouterAI (обязателен для api/auto→api) |
| `EMBEDDER_BASE_URL` | `https://routerai.ru/api/v1` | OpenAI-совместимый base URL |
| `EMBEDDER_MODEL` | `voyageai/voyage-4-lite` | Модель эмбеддингов |
| `EMBEDDER_DIM` | `256` | Размерность (Matryoshka-усечение) |

CLI вызывает `load_dotenv()`. `embedder_settings_from_env()` читает настройки;
`build_embedder` принимает их явно (DI). RouterAI — OpenAI-совместимый шлюз:
`client.embeddings.create(model, input, dimensions, encoding_format="float")`.

### Выбор эмбеддера: режим `auto`

`embedder_mode` — три режима:

- **`auto`** (дефолт): если ключ доступен в окружении (`EMBEDDER_API_KEY`) →
  `ApiEmbedder`; иначе → `FakeEmbedder`. Тесты (ключ очищен `conftest.py`)
  детерминированы; боевой запуск (с ключом) использует реальный эмбеддер.
- **`fake`**: всегда `FakeEmbedder` (тесты, replay, офлайн).
- **`api`**: всегда `ApiEmbedder` (явный боевой режим; ошибка, если нет ключа).

Фабрика: `build_embedder(mode, dim, model, base_url, api_dim, api_key)`.
`auto` логирует выбранный режим при старте (наблюдаемость).

**Нормировка:** и `FakeEmbedder`, и `ApiEmbedder` возвращают L2-нормированный
вектор → масштаб F не зависит от модели/размерности, пороги переносимы,
косинус = скалярное произведение.

**Инварианты:**
- `embed("")` → вектор нулей (не ошибка).
- `FakeEmbedder`: одинаковый текст → одинаковый вектор (детерминизм).
- Возврат: 1-D `float64`, `shape == (dim,)`, конечные значения.
- Возврат L2-нормирован (норма 1.0 для непустого текста) — оба эмбеддера.
- `auto` без ключа → `fake`; `api` без ключа → `ValueError` (fail-fast).

## 2. Коммуникативный вход как текст

`TextMessageProvider` заменяет вектор-заглушку реальным текстом:

```python
@dataclass(frozen=True)
class TextMessageProvider:
    """Сообщение собеседника: текст → эмбеддинг в шину.

    Детерминирован при детерминированном эмбеддере. ``messages`` — скрипт
    (tick, text) для mock/тестов; в S3 заменится реальным вводом без смены
    интерфейса.
    """
    embedder: Embedder
    messages: tuple[tuple[int, str], ...] = ()
    tag: str = "user_message"
    category: SignalCategory = SignalCategory.COMMUNICATIVE
    period: int = 1

    @property
    def dim(self) -> int: ...          # == embedder.dim
    def text_at(self, tick: int) -> str:
        """Текст последнего сообщения с tick'ом ≤ текущего ("" если нет)."""
    def read(self, tick: int, now: float) -> SignalSource: ...  # data = embed(text)
```

`text_at` нужен loop'у для `content` эпизода и для контекста recall.

## 3. Значимые события и content (чистое ядро)

```python
def is_significant_event(
    f: float, prev_f: float, reflex_tags: tuple[str, ...],
    spike_threshold: float,
) -> bool:
    """Событие значимо, если есть reflex ИЛИ (f − prev_f) > spike_threshold."""

def build_event_content(
    active_tags: tuple[str, ...], reflex_tags: tuple[str, ...],
    valence: float, stress: float,
) -> str:
    """Текстовый дескриптор ситуации (когда нет текста собеседника)."""
```

`content` эпизода = `text_at(tick)`, если непустой, иначе `build_event_content`.

## 4. Приор памяти и роутер

### encode_memory_prior (чистое ядро)

```python
MEMORY_PRIOR_DIM = 4

def encode_memory_prior(
    episode: Episode | None, query: Vector | None,
) -> Vector:
    """[cos(query, ep.embedding), tanh(valence), tanh(stress), tanh(f)].

    Нет эпизода / нет запроса → вектор нулей shape=(MEMORY_PRIOR_DIM,).
    Косинус ∈ [-1, 1]; tanh держит аффект ограниченным (защита от blow-up).
    """
```

### MemoryRouter (Shell)

```python
class MemoryRouter:
    """Оркестрация памяти: recall → приор, запись на значимых событиях.

    Владеет ссылками на store (SupportsStore & SupportsRecall) и embedder.
    Кэширует последний (text → embedding), чтобы не звать API каждый тик.
    """
    def __init__(
        self, store, embedder: Embedder, spike_threshold: float,
        recall_limit: int = 1, prior_dim: int = MEMORY_PRIOR_DIM,
    ) -> None: ...

    @property
    def prior_dim(self) -> int: ...

    def context_embedding(self, text: str) -> Vector | None:
        """Эмбеддинг текста с кэшем; "" → None (нет контекста)."""

    def recall_prior(self, query: Vector | None) -> Vector:
        """Приор из top-1 похожего эпизода; пусто/None → нули."""

    def maybe_store(
        self, *, text: str, query: Vector | None, f: float, prev_f: float,
        valence: float, stress: float, active_tags: tuple[str, ...],
        reflex_tags: tuple[str, ...], tick: int,
    ) -> int | None:
        """Записать эпизод, если событие значимо; иначе None.

        content = text или build_event_content(...); embedding = query или
        embed(content). Ошибки store → логируются, возврат None (память не
        должна ронять тик — в отличие от MemoryStore, здесь fire-and-forget
        на уровне loop: сбой памяти не критичен для текущего тика).
        """
```

## 5. Проводка в loop

`HostLoop` получает два новых поля:

```python
memory: MemoryRouter | None = None
message_provider: TextMessageProvider | None = None
```

Порядок тика (изменения относительно S1 выделены):

```
dt, now  = time source
u_base   = bus.step(tick, now)
text     = message_provider.text_at(tick)           # НОВОЕ
# Событийно: эмбеддинг+recall только при смене текста (не каждый тик!)
if text != _last_text:
    query = memory.context_embedding(text)           # НОВОЕ (event-triggered)
    prior = memory.recall_prior(query)               # НОВОЕ
    _last_text, _cached_query, _cached_prior = ...
prior    = _cached_prior (нули, если сообщений не было)
u        = concat(u_base, prior)                     # НОВОЕ
segments = bus.segments + (memory_segment,)          # НОВОЕ
γ        = precision(u)                              # dim = bus_dim + prior_dim
outcome  = pipeline.tick(u, γ, dt, segments, reflex_tags)
check_finite(outcome.result)
drift    = DriftDetector.update(outcome.result)
memory.maybe_store(...)                              # НОВОЕ
telemetry.log(...)
```

**Коммуникативный вход — событийный** (манифест §3.Е, ADR-0006): тик —
непрерывный аффективный контур, а текст обрабатывается **один раз при
появлении**. Между сообщениями сеть/recall не трогаются, используется
сохранённый приор (нули до первого сообщения). Приор держится до следующего
сообщения (затухание веса — BACKLOG).

- `memory_segment = BusSegment(name="memory", offset=bus.bus_dim,
  dim=prior_dim, period=1)`.
- Колонки создаются под `total_dim = bus.bus_dim + prior_dim`; estimator — тоже.
- `memory is None` (или `enabled=False`) → `prior` не добавляется, поведение S1.

## 6. Конфигурация

Новая `MemoryConfig` (+ поле `memory` в `HostConfig`):

| Поле | Дефолт | Смысл |
|---|---|---|
| `enabled` | True | Включить память в loop |
| `embedder_mode` | "auto" | "auto" (ключ→api, иначе fake), "fake", "api" |
| `embedding_dim` | 8 | Размерность fake-эмбеддера (= dim сообщения) |
| `embedding_model` | "text-embedding-3-small" | Модель API |
| `db_path` | "host_memory.db" | Файл БД памяти |
| `episode_spike_threshold` | 1.0 | Порог всплеска F для эпизода |
| `recall_limit` | 1 | Сколько эпизодов извлекать |
| `prior_dim` | 4 | Размерность приора в шине |

Валидация: `embedding_dim > 0`, `prior_dim > 0`, `recall_limit ≥ 1`,
`episode_spike_threshold ≥ 0`, `embedder_mode ∈ {auto, fake, api}`.

**API-эмбеддер `api` без ключа** → `ValueError` при сборке (fail-fast);
`auto` без ключа → `fake` (не ошибка).

**Миграция:** `HostConfig.message_dim` заменяется на
`MemoryConfig.embedding_dim` (fake-эмбеддер использует его как размерность
сообщения). Тесты, хардкодящие `message_dim`/`bus_dim=14`, обновляются.

CLI: `--embedder {fake,api}`, `--db PATH`, `--no-memory`.

## 7. Персистентность и детерминизм

- БД памяти — файл (`db_path`); store/recall персистентны (готово в memory).
- `embedder_mode="fake"` + `clock_mode="synthetic"` + свежая БД → replay.
- Со «второго запуска» на той же БД поведение отличается — это и есть ворота.

## Инварианты S2

1. **Приор ограничен:** `encode_memory_prior` ∈ `[-1, 1]⁴` (tanh); нет NaN/inf.
2. **Пустая память → нулевой приор:** recall `[]` → вектор нулей; F как в S1.
3. **Память не роняет тик:** сбой store/recall логируется; loop продолжает.
4. **Дедупликация:** одинаковый `content` → одна запись (наследуется).
5. **Детерминизм:** fake-эмбеддер + synthetic + свежая БД → идентичный прогон.
6. **Персистентность:** store → close → reopen → recall находит (наследуется).
7. **Память влияет на поведение:** при непустой релевантной памяти F-трасса
   отличается от прогона с пустой памятью — измеримо.
8. **Обратная совместимость:** `memory=None`/`enabled=False` → контур S1.
9. **Наблюдаемость:** факт recall/записи виден в телеметрии (см. §8).

## 8. Телеметрия (минимальное расширение)

К 15 полям S1 добавляются (обратно совместимо, дефолты):

| Поле | Тип | Смысл |
|---|---|---|
| `memory_prior` | float | `cos` извлечённого эпизода (0.0 если нет) |
| `memory_hit` | bool | recall нашёл релевантный эпизод |
| `episode_stored` | bool | эпизод записан на этом тике |

Расширение обосновано инвариантом наблюдаемости (§2.5 VALIDATION): значимое
событие (запись/recall) обязано оставлять след.

## Критерии приёмки

См. `VALIDATION.md` §4 (ворота S2). Кратко:
- [ ] `FakeEmbedder`: детерминирован; похожий текст → выше косинус; "" → нули
- [ ] `TextMessageProvider`: текст → эмбеддинг; `text_at` корректен
- [ ] `is_significant_event` / `build_event_content` — чистое ядро, покрыто
- [ ] `encode_memory_prior`: форма/границы; пусто → нули
- [ ] `MemoryRouter`: recall→приор; запись на событии; сбой не роняет
- [ ] эпизод записывается на значимом событии (всплеск F / reflex)
- [ ] после перезапуска recall возвращает релевантный эпизод
- [ ] поведение (F) зависит от прошлого — измеримо
- [ ] инварианты §2 VALIDATION (6, 8) держатся
- [ ] `ruff check`/`ruff format` чисто; mypy strict для `src/memory/`; тесты зелёные

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| Категория сигнала памяти | Решено (S2) | Сегмент шины, не `SignalCategory`; кандидат `METACOGNITIVE` — S4 |
| Приор в шину vs точность γ | Решено (S2) | **Два этапа:** приор в шину в S2; recall→γ в S4 (пред-колоночный барьер) |
| Режим эмбеддера | Решено | `auto` (ключ→api, иначе fake); `fake`/`api` явно; `api` без ключа → ошибка |
| `embedding_dim` vs `message_dim` | Решено | Замена на `MemoryConfig.embedding_dim` |
| `prior_dim` | Решено | 4 (наращиваемо позже) |
| Порог всплеска F | Решено | 1.0 стартовое; калибровка на воротах |
| Проекция эмбеддинга (1536→D) | Отложено | S4+ (`ColumnConfig.reads`); в S2 dim как есть |
| Дебаунс записи эпизодов | Решено | Не писать, если последний эпизод идентичен (дедуп по content_hash уже есть) |
| Ночная консолидация | Отложено | S6 |
| ADR по памяти-приору | Отложено | Кандидат ADR-0007 при стабилизации |

## Implementation Notes

1. **FC/IS:** `embedder`/`prior`/`events` — Core+Shell с DI; `MemoryRouter` —
   Shell; `MemoryStore` не меняется.
2. **Кэш эмбеддинга:** `MemoryRouter` хранит `(last_text, last_embedding)`;
   повторный текст не эмбеддится (экономия API).
3. **Пустой запрос:** `query is None` или `‖query‖ == 0` → recall не вызывается,
   приор нули (не ищем «по нулю»).
4. **Сбой памяти не критичен:** `maybe_store`/`recall_prior` ловят
   `MemoryStoreError`/`EmbedderError` → `logging.error` → безопасный дефолт.
   Отличие от `MemoryStore` (fail-fast для caller'а): loop продолжает тик.
5. **Детерминизм тестов:** всегда `FakeEmbedder` + свежая БД на `tmp_path`.
6. **bus_dim:** 14 → 18 (при `prior_dim=4`); тесты с хардкодом обновить.
7. **Reference:** `TelemetryWriter` (Shell owning resource), `ResourceProvider`
   (инъекция для детерминизма), `SupportsWrite` (Protocol DI).