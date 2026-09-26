# PLAN.md — S2: Непрерывность

Реализация `stages/S2_SPEC.md`. Порядок — от чистого ядра к обвязке, каждый
шаг с тестами. Модульные SPEC (`src/memory/SPEC.md`, `src/host/SPEC.md`,
`src/config/SPEC.md`, `src/telemetry/SPEC.md`) обновляются по ходу.

## Файлы

### Новые

1. `src/memory/embedder.py` — `Embedder` (Protocol), `EmbedderError`,
   `FakeEmbedder` (Core), `ApiEmbedder` (Shell, API)
2. `src/memory/events.py` — `is_significant_event`, `build_event_content` (Core)
3. `src/memory/prior.py` — `MEMORY_PRIOR_DIM`, `encode_memory_prior` (Core)
4. `src/memory/router.py` — `MemoryRouter` (Shell: recall→приор, запись)
5. `src/host/text_source.py` — `TextMessageProvider` (текст → эмбеддинг)
6. `src/tests/test_memory_embedder.py`
7. `src/tests/test_memory_events.py`
8. `src/tests/test_memory_prior.py`
9. `src/tests/test_memory_router.py`
10. `src/tests/test_host_text_source.py`
11. `stages/S2_SPEC.md`, `stages/S2_PLAN.md` — эти файлы

### Изменяемые

1. `src/memory/__init__.py` — re-export новых сущностей
2. `src/memory/SPEC.md` — эмбеддер, events, prior, router
3. `src/host/loop.py` — `memory`, `message_provider`; prior-сегмент; запись;
   новые поля телеметрии
4. `src/host/sources.py` — `default_providers` без `UserMessageProvider`
   (заменяется `TextMessageProvider` вне шины) ИЛИ оставить как fallback
5. `src/host/wiring.py` — `total_dim` = bus_dim + prior_dim (колонки/estimator)
6. `src/host/SPEC.md` — память в тике
7. `src/config/params.py` — `MemoryConfig` + поле `memory`; валидация
8. `src/config/SPEC.md` — `MemoryConfig`
9. `src/telemetry/models.py` — 3 поля (`memory_prior`, `memory_hit`,
   `episode_stored`)
10. `src/telemetry/logger.py` — новые параметры `log()`
11. `src/telemetry/SPEC.md` — 18 полей
12. `src/__main__.py` — `--embedder`, `--db`, `--no-memory`
13. `src/tests/test_host_loop.py`, `test_behavioral_regress.py`, `test_config.py`,
    `test_telemetry_*` — адаптация + новые сценарии

## Зависимости

- **Внешние:** `numpy`; `openai` (уже в pyproject); stdlib `hashlib`, `logging`,
  `os`, `pathlib`.
- **Внутренние:** memory ← (embedder/events/prior); host ← memory; config ← memory.

## Порядок реализации

### Шаг 1. Embedder (Core + Shell)

1. `Embedder` Protocol (`dim`, `embed`); `EmbedderError`.
2. `FakeEmbedder`: bag-of-tokens hashing (sha256 → bucket), L2-норм; `""` → нули.
3. `ApiEmbedder`: ленивый клиент OpenAI, `EmbedderError` при сбое.
4. `build_embedder(mode, dim, model, api_key=None)`: `auto` (ключ→api, иначе
   fake), `fake`, `api` (без ключа → ValueError).
5. Тесты: детерминизм; похожий текст → выше косинус; разные → ниже; `""` → нули;
   форма/конечность; `dim` совпадает; `build_embedder` режимы. API — только
   контракт (без сети).

### Шаг 2. Events + Prior (Core)

1. `is_significant_event(f, prev_f, reflex_tags, spike_threshold)`.
2. `build_event_content(active_tags, reflex_tags, valence, stress)`.
3. `MEMORY_PRIOR_DIM=4`; `encode_memory_prior(episode, query)`.
4. Тесты: событие по всплеску/reflex/нет; content непуст и детерминирован;
   приор — форма/границы; `None` → нули; косинус корректен.

### Шаг 3. MemoryRouter (Shell)

1. `__init__(store, embedder, spike_threshold, recall_limit, prior_dim)`.
2. `context_embedding(text)` с кэшем (`""` → None).
3. `recall_prior(query)` → top-1 → `encode_memory_prior`.
4. `maybe_store(...)` → событие → `Episode` → `store.store`; сбой → None.
5. Тесты: recall→приор; запись на событии; дедуп; сбой store/recall не роняет
   (fake store, бросающий `MemoryStoreError`); кэш эмбеддинга.

### Шаг 4. TextMessageProvider (host)

1. `text_at(tick)` — последнее сообщение с `tick' ≤ tick`.
2. `read(tick, now)` → `embed(text)` в `SignalSource`.
3. Тесты: эмбеддинг в data; пустой скрипт → нули; `text_at` границы.

### Шаг 5. Telemetry

1. +3 поля (`memory_prior: float`, `memory_hit: bool`, `episode_stored: bool`).
2. `log(...)` — параметры с дефолтами (обратная совместимость).
3. Тесты: сериализация 18 полей; дефолты.

### Шаг 6. Config

1. `MemoryConfig` (`enabled`, `embedder_mode="auto"`, `embedding_dim`,
   `embedding_model`, `db_path`, `episode_spike_threshold`, `recall_limit`,
   `prior_dim`); валидация (`embedder_mode ∈ {auto, fake, api}`).
2. `HostConfig.memory: MemoryConfig`; `message_dim` → `memory.embedding_dim`.
3. Тесты: дефолты; валидация (dim>0, mode, recall_limit≥1).

### Шаг 7. Wiring: total_dim + сегмент памяти

1. Колонки/estimator создаются под `bus_dim + prior_dim`.
2. `memory_segment` = `BusSegment("memory", offset=bus_dim, dim=prior_dim)`.
3. `build_host_loop` собирает `MemoryStore`/`MemoryRouter`/`TextMessageProvider`
   из `MemoryConfig` (или None при `enabled=False`).
4. Тесты: total_dim; сегмент присутствует; `memory=None` → S1-контур.

### Шаг 8. Loop: prior + store + телеметрия

1. Тик: `text → query → prior → u=concat → pipeline → store`.
2. `_prev_f` для детектора всплеска (в loop или observer).
3. Заполнение `memory_prior`/`memory_hit`/`episode_stored`.
4. Тесты: synthetic детерминирован; пустая память = S1; запись на всплеске;
   приор влияет на F.

### Шаг 9. CLI

1. `--embedder {fake,api}`, `--db PATH`, `--no-memory`.
2. Тесты argparse.

### Шаг 10. Адаптация + регресс

1. Обновить тесты под 18 полей и `bus_dim=18`.
2. `test_behavioral_regress.py`: C5 (bus_dim), новый C11 (память: запись+recall);
   `behavioral_fingerprint` — добавить `memory_hits`/`episodes`.
3. Обновить SPEC/README/SPECS/BACKLOG.

## План тестов

| Тест | Покрытие | Инвариант |
|---|---|---|
| `test_fake_embedder_deterministic` | эмбеддер | replay |
| `test_fake_embedder_similarity` | эмбеддер | похожий → выше cos |
| `test_fake_embedder_empty` | edge | "" → нули |
| `test_is_significant_event_*` | событие | spike/reflex |
| `test_build_event_content` | content | непуст, детерминирован |
| `test_encode_memory_prior_*` | приор | форма/границы/None |
| `test_router_recall_prior` | recall→приор | влияние |
| `test_router_maybe_store` | запись | событие |
| `test_router_swallows_errors` | crash-safety | loop жив |
| `test_text_provider_*` | текст | эмбеддинг в data |
| `test_telemetry_memory_fields` | 18 полей | сериализация |
| `test_config_memory_*` | валидация | границы |
| `test_loop_memory_off_is_s1` | совместимость | memory=None |
| `test_loop_stores_on_spike` | значимость | запись |
| `test_loop_recall_after_reopen` | персистентность | ворота S2 |
| `test_loop_prior_changes_f` | влияние | F отличается |
| `test_behavioral_c11_memory` | сценарий | запись+recall |

## Заметки

- **Не ломать S1:** `memory=None`/`enabled=False` → идентичный S1-контур
  (отдельный тест на каждом шаге, где возможно).
- **bus_dim 14 → 18:** при `prior_dim=4`; тесты с хардкодом 14 обновить.
- **fake-эмбеддер по умолчанию:** детерминизм и replay без сети/ключей.
- **Кэш эмбеддинга:** не звать `embed` дважды для одного текста (API-экономия).
- **Сбой памяти не критичен:** loop продолжает; ошибка логируется.
- **Порядок**: memory перед telemetry; `_prev_f` для всплеска — из observer.state.
- **Миграция `message_dim`:** заменяется `memory.embedding_dim`; обновить
  `default_providers` и тесты `test_host_sources.py`.
- **Интеграция store:** `MemoryStore(db_path, embedding_dim)` создаётся в
  `build_host_loop`; `close()` — в `HostLoop.close()`.
- **Reference:** S1 PLAN (порядок Core→Shell→Loop), `ResourceProvider`
  (инъекция для детерминизма), `SupportsWrite` (Protocol DI).