# PLAN.md — src/memory

## Файлы

1. `src/memory/models.py` — `Episode` dataclass ✅
2. `src/memory/similarity.py` — `cosine_similarity()` (чистое ядро) ✅
3. `src/memory/serialize.py` — `serialize_embedding()`, `deserialize_embedding()` ✅
4. `src/memory/hash.py` — `content_hash()` ✅
5. `src/memory/errors.py` — `MemoryStoreError` ✅
6. `src/memory/store.py` — `MemoryStore` (SQLite + sqlite-vec) ✅
7. `src/memory/protocols.py` — `SupportsStore`, `SupportsRecall` ✅
8. `src/memory/embedder.py` — `Embedder`, `FakeEmbedder`, `ApiEmbedder`, `build_embedder` ✅ (S2)
9. `src/memory/events.py` — `is_significant_event`, `build_event_content` ✅ (S2)
10. `src/memory/prior.py` — `MEMORY_PRIOR_DIM`, `encode_memory_prior` ✅ (S2)
11. `src/memory/router.py` — `MemoryRouter` ✅ (S2)
12. `src/memory/__init__.py` — re-exports ✅
13. `src/tests/test_memory_*.py` ✅

## Зависимости

**Внешние:** `numpy`, `sqlite-vec`, `openai` (ленивый, только `ApiEmbedder`).
**Стандартная библиотека:** `dataclasses`, `typing`, `pathlib`, `sqlite3`,
`hashlib`, `logging`, `os`.

## S2 (выполнено)

1. `FakeEmbedder` (bag-of-tokens hashing, детерминизм) + `ApiEmbedder` +
   `build_embedder(mode=auto|fake|api)`.
2. `is_significant_event` / `build_event_content` (Core).
3. `encode_memory_prior` (Core, dim=4, ограничен tanh).
4. `MemoryRouter` (recall→приор, запись на событиях, кэш, crash-safety).
5. Wiring в host loop (см. `src/host/PLAN.md`).

## Заметки

- **Приор в шину (S2), recall→γ (S4)** — два этапа (ADR-0005 §2).
- **`auto` дефолт:** ключ→api, иначе fake (тесты детерминированы).
- **Сбой памяти не роняет тик:** router ловит MemoryStoreError/EmbedderError.
- **DDL без placeholder'а:** `float[{embedding_dim}]`; валидация dim обязательна.
- **Прямой JOIN с vec0 зависает** (баг 0.1.9) — только подзапрос.
- **Родительская директория БД:** создаётся в `_open()` (S2).
- **Reference:** `TelemetryWriter`, `SupportsWrite`, `ResourceProvider`.