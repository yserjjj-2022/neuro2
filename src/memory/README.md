# Episodic Memory

Эпизодическая память хоста: SQLite + sqlite-vec.

Быстрый буфер прецедентов (правки оператора, ситуации): текст + эмбеддинг +
аффективный контекст (valence, stress, free_energy) на момент эпизода.

Архитектура (ADR-0001 — SQLite over Postgres, ADR-0004 — FC/IS):
- Чистое ядро: `cosine_similarity`, `serialize_embedding`/`deserialize_embedding`,
  `content_hash`, `is_significant_event`/`build_event_content`,
  `encode_memory_prior` — без I/O, тестируются в изоляции
- Shell: `MemoryStore` — единственное место SQLite I/O, владеет соединением
- Дедупликация: SHA-256 по content (`INSERT OR IGNORE` + rowcount)
- Векторный поиск: vec0 (sqlite-vec), MATCH в подзапросе (обход бага 0.1.9)

S2 (непрерывность) добавлено:
- `Embedder` Protocol + `FakeEmbedder` (детерминизм) + `ApiEmbedder` (OpenAI)
  + `build_embedder` (режим `auto`: ключ→api, иначе fake)
- `MemoryRouter` — оркестрация: recall → приор в шину, запись эпизодов на
  значимых событиях (reflex / всплеск F); сбой памяти не роняет тик
- Приор `[cos, tanh(valence), tanh(stress), tanh(f)]` (dim=4) — сегмент шины

Статус: подключено к host loop (S2). Фаза 2+: консолидация во сне,
Structure Learning (schemas), EvolvingSteeringMemory — манифест §3.Д.
