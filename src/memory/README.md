# Episodic Memory

Эпизодическая память хоста: SQLite + sqlite-vec.

Быстрый буфер прецедентов (правки оператора, ситуации): текст + эмбеддинг +
аффективный контекст (valence, stress, free_energy) на момент эпизода.

Архитектура (ADR-0001 — SQLite over Postgres, ADR-0004 — FC/IS):
- Чистое ядро: `cosine_similarity`, `serialize_embedding`/`deserialize_embedding`,
  `content_hash` — без I/O, тестируются в изоляции
- Shell: `MemoryStore` — единственное место SQLite I/O, владеет соединением
- Дедупликация: SHA-256 по content (`INSERT OR IGNORE` + rowcount)
- Векторный поиск: vec0 (sqlite-vec), MATCH в подзапросе (обход бага 0.1.9)

Статус: Фаза 1 — store/recall готовы. Не подключено к host loop:
нет источника content/embedding (BACKLOG `[Phase1][memory-wiring] Blocked`).

Фаза 2+: консолидация во сне, Structure Learning (schemas) — манифест §3.Д.
