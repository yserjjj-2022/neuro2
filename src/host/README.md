# host

Обвязка хоста: связывает модули в работающую систему и оживляет её во времени.

- `sources.py` — сенсорная шина: провайдеры сигналов → `u(t)`, карта сегментов.
- `resources.py` — `ResourceMeter`/`ResourceProvider` (интероцепция ресурсов).
- `wiring.py` — чистая композиция `CMCPipeline` → `TickOutcome` (без I/O).
- `loop.py` — `HostLoop`: время (synthetic/wall, time_scale), precision,
  ресурсы, guard, drift, телеметрия.

S1: телеметрия переехала в loop; шина 14 каналов; человеческий темп (10 Гц).
См. `SPEC.md`, `PLAN.md` и ADR-0006. Точка входа: `uv run python -m src`.