# host

Обвязка хоста: связывает модули в работающую систему и оживляет её во времени.

- `sources.py` — сенсорная шина: провайдеры сигналов → `u(t)`, карта сегментов.
- `resources.py` — `ResourceMeter`/`ResourceProvider` (интероцепция ресурсов).
- `text_source.py` — `TextMessageProvider` (текст → эмбеддинг, S2).
- `wiring.py` — чистая композиция `CMCPipeline` → `TickOutcome` (без I/O).
- `loop.py` — `HostLoop`: время (synthetic/wall, time_scale), precision,
  ресурсы, guard, drift, память (recall→приор, запись), телеметрия.

S1: телеметрия переехала в loop; человеческий темп (10 Гц).
S2: память подключена — приор (4 канала) добавляется к шине (`total_dim` =
bus_dim + prior_dim), эпизоды пишутся на значимых событиях.
См. `SPEC.md`, `PLAN.md` и ADR-0006. Точка входа: `uv run python -m src`.