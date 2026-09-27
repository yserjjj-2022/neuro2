# host

Обвязка хоста: связывает модули в работающую систему и оживляет её во времени.

- `sources.py` — сенсорная шина: провайдеры сигналов → `u(t)`, карта сегментов.
- `resources.py` — `ResourceMeter`/`ResourceProvider` (интероцепция ресурсов).
- `text_source.py` — `TextMessageProvider` (текст → эмбеддинг, S2).
- `wiring.py` — чистая композиция `CMCPipeline` → `TickOutcome` (без I/O).
- `throttle.py` — `ThrottlePlan`/`plan_throttle` (рефлекс-throttle, S4).
- `gate.py` — `CapabilityGate` (единая точка side-effect, S4 заготовка).
- `control.py` — `ControlChannel` (status/pause/resume/step, S4).
- `loop.py` — `HostLoop`: время (synthetic/wall, time_scale), precision,
  гомеостаз, throttle, attention-барьер, ресурсы, guard, drift, память
  (recall→приор, запись), policy-контекст, телеметрия.

S1: телеметрия переехала в loop; человеческий темп (10 Гц).
S2: память подключена — приор (4 канала) добавляется к шине (`total_dim` =
bus_dim + prior_dim), эпизоды пишутся на значимых событиях.
S4: гомеостаз → рефлекс-throttle (≤ 1 тик, обратимый) до pipeline;
пред-колоночный γ-барьер; policy-контекст и телеметрия 23 поля; gate/control.
См. `SPEC.md`, `PLAN.md`, `stages/S4_SPEC.md` и ADR-0006.
Точка входа: `uv run python -m src`.