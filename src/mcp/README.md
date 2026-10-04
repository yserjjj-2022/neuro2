# MCP / Sensory Contracts

Контракт сенсорных сигналов: разведение входящих сигналов на три категории
(манифест §3.З).

- `SignalCategory`: exteroceptive / interoceptive / communicative
- `SignalSource`: категория + data-вектор + severity + is_reflex + tag
- `SignalRegistry`: регистрация источников и агрегация в шину `u(t)`

Инварианты: `severity ≥ 0.9` у interoceptive → `is_reflex=True` (reflex-path);
`is_reflex` только для interoceptive; `severity ∈ [0, 1]`.

S6 (автономия, проход 2) добавлено:
- `probe.py`: карта аффордансов (`Affordance`/`AffordanceMap`) + чистая
  `select_affordance` (эпистемическое зондирование по неопределённости).
- Исполнение — `src/host/probe.py::ProbeEffector` (Shell) через capability
  gate (обратимое — автономно T3, необратимое — с HITL T4, fail-safe deny).

Статус: контракт данных + карта аффордансов готовы. MCP-транспорт реализован
(`client.py`: `MCPClient.list_tools`/`call_tool` — sync-обёртка над async
`mcp.ClientSession`, stdio); реестр интеграций и сборка — `src/integrations/`.
HTTP-транспорт и MCP Resources (SENSOR-ветка) — отложены.
Провайдеры сигналов и карта сегментов шины — `src/host/sources.py`.