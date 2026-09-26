# MCP / Sensory Contracts

Контракт сенсорных сигналов: разведение входящих сигналов на три категории
(манифест §3.З).

- `SignalCategory`: exteroceptive / interoceptive / communicative
- `SignalSource`: категория + data-вектор + severity + is_reflex + tag
- `SignalRegistry`: регистрация источников и агрегация в шину `u(t)`

Инварианты: `severity ≥ 0.9` у interoceptive → `is_reflex=True` (reflex-path);
`is_reflex` только для interoceptive; `severity ∈ [0, 1]`.

Статус: контракт данных готов. MCP-транспорт (resources/tools) — не реализован.
Провайдеры сигналов и карта сегментов шины — `src/host/sources.py`.