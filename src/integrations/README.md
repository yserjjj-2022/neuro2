# integrations — реестр интеграций

**Каталог** органов хоста: какие сенсоры/тулы/действия подключены, как их
запускать, чем они являются и откуда взяты. Источник истины для сборки
runtime-артефактов (провайдеры шины, карта аффордансов).

Разведение реестров (ADR-0011 §1):

| Реестр | Роль |
|---|---|
| `IntegrationRegistry` (этот модуль) | каталог подключений (конфиг) |
| `AffordanceMap` (`src/mcp/probe.py`) | runtime-вид «что доступно сейчас» (`tools/list`) |
| `SignalRegistry` (`src/mcp/registry.py`) | агрегация сигналов в `u(t)` |

## Состав

- `models.py` — `IntegrationKind`, `Provenance`, транспорты, `IntegrationSpec`
- `registry.py` — `IntegrationRegistry`, `default_integrations()`
- `toml_loader.py` — `load_integrations` (Python-база + TOML-override)
- `bridges.py` — `to_provider` (SENSOR), `to_affordances` (TOOL) — Core
- `runtime.py` — `ProbeTransport`, `connect_probe_transport` — Shell (MCP-клиенты)

MCP-транспорт: `src/mcp/client.py` (`MCPClient.list_tools`/`call_tool`,
sync-обёртка над async SDK).

## Использование

```bash
uv run python -m src --integrations configs/integrations.toml --ticks 5
```

Без флага контур S6 идентичен (mock-транспорт + `default_affordances()`).

## Ростер (порядок подключения)

| Приоритет | Сервер | Транспорт | Provenance |
|---|---|---|---|
| 1 | `server-everything` | stdio `npx` | official |
| 1 | `mcp-server-time` | stdio `uvx` | official |
| 2 | `weather-mcp` | stdio `npx` | official |
| 3 | `web-search-mcp` | stdio `npx` | community |

Документы: [SPEC](SPEC.md), [PLAN](PLAN.md); решения — ADR-0011.
Статус: **реализовано** (реестр, TOML-override, мосты, MCP-клиент stdio,
сборка из реестра через `--integrations`). HTTP-транспорт и ACTION-вид —
отложены.
