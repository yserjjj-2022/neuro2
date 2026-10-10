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
  (`rank₀`)
- `registry.py` — `IntegrationRegistry`, `default_integrations()`
- `toml_loader.py` — `load_integrations` (Python-база + TOML-override)
- `bridges.py` — `to_affordances` (TOOL) — Core, чистый
- `factories.py` — `SensorContext`, `build_provider(s)` (SENSOR → провайдеры
  шины) — Shell-glue
- `runtime.py` — `ProbeTransport`, `connect_probe_transport` — Shell (MCP-клиенты)

MCP-транспорт: `src/mcp/client.py` (`MCPClient.list_tools`/`call_tool`,
sync-обёртка над async SDK).

## Сборка шины

`build_host_loop` собирает `u(t)` из **SENSOR-записей реестра** (порядок
записей = порядок укладки каналов). `rank₀` в записи (`IntegrationSpec.rank`)
включает веса каналов (`wᵢ = rank/dim`); без рангов — legacy `F = 0.5·Σγ·e²`.
Набор сенсоров на экземпляр киберперсоны задаётся составом реестра.

```bash
uv run python -m src --integrations configs/integrations.toml --ticks 5
```

Без флага: шина из `default_integrations()` (идентична прежней), карта
аффордансов — `default_affordances()` (контур S6).

## Ростер

| Вид | Запись | Транспорт | Provenance |
|---|---|---|---|
| SENSOR | `circadian`, `battery`, `cpu`, `message`, `resources` | local | local |
| TOOL | `server-everything` | stdio `npx` | official |
| TOOL | `mcp-server-time` | stdio `uvx` | official |
| TOOL | `weather-mcp` | stdio `npx` | official |
| TOOL | `web-search-mcp` | stdio `npx` | community (off) |

Документы: [SPEC](SPEC.md), [PLAN](PLAN.md); решения — ADR-0011.
Статус: **реализовано** (реестр, TOML-override, мосты, MCP-клиент stdio,
сборка из реестра через `--integrations`). HTTP-транспорт и ACTION-вид —
отложены.
