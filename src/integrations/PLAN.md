# PLAN.md — src/integrations

Реализация реестра интеграций и MCP-транспорта. Решения — ADR-0011,
интерфейс — `SPEC.md`. Порядок — SDD: Spec → Plan → Tasks (BACKLOG) → код.

> **Статус: реализовано.** P1-объём выполнен и проверен (819 тестов, ruff/mypy
> чисто, e2e на `mcp-server-time`). Переезд сборки шины на реестр выполнен
> (SENSOR-записи → провайдеры, `rank₀` в `IntegrationSpec`). Отложено (P2):
> HTTP-транспорт, MCP Resources→u(t), ACTION-вид.

## Файлы для создания/изменения

1. `src/integrations/__init__.py` — re-exports.
2. `src/integrations/models.py` — `IntegrationKind`, `Provenance`, транспорты,
   `IntegrationSpec` + `tool_args` (Core).
3. `src/integrations/registry.py` — `IntegrationRegistry` (Shell) +
   `default_integrations()`.
4. `src/integrations/toml_loader.py` — `load_integrations` (Shell, `tomllib`).
5. `src/integrations/bridges.py` — `to_affordances` (Core, чистый).
6. `src/integrations/factories.py` — `SensorContext` + `build_provider`/
   `build_providers`/`enabled_sensors` (Shell-glue; SENSOR → провайдеры шины).
7. `src/mcp/client.py` — `MCPClient`, `ToolInfo`, `ToolResult` (Shell).
8. `configs/integrations.toml` — пример override.
9. `src/tests/test_integrations_models.py` — модели (Core).
10. `src/tests/test_integrations_registry.py` — реестр (Shell).
11. `src/tests/test_integrations_toml.py` — TOML-override (Shell).
12. `src/tests/test_integrations_bridges.py` — мосты/фабрики (Core/Shell).
13. `src/tests/test_integrations_runtime.py` — runtime-glue + проводка (Shell).
14. `src/tests/test_mcp_client.py` — клиент на stdio fake-сервере.
15. `src/host/loop.py`, `src/host/wiring.py` — сборка шины из реестра (SENSOR →
    провайдеры, `rank₀` → importance); CLI-флаг `--integrations PATH`.
16. `src/__main__.py` — флаг `--integrations`.
17. `src/integrations/README.md` — обзор.

## Зависимости

**Внешние:** `mcp>=2.0.0` (уже в `pyproject.toml`), `numpy` (через провайдеры).

**Стандартная библиотека:** `dataclasses`, `enum`, `tomllib`, `pathlib`,
`typing`.

**Внутренние:** `src.mcp.probe` (`Affordance`, `AffordanceMap`),
`src.host.sources` (`SignalProvider`, провайдеры), `src.mcp.models`
(`SignalCategory`).

## Порядок реализации

### 1. Модели (`models.py`, Core)

- `IntegrationKind` (SENSOR/TOOL/ACTION), `Provenance` (LOCAL/OFFICIAL/COMMUNITY).
- `StdioTransport`/`HttpTransport`/`LocalTransport` (frozen, валидация).
- `IntegrationSpec` (frozen): name/kind/transport/category/provenance/
  reversible/period/enabled/tools.
- Валидация: непустой name; `period >= 1`; `LocalTransport` только SENSOR.

### 2. Реестр (`registry.py`, Shell)

- `IntegrationRegistry`: add (дубль → ValueError), find, specs, by_kind,
  by_category, enabled.
- `default_integrations()`: everything/time/weather (official), web-search
  (community, disabled), локальные (circadian/resources).

### 3. TOML-override (`toml_loader.py`, Shell)

- `load_integrations(override)`: база + `[[integrations]]` из TOML.
- Fail-fast: неизвестный ключ/kind/транспорт → ValueError.
- `configs/integrations.toml` — пример.

### 4. Мосты и фабрики (`bridges.py` Core, `factories.py` Shell)

- `to_affordances(specs)`: TOOL → `AffordanceMap` (Core, чистый).
- `build_provider(spec, ctx)`: SENSOR → провайдер (Shell-glue; `SensorContext`
  инжектит seed/meter/message-provider). MCP-сенсор → `NotImplementedError`.
- `build_providers(specs, ctx)`: порядок = порядок записей → укладка на шину.

### 5. MCP-клиент (`src/mcp/client.py`, Shell)

- `MCPClient`: stdio через `mcp.client.stdio.stdio_client` + `ClientSession`.
- `list_tools()`, `call_tool()`, `close()`; таймауты; сбои → типизированно.
- Async↔sync мост **решён:** воркер-поток с event loop, соединение и сессия
  открываются/закрываются в одной корутине (anyio cancel scopes).

### 6. Проводка

- `loop.py`: `bus` собирается из `IntegrationRegistry` (SENSOR-записи) через
  `build_providers`; `rank₀` из записей → `importance`; `probe_fn` — из
  `MCPClient` (если включён); fallback — mock.
- CLI `--integrations PATH`.
- Обратная совместимость: `integrations=None` → `default_integrations()` (порядок
  SENSOR идентичен `default_providers()`); карта аффордансов — `default_affordances()`.

## План тестов

| Тест | Покрытие | Инвариант |
|---|---|---|
| `test_models_validation` | модели | ValueError на некорректном |
| `test_registry_add_duplicate` | реестр | дубль → ValueError |
| `test_registry_filters` | реестр | by_kind/by_category/enabled |
| `test_load_toml_override` | загрузка | override поверх базы |
| `test_load_toml_fail_fast` | загрузка | неизвестный ключ → ValueError |
| `test_to_provider_local` | мост | SENSOR → SignalProvider |
| `test_build_provider_local` | фабрика | SENSOR → SignalProvider |
| `test_build_providers_order` | фабрика | порядок укладки = порядок записей |
| `test_bus_from_registry_subset` | проводка | узкая шина из подмножества |
| `test_to_affordances` | мост | TOOL → AffordanceMap |
| `test_mcp_client_list_tools` | клиент | fake-сервер → тулы |
| `test_mcp_client_call_tool` | клиент | вызов → результат |
| `test_mcp_client_error_safe` | клиент | сбой → не роняет |
| `test_loop_uses_registry` | проводка | карта из реестра |
| `test_s6_compat_without_registry` | совместимость | контур S6 идентичен |

## Заметки для реализации

- **FC/IS:** модели/мосты — чистые; TOML и клиент — Shell.
- **Единый источник:** `default_affordances()` (mcp) → fallback, если реестр
  пуст; иначе карта из реестра.
- **Async↔sync:** mcp SDK async; синхронный loop требует моста. **Решено:**
  выделенный event-loop поток + одна долгоживущая корутина владеет соединением
  (все enter/exit в одной задаче — требование anyio).
- **Fake-сервер для тестов:** stdio-подпроцесс (`src/tests/fake_mcp_server.py`)
  без сети; в `test_integrations_runtime.py` — fake-клиент (инъекция).
- **Provenance:** `OFFICIAL` только для reference/официального реестра;
  `web-search` — `COMMUNITY`.
- **CLI:** `--integrations PATH` (override), по аналогии с `--preset-file` (S7).
