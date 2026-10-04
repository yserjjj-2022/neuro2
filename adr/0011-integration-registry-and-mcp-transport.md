# ADR-0011: Реестр интеграций и MCP-транспорт

## Date
2026-10-04

## Status
Accepted

## Context

S6 проход 2 замкнул контракт MCP-зондирования: карта аффордансов
(`src/mcp/probe.py`) + gated-эффектор (`src/host/probe.py`). Но **транспорта
нет**: `_mock_probe` возвращает нулевой вектор, карта аффордансов захардкожена
(`default_affordances()`), а внешние «органы чувств» (MCP Resources/Tools) не
подключены. Это осознанный долг `[Phase2][sensors] Реальные интеграции`
(BACKLOG) и `MCP transport (не реализован)` (`src/mcp/README.md`).

Проблемы, которые надо развести:

1. **Нет источника истины о подключениях.** Какие серверы мы вообще
   подключаем, как их запускать, чем они являются (сенсор / тул / action),
   откуда взяты (официальные / community) — нигде не зафиксировано.
2. **Два разных «реестра» смешиваются.** `AffordanceMap` — это *runtime-вид*
   («что доступно прямо сейчас», из `tools/list`), а нужен ещё *каталог*
   («что мы подключили»). Первый — следствие второго, не наоборот.
3. **Качество источников разное.** Экосистема MCP созрела: есть официальные
   reference-серверы (steering group), серверы в официальном реестре, и
   сырые community-проекты на веб-скрейпинге. Без явной пометки provenance
   легко подключить нестабильное/безлицензионное.
4. **Декларативность.** Подключение сервера не должно требовать правки кода
   loop: должно быть конфигом (по аналогии с пресетами, ADR-0010 §6).
5. **MCP-транспорт.** Нужен реальный клиент (`initialize` → `tools/list` →
   `tools/call`) поверх stdio/http, заменяющий mock-эффектор.

Ограничения:
- CONSTITUTION §1.1: зависимости только через uv; `mcp>=2.0.0` уже в
  `pyproject.toml` — новых зависимостей не вводим.
- CONSTITUTION §2.4: модуль тестируем в изоляции, без глобального состояния.
- ADR-0004: Functional Core / Imperative Shell.
- ADR-0005 §9: необратимые действия — только через capability gate (HITL).
- ADR-0009/0010: пресеты — гибрид Python + TOML (`tomllib`, stdlib).

## Decision

### 1. Реестр интеграций — отдельный каталог, не runtime-карта

Вводится **`IntegrationRegistry`** (каталог подключений) — отдельный модуль
`src/integrations/`. Он **не заменяет** `AffordanceMap`: каталог описывает
«что подключено», runtime-карта рождается из него (`tools/list`). Разведение
явное, во избежание путаницы с `SignalRegistry` (агрегация сигналов).

### 2. Одна запись = один орган; три вида (kind)

```python
class IntegrationKind(Enum):
    SENSOR = "sensor"   # Resources → подмешивается в u(t)
    TOOL = "tool"       # активное эпистемическое зондирование (gate)
    ACTION = "action"   # исполнители среды (позже; HITL)
```

Вид определяет маршрут: `SENSOR → SignalProvider`, `TOOL → AffordanceMap +
ProbeEffector`, `ACTION → gate (T4/HITL)`.

### 3. Транспорты: local / stdio / http

```python
StdioTransport(command, args, env)   # npx/uvx-серверы
HttpTransport(url)                   # сетевые серверы
LocalTransport(provider)             # встроенные (datetime/psutil)
```

`local` покрывает существующие провайдеры (`sources.py`), `stdio`/`http` —
MCP-серверы.

### 4. Хранение: Python-дефолты + TOML-override (fail-fast)

Единообразно с S7-пресетами (ADR-0010 §6): **база** — типобезопасные
`default_integrations()` в Python; **override** — `configs/integrations.toml`
(stdlib `tomllib`), с валидацией. Неизвестный ключ/недопустимое значение →
`ValueError` до прогона. Реестр — декларативный конфиг, loop не меняется.

### 5. Provenance — честность источника

```python
class Provenance(Enum):
    LOCAL = "local"        # встроенный провайдер
    OFFICIAL = "official"  # reference-сервер / официальный реестр MCP
    COMMUNITY = "community"  # сторонний проект
```

Поле `provenance` фиксирует качество источника: официальные серверы —
приоритет; community (особенно скрейпинг) — осознанный риск. Статус
подключения (`disconnected/ready/error`) — **runtime**, в TOML не хранится.

### 6. Мост реестр → runtime

```python
def to_provider(spec) -> SignalProvider: ...      # SENSOR
def to_affordances(specs) -> AffordanceMap: ...   # TOOL
```

Реестр — единственный источник, из которого собираются провайдеры шины и
карта аффордансов. Mock (`_mock_probe`) заменяется реальным `probe_fn`.

### 7. MCP-транспорт — клиент в `src/mcp/client.py`

Синхронная обёртка над async `mcp.ClientSession` (stdio первым, http —
позже). Интерфейс: `list_tools()`, `call_tool(name, args)`. Устойчивость:
таймауты, переподключение, изоляция сбоев (эффектор уже не роняет тик).
Async↔sync мост — открытый риск (см. Open Questions).

### 8. Начальный ростер (порядок подключения)

| Приоритет | Сервер | Транспорт | Provenance | Роль |
|---|---|---|---|---|
| 1 (отладка) | `server-everything` | stdio `npx` | official | Resources+Tools+Prompts; валидация обеих веток |
| 1 (отладка) | `mcp-server-time` | stdio `uvx` | official | детерминированный tool → unit-тесты |
| 2 (реальный мир) | `weather-mcp` | stdio `npx` | official | экстероцепция без ключей (17 tools) |
| 3 (позже, риск) | `web-search-mcp` | stdio `npx` | community | веб-поиск; скрейпинг + браузеры — не первый орган |

Реальные локальные сенсоры (`datetime`, `psutil`) — как `LocalTransport`,
до/параллельно с MCP.

### 9. Гардрейлы сохраняются

Capability gate не трогаем: `SENSOR`/`TOOL` обратимы (T3), `ACTION`
необратим (T4/HITL), fail-safe deny. Сбой транспорта → `ProbeResult(success=
False)`, тик не падает.

## Consequences

### Положительные
- Единый источник истины: что подключено, как, чем является, откуда.
- Декларативность: новый сервер — запись в TOML, без правки loop.
- Честность: provenance отделяет проверенное от рискованного.
- Разведение двух реестров снимает путаницу runtime/catalog.
- Переиспользование: транспорт и gate уже спроектированы (S6 проход 2).

### Отрицательные
- Ещё один модуль и слой конфигурации.
- Async MCP SDK в синхронном loop — нужен мост (риск).
- Внешние серверы недетерминированы → отдельные прогоны, не в отпечатках.

### Tradeoffs
- **Отдельный модуль vs расширение `src/mcp/`:** отдельный чище (FC/IS,
  тестируемость), но больше файлов; mcp остаётся про протокол/транспорт.
- **TOML vs Python:** гибрид (как в ADR-0010 §6) — типобезопасная база +
  гибкий override.
- **stdio vs http:** stdio проще и локальнее (запуск процесса); http — для
  удалённых серверов, позже.

## Confidence
Medium-High. Разведение каталога и runtime-карты, гибрид Python+TOML и
порядок ростера следуют из практики MCP и решений ADR-0010. Детали
async↔sync моста и формат TOML-схемы уточняются при реализации.

## References
- [src/mcp/SPEC.md](../src/mcp/SPEC.md) (карта аффордансов, S6 проход 2)
- [src/host/SPEC.md](../src/host/SPEC.md) (`ProbeEffector`, gate)
- [stages/S6_PLAN.md](../stages/S6_PLAN.md) §Шаг 12, [S6_SPEC.md](../stages/S6_SPEC.md)
- ADR-0004 (FC/IS), ADR-0005 §9 (fail-safe deny), ADR-0010 §6 (Python+TOML)
- BACKLOG `[Phase2][sensors] Реальные интеграции`, `[S6][mcp] MCP-зондирование`
