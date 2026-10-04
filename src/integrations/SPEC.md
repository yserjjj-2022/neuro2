# SPEC.md — src/integrations

## Назначение

Реестр интеграций — **каталог** внешних и встроенных органов хоста (сенсоры,
активные тулы, действия). Источник истины «что подключено, как, чем является,
откуда взято». Runtime-артефакты (`SignalProvider`, `AffordanceMap`,
`ProbeEffector`) **рождаются из** реестра, а не хранятся в нём. Решения —
ADR-0011.

Разведение реестров:

| Реестр | Роль | Когда |
|---|---|---|
| `IntegrationRegistry` (этот модуль) | каталог подключений (конфиг) | статично, из Python+TOML |
| `AffordanceMap` (`src/mcp/probe.py`) | runtime-вид «что доступно сейчас» | из `tools/list` |
| `SignalRegistry` (`src/mcp/registry.py`) | агрегация сигналов в `u(t)` | каждый тик |

## Область (входит)

1. Модель записи интеграции (`IntegrationSpec`) + вид (`IntegrationKind`) +
   происхождение (`Provenance`) + транспорт (`Transport`).
2. `IntegrationRegistry`: загрузка базы + TOML-override, поиск, фильтрация.
3. Мосты в runtime: `to_provider` (SENSOR), `to_affordances` (TOOL).
4. MCP-транспорт (`src/mcp/client.py`): синхронный клиент над `mcp.ClientSession`
   (stdio первым), `list_tools()` / `call_tool()`.
5. Начальный ростер: `everything`, `time` (official), `weather` (official),
   `web-search` (community, позже).

## Явно НЕ входит

- Реализация MCP-серверов (берём готовые).
- `ACTION`-вид (исполнители среды) — только тип, без реализации.
- HTTP-транспорт — заложен типом, реализация позже.
- Подключение к сети в unit-тестах — fake-сервер / fake-клиент.
- Замена гомеостаза/гомеостатических порогов.

## Публичный интерфейс

### Виды и происхождение

```python
class IntegrationKind(Enum):
    SENSOR = "sensor"   # Resources → u(t)
    TOOL = "tool"       # эпистемическое зондирование (gate)
    ACTION = "action"   # исполнители (позже, HITL)


class Provenance(Enum):
    LOCAL = "local"          # встроенный провайдер
    OFFICIAL = "official"    # reference / официальный реестр MCP
    COMMUNITY = "community"  # сторонний проект
```

### Транспорт

```python
@dataclass(frozen=True)
class StdioTransport:
    command: str
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()

@dataclass(frozen=True)
class HttpTransport:
    url: str

@dataclass(frozen=True)
class LocalTransport:
    provider: str  # имя встроенного провайдера ("circadian", "resources", ...)

Transport = StdioTransport | HttpTransport | LocalTransport
```

### Запись интеграции

```python
@dataclass(frozen=True)
class IntegrationSpec:
    name: str
    kind: IntegrationKind
    transport: Transport
    category: SignalCategory = SignalCategory.EXTEROCEPTIVE
    provenance: Provenance = Provenance.COMMUNITY
    reversible: bool = True       # для TOOL/ACTION → tier gate
    period: int = 1               # для SENSOR → медленный такт
    enabled: bool = True
    tools: tuple[str, ...] = ()   # ожидаемые тулы (пусто → из tools/list)
    tool_args: ToolArgs = ()      # аргументы тулов по умолчанию
```

Инварианты: `name` непустой; `dim`/`period` корректны; `LocalTransport` только
для `SENSOR`; `provenance=OFFICIAL` не выставляется для `HttpTransport` без
явного override.

### Реестр (Shell)

```python
class IntegrationRegistry:
    def __init__(self, specs: tuple[IntegrationSpec, ...] = ()) -> None: ...
    def add(self, spec: IntegrationSpec) -> None: ...        # ValueError при дубле
    def find(self, name: str) -> IntegrationSpec | None: ...
    @property
    def specs(self) -> tuple[IntegrationSpec, ...]: ...
    def by_kind(self, kind: IntegrationKind) -> tuple[IntegrationSpec, ...]: ...
    def by_category(self, category: SignalCategory) -> tuple[IntegrationSpec, ...]: ...
    def enabled(self) -> tuple[IntegrationSpec, ...]: ...


def default_integrations() -> tuple[IntegrationSpec, ...]: ...
def load_integrations(override: Path | None = None) -> IntegrationRegistry: ...
```

`load_integrations` — база `default_integrations()` + TOML-override (stdlib
`tomllib`), fail-fast: неизвестный ключ/вид/транспорт → `ValueError`.

### Мосты в runtime (Core)

```python
def to_provider(spec: IntegrationSpec) -> SignalProvider: ...
def to_affordances(specs: Sequence[IntegrationSpec]) -> AffordanceMap: ...
```

`to_provider` — только для `SENSOR`; `LocalTransport` → встроенный провайдер,
MCP-сенсоры → провайдер-обёртка над клиентом (позже). `to_affordances` —
только `TOOL`, строит карту из ожидаемых/полученных тулов.

### MCP-транспорт (Shell, `src/mcp/client.py`)

```python
class MCPClient:
    def __init__(self, transport: StdioTransport | HttpTransport) -> None: ...
    def list_tools(self) -> tuple[ToolInfo, ...]: ...
    def call_tool(self, name: str, arguments: dict[str, object]) -> ToolResult: ...
    def close(self) -> None: ...
```

- Синхронная обёртка над async `mcp.ClientSession` (stdio первым).
- `ToolInfo`: `{name, description, category?, reversible?}` — категория и
  обратимость выводятся из метаданных/эвристик (открыто, см. Open Questions).
- Сбои → типизированное исключение/`None`, не роняют вызывающего.

Мост async↔sync: один воркер-поток с event loop; соединение и сессия
открываются/закрываются в **одной** корутине (требование anyio cancel scopes).

### Runtime-glue (Shell, `src/integrations/runtime.py`)

```python
class ProbeTransport:                       # ProbeFn поверх MCP-клиентов
    def __call__(self, affordance: Affordance) -> tuple[float, ...]: ...
    def affordances(self) -> AffordanceMap: ...

def connect_probe_transport(
    registry: IntegrationRegistry,
) -> tuple[ProbeTransport | None, tuple[MCPClient, ...]]: ...

def hash_text_to_vector(text: str, dim: int = 4) -> tuple[float, ...]: ...
```

Текстовый вывод тула хешируется (SHA-256) в вектор фиксированной размерности
(шина говорит числами). Сбой отдельного сервера не роняет остальные.

## Инварианты

1. **Каталог ≠ runtime:** `IntegrationRegistry` не хранит `SignalSource`,
   `Affordance`, подключённые процессы — только декларации.
2. **FC/IS:** модели и мосты (`to_provider`, `to_affordances`) — чистые;
   загрузка TOML и клиент — Shell.
3. **Fail-fast override:** неизвестный ключ/значение → `ValueError` до прогона.
4. **Единый источник:** провайдеры шины и карта аффордансов собираются из
   реестра; хардкод `default_affordances()` уходит (или становится fallback).
5. **Гардрейлы:** `TOOL`/`SENSOR` обратимы (T3); `ACTION` — T4/HITL; сбой
   транспорта → `success=False`, тик не падает.
6. **Без новых зависимостей:** `mcp` уже есть; TOML — stdlib.
7. **Обратная совместимость:** пустой реестр / `mcp.enabled=False` → контур S6
   идентичен.

## Критерии приёмки

- [ ] `IntegrationSpec`/`IntegrationKind`/`Provenance`/транспорты — frozen, с валидацией
- [ ] `IntegrationRegistry`: add/find/by_kind/by_category/enabled, дубль → ValueError
- [ ] `load_integrations`: база + TOML-override, fail-fast
- [ ] `to_provider`/`to_affordances` — чистые, покрыты unit-тестами
- [ ] `MCPClient.list_tools`/`call_tool` — на stdio fake-сервере
- [x] `ProbeEffector` может получать реальный `probe_fn` из клиента
- [x] ростер: everything/time/weather зарегистрированы, web-search отмечен community
- [x] без реестра/флага контур S6 идентичен
- [x] ruff/тесты зелёные; mypy strict для новых Core-модулей

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| Каталог vs runtime-карта | Решено | раздельные (ADR-0011 §1) |
| Хранение | Решено | Python-база + TOML-override (ADR-0011 §4) |
| Provenance | Решено | поле `Provenance` (ADR-0011 §5) |
| Порядок ростера | Решено | official-first (ADR-0011 §8) |
| Async↔sync мост | Решено | воркер-поток + одна корутина владеет соединением |
| Категория/обратимость тулов | Открыто | метаданные сервера vs эвристика по имени |
| Формат TOML-схемы | Решено | плоские записи `[[integrations]]` + `tool_args` |
| Аргументы тулов | Решено | `tool_args` в записи (дефолты для требующих вход) |
| Реализация HTTP | Отложено | после stdio |
