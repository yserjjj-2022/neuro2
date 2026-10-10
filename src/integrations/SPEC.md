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
3. Мосты в runtime: `to_affordances` (TOOL, Core) + `build_provider`/
   `build_providers` (SENSOR, Shell-glue в `factories.py`).
4. MCP-транспорт (`src/mcp/client.py`): синхронный клиент над `mcp.ClientSession`
   (stdio первым), `list_tools()` / `call_tool()`.
5. Начальный ростер: локальные сенсоры `circadian`/`battery`/`cpu`/`message`/
   `resources` + `everything`/`time`/`weather` (official), `web-search`
   (community, позже).

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
    rank: float | None = None     # rank₀ (> 0) или None (не объявлен)
```

Инварианты: `name` непустой; `dim`/`period` корректны; `rank` > 0 или None;
`LocalTransport` только для `SENSOR`; `provenance=OFFICIAL` не выставляется для
`HttpTransport` без явного override.

`rank` — видовой приор важности канала (BACKLOG): скаляр > 0, применяется к
сегменту провайдера как `rank/dim`. Ключ сопоставления — **`tag` провайдера**
(напр. `message`-запись → провайдер с `tag="user_message"`), а не `name`
записи. None → важность не объявлена (legacy `F = 0.5·Σγ·e²`); веса
включаются только при явном объявлении хотя бы одного ранга.

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

### Мосты в runtime (Core / Shell)

```python
# Core (bridges.py): чистые, без I/O
def to_affordances(specs: Sequence[IntegrationSpec]) -> AffordanceMap: ...

# Shell-glue (factories.py): инъекция зависимостей (meter/embedder)
@dataclass(frozen=True)
class SensorContext:
    seed: int = 0
    message_dim: int = 8
    resource_provider: SignalProvider | None = None
    message_provider: SignalProvider | None = None

def build_provider(spec: IntegrationSpec, ctx: SensorContext) -> SignalProvider: ...
def build_providers(
    specs: Sequence[IntegrationSpec], ctx: SensorContext
) -> list[SignalProvider]: ...
def enabled_sensors(specs: Sequence[IntegrationSpec]) -> tuple[IntegrationSpec, ...]: ...
```

`build_provider` — только для `SENSOR`; `LocalTransport` → встроенный провайдер
(категория берётся из `spec`), MCP-сенсоры → `NotImplementedError` (нужен
клиент). Зависимости (ресурсный meter, message-провайдер с эмбеддером)
инжектятся **готовыми** через `SensorContext`; `message` без инъекции →
заглушка `UserMessageProvider`; `resources` без meter → `NotImplementedError`.
`to_affordances` — только `TOOL`, строит карту из ожидаемых/полученных тулов.

`factories.py` отделён от `bridges.py`, чтобы чистый Core-мост не зависел от
host/memory типов (FC/IS, ADR-0004).

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
2. **FC/IS:** модели и мосты (`to_affordances`) — чистые; загрузка TOML,
   клиент и фабрики провайдеров (`factories.py`) — Shell.
3. **Fail-fast override:** неизвестный ключ/значение → `ValueError` до прогона.
4. **Единый источник:** состав шины `u(t)` и карта аффордансов собираются из
   реестра; `default_providers()` и хардкод `default_affordances()` уходят
   (последний — fallback при `integrations=None`).
5. **Гардрейлы:** `TOOL`/`SENSOR` обратимы (T3); `ACTION` — T4/HITL; сбой
   транспорта → `success=False`, тик не падает.
6. **Без новых зависимостей:** `mcp` уже есть; TOML — stdlib.
7. **Обратная совместимость:** `integrations=None` → `default_integrations()`,
   порядок SENSOR совпадает с прежним `default_providers()` (шина идентична);
   пустые ранги → `F = 0.5·Σγ·e²` (S1–S8).

## Критерии приёмки

- [ ] `IntegrationSpec`/`IntegrationKind`/`Provenance`/транспорты — frozen, с валидацией
- [ ] `IntegrationRegistry`: add/find/by_kind/by_category/enabled, дубль → ValueError
- [ ] `load_integrations`: база + TOML-override, fail-fast
- [x] `build_provider`/`to_affordances` — покрыты unit-тестами (Core/Shell разделены)
- [x] `MCPClient.list_tools`/`call_tool` — на stdio fake-сервере
- [x] `ProbeEffector` может получать реальный `probe_fn` из клиента
- [x] ростер: everything/time/weather зарегистрированы, web-search отмечен community
- [x] без реестра/флага контур S6 идентичен
- [x] ruff/тесты зелёные; mypy strict для новых Core-модулей
- [x] состав шины `u(t)` собирается из реестра (SENSOR-записи), не из `default_providers()`
- [x] подмножество SENSOR в реестре → узкая шина (набор сенсоров на экземпляр)
- [x] `rank₀` в `IntegrationSpec`; веса включаются только при явном объявлении

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| Каталог vs runtime-карта | Решено | раздельные (ADR-0011 §1) |
| Хранение | Решено | Python-база + TOML-override (ADR-0011 §4) |
| Provenance | Решено | поле `Provenance` (ADR-0011 §5) |
| Порядок ростера | Решено | official-first (ADR-0011 §8) |
| Состав шины | Решено | из SENSOR-записей реестра; `default_providers()` — legacy-хелпер тестов |
| Место `rank₀` | Решено | `IntegrationSpec.rank` (`float \| None`); ключ — `tag` провайдера |
| Async↔sync мост | Решено | воркер-поток + одна корутина владеет соединением |
| Категория/обратимость тулов | Открыто | метаданные сервера vs эвристика по имени |
| Формат TOML-схемы | Решено | плоские записи `[[integrations]]` + `tool_args` |
| Аргументы тулов | Решено | `tool_args` в записи (дефолты для требующих вход) |
| Реализация HTTP | Отложено | после stdio |
