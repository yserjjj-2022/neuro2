"""Domain objects for the integration registry (ADR-0011).

The registry is a *catalog* of the host's organs (sensors / tools / actions):
what is connected, how to launch it, what it is and where it came from. It is
deliberately separate from the runtime artifacts it feeds (``AffordanceMap``,
``SignalProvider``) — the catalog is the source of truth, the runtime view is
derived (ADR-0011 §1).

Functional Core / Imperative Shell (ADR-0004): everything here is pure data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.mcp.models import SignalCategory

# Аргументы тула по умолчанию: (имя тула, ((ключ, значение), ...)).
ToolArgs = tuple[tuple[str, tuple[tuple[str, str], ...]], ...]


class IntegrationKind(Enum):
    """Вид интеграции — определяет маршрут в runtime (ADR-0011 §2).

    Attributes:
        SENSOR: MCP Resources → подмешивается в u(t) на медленном такте.
        TOOL: активное эпистемическое зондирование (через capability gate).
        ACTION: исполнители среды (позже; необратимо, HITL).
    """

    SENSOR = "sensor"
    TOOL = "tool"
    ACTION = "action"


class Provenance(Enum):
    """Происхождение интеграции — честность источника (ADR-0011 §5).

    Attributes:
        LOCAL: встроенный провайдер (datetime, psutil-уровень).
        OFFICIAL: reference-сервер / официальный реестр MCP.
        COMMUNITY: сторонний проект (осознанный риск).
    """

    LOCAL = "local"
    OFFICIAL = "official"
    COMMUNITY = "community"


@dataclass(frozen=True)
class StdioTransport:
    """Запуск MCP-сервера как процесса и общение через stdin/stdout.

    Attributes:
        command: Исполняемый файл ("npx", "uvx", "node").
        args: Аргументы команды.
        env: Дополнительные переменные окружения (пары key/value).
    """

    command: str
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("stdio command must not be empty")


@dataclass(frozen=True)
class HttpTransport:
    """Сетевой MCP-сервер (Streamable HTTP).

    Attributes:
        url: Адрес сервера.
    """

    url: str

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("http url must not be empty")


@dataclass(frozen=True)
class LocalTransport:
    """Встроенный провайдер (без внешнего процесса).

    Attributes:
        provider: Имя встроенного провайдера ("circadian", "resources", ...).
    """

    provider: str

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("local provider name must not be empty")


# Любой поддерживаемый транспорт (дискриминируется по типу).
Transport = StdioTransport | HttpTransport | LocalTransport


@dataclass(frozen=True)
class IntegrationSpec:
    """Одна запись каталога — декларация органа (ADR-0011 §2–5).

    Attributes:
        name: Уникальное имя интеграции ("weather", "time", ...).
        kind: Вид (sensor/tool/action) — задаёт маршрут.
        transport: Как подключаться.
        category: Категория сигнала (extero/intero/communicative).
        provenance: Происхождение (local/official/community).
        reversible: Обратимо ли действие (для TOOL/ACTION → tier gate).
        period: Период обновления сенсора в тиках (1 = каждый тик).
        enabled: Включена ли интеграция.
        tools: Ожидаемые имена тулов (пусто → из tools/list).
        tool_args: Аргументы тулов по умолчанию (для тулов, требующих вход).
        rank: Видовой приор важности канала (``rank₀``, BACKLOG). Скаляр > 0
            применяется к сегменту провайдера как ``rank/dim``. None →
            важность не объявлена (legacy: F = 0.5·Σγ·e²). Ключ сопоставления
            — ``tag`` провайдера (см. ``factories.build_providers``).
    """

    name: str
    kind: IntegrationKind
    transport: Transport
    category: SignalCategory = SignalCategory.EXTEROCEPTIVE
    provenance: Provenance = Provenance.COMMUNITY
    reversible: bool = True
    period: int = 1
    enabled: bool = True
    tools: tuple[str, ...] = ()
    tool_args: ToolArgs = ()
    rank: float | None = None

    def args_for(self, tool: str) -> dict[str, str]:
        """Аргументы вызова тула по умолчанию (пусто, если не заданы)."""
        for name, pairs in self.tool_args:
            if name == tool:
                return dict(pairs)
        return {}

    def __post_init__(self) -> None:
        """Валидация инвариантов записи (fail-fast).

        Raises:
            ValueError: Если name пуст, period < 1, или LocalTransport
                используется не для SENSOR.
        """
        if not self.name:
            raise ValueError("integration name must not be empty")
        if self.period < 1:
            raise ValueError(
                f"integration period must be >= 1, got {self.period}"
            )
        if self.rank is not None and self.rank <= 0.0:
            raise ValueError(
                f"integration rank must be > 0, got {self.rank}"
            )
        if isinstance(self.transport, LocalTransport) and (
            self.kind is not IntegrationKind.SENSOR
        ):
            raise ValueError(
                f"LocalTransport only allowed for SENSOR, got {self.kind.value}"
            )

    @property
    def is_mcp(self) -> bool:
        """Использует ли интеграция внешний MCP-транспорт (stdio/http)."""
        return isinstance(self.transport, (StdioTransport, HttpTransport))
