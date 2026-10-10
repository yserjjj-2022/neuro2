"""Integration registry — Imperative Shell owning the catalog (ADR-0011 §1).

The registry holds declarative ``IntegrationSpec`` records; it does NOT own
runtime artifacts (no processes, no ``SignalSource``). Runtime views are built
from it by ``bridges.py``.

``default_integrations()`` is the Python base of the roster; TOML overrides are
applied on top by ``toml_loader.py`` (ADR-0011 §4).
"""

from __future__ import annotations

import logging

from src.integrations.models import (
    IntegrationKind,
    IntegrationSpec,
    LocalTransport,
    Provenance,
    StdioTransport,
)
from src.mcp.models import SignalCategory

logger = logging.getLogger(__name__)


class IntegrationRegistry:
    """Каталог подключений хоста (ADR-0011 §1).

    Attributes:
        _specs: Записи в порядке добавления (имена уникальны).
    """

    def __init__(self, specs: tuple[IntegrationSpec, ...] = ()) -> None:
        """Создать реестр из записей.

        Args:
            specs: Начальные записи (имена должны быть уникальны).

        Raises:
            ValueError: Если имена дублируются.
        """
        self._specs: list[IntegrationSpec] = []
        for spec in specs:
            self.add(spec)

    def add(self, spec: IntegrationSpec) -> None:
        """Добавить запись в каталог.

        Args:
            spec: Запись интеграции.

        Raises:
            ValueError: Если имя уже зарегистрировано.
        """
        if self.find(spec.name) is not None:
            raise ValueError(f"duplicate integration name: {spec.name!r}")
        self._specs.append(spec)
        logger.debug("integration registered: %s (%s)", spec.name, spec.kind.value)

    def find(self, name: str) -> IntegrationSpec | None:
        """Найти запись по имени (None, если нет)."""
        for spec in self._specs:
            if spec.name == name:
                return spec
        return None

    @property
    def specs(self) -> tuple[IntegrationSpec, ...]:
        """Все записи в порядке добавления."""
        return tuple(self._specs)

    def by_kind(self, kind: IntegrationKind) -> tuple[IntegrationSpec, ...]:
        """Записи указанного вида (sensor/tool/action)."""
        return tuple(s for s in self._specs if s.kind is kind)

    def by_category(
        self, category: SignalCategory
    ) -> tuple[IntegrationSpec, ...]:
        """Записи указанной категории сигнала."""
        return tuple(s for s in self._specs if s.category is category)

    def enabled(self) -> tuple[IntegrationSpec, ...]:
        """Только включённые записи."""
        return tuple(s for s in self._specs if s.enabled)

    def __len__(self) -> int:
        """Число записей в каталоге."""
        return len(self._specs)


def default_integrations() -> tuple[IntegrationSpec, ...]:
    """База ростера (ADR-0011 §8): official-first, community — выключен.

    Порядок SENSOR-записей задаёт укладку каналов на шину ``u(t)`` и должен
    совпадать с прежним ``default_providers()``:
    ``circadian, battery, cpu, message, resources`` — иначе сдвинутся offset
    сегментов и поедут отпечатки. TOOL-записи идут после SENSOR.

    Returns:
        Кортеж записей: локальные сенсоры, official MCP (everything/time/
        weather), community web-search (disabled).
    """
    return (
        IntegrationSpec(
            name="circadian",
            kind=IntegrationKind.SENSOR,
            transport=LocalTransport("circadian"),
            category=SignalCategory.EXTEROCEPTIVE,
            provenance=Provenance.LOCAL,
        ),
        IntegrationSpec(
            name="battery",
            kind=IntegrationKind.SENSOR,
            transport=LocalTransport("battery"),
            category=SignalCategory.INTEROCEPTIVE,
            provenance=Provenance.LOCAL,
        ),
        IntegrationSpec(
            name="cpu",
            kind=IntegrationKind.SENSOR,
            transport=LocalTransport("cpu"),
            category=SignalCategory.INTEROCEPTIVE,
            provenance=Provenance.LOCAL,
        ),
        IntegrationSpec(
            name="message",
            kind=IntegrationKind.SENSOR,
            transport=LocalTransport("message"),
            category=SignalCategory.COMMUNICATIVE,
            provenance=Provenance.LOCAL,
        ),
        IntegrationSpec(
            name="resources",
            kind=IntegrationKind.SENSOR,
            transport=LocalTransport("resources"),
            category=SignalCategory.INTEROCEPTIVE,
            provenance=Provenance.LOCAL,
        ),
        IntegrationSpec(
            name="everything",
            kind=IntegrationKind.TOOL,
            transport=StdioTransport(
                "npx", ("-y", "@modelcontextprotocol/server-everything")
            ),
            category=SignalCategory.EXTEROCEPTIVE,
            provenance=Provenance.OFFICIAL,
        ),
        IntegrationSpec(
            name="time",
            kind=IntegrationKind.TOOL,
            transport=StdioTransport("uvx", ("mcp-server-time",)),
            category=SignalCategory.EXTEROCEPTIVE,
            provenance=Provenance.OFFICIAL,
            tools=("get_current_time", "convert_time"),
            tool_args=(
                ("get_current_time", (("timezone", "UTC"),)),
            ),
        ),
        IntegrationSpec(
            name="weather",
            kind=IntegrationKind.TOOL,
            transport=StdioTransport(
                "npx", ("-y", "@dangahagan/weather-mcp@latest")
            ),
            category=SignalCategory.EXTEROCEPTIVE,
            provenance=Provenance.OFFICIAL,
        ),
        IntegrationSpec(
            name="web-search",
            kind=IntegrationKind.TOOL,
            transport=StdioTransport(
                "npx", ("-y", "web-search-mcp")
            ),
            category=SignalCategory.EXTEROCEPTIVE,
            provenance=Provenance.COMMUNITY,
            enabled=False,
        ),
    )
