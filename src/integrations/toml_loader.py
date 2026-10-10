"""TOML override loader for the integration registry (ADR-0011 §4).

Base roster is ``default_integrations()`` (Python); a TOML file may override
or extend it. Fail-fast: unknown keys, kinds, transports or invalid values
raise ``ValueError`` before any run. Uses stdlib ``tomllib`` — no new deps.

Schema (``configs/integrations.toml``)::

    [[integrations]]
    name = "weather"
    kind = "tool"          # sensor | tool | action
    transport = "stdio"    # stdio | http | local
    command = "npx"
    args = ["-y", "@dangahagan/weather-mcp@latest"]
    category = "exteroceptive"   # exteroceptive | interoceptive | communicative
    provenance = "official"      # local | official | community
    reversible = true
    period = 1
    enabled = true
    tools = []
    tool_args = { get_current_time = { timezone = "UTC" } }  # аргументы тулов
    rank = 3.0                   # видовой приор важности канала (> 0)

A record matching an existing base name replaces it; a new name appends.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

from src.integrations.models import (
    HttpTransport,
    IntegrationKind,
    IntegrationSpec,
    LocalTransport,
    Provenance,
    StdioTransport,
)
from src.integrations.registry import IntegrationRegistry, default_integrations
from src.mcp.models import SignalCategory

logger = logging.getLogger(__name__)

_ALLOWED_KEYS = {
    "name",
    "kind",
    "transport",
    "command",
    "args",
    "env",
    "url",
    "provider",
    "category",
    "provenance",
    "reversible",
    "period",
    "enabled",
    "tools",
    "tool_args",
    "rank",
}


def _build_transport(
    raw: dict[str, Any], name: str
) -> StdioTransport | HttpTransport | LocalTransport:
    """Собрать транспорт из TOML-полей (fail-fast)."""
    kind = raw.get("transport")
    if kind == "stdio":
        if "command" not in raw:
            raise ValueError(f"integration {name!r}: stdio requires 'command'")
        env = tuple((str(k), str(v)) for k, v in dict(raw.get("env", {})).items())
        return StdioTransport(
            str(raw["command"]),
            tuple(str(a) for a in raw.get("args", ())),
            env,
        )
    if kind == "http":
        if "url" not in raw:
            raise ValueError(f"integration {name!r}: http requires 'url'")
        return HttpTransport(str(raw["url"]))
    if kind == "local":
        if "provider" not in raw:
            raise ValueError(f"integration {name!r}: local requires 'provider'")
        return LocalTransport(str(raw["provider"]))
    raise ValueError(f"integration {name!r}: unknown transport {kind!r}")


def _spec_from_toml(
    raw: dict[str, Any], base: IntegrationSpec | None = None
) -> IntegrationSpec:
    """Преобразовать одну TOML-запись в IntegrationSpec (fail-fast).

    Если ``base`` задан (запись с таким именем уже есть в базе), отсутствующие
    поля берутся из базы — это позволяет частичный override (например, только
    ``enabled = false``). Транспорт заменяется целиком, если задан.

    Args:
        raw: TOML-запись.
        base: Существующая запись для слияния (None → новая запись).

    Returns:
        Готовая запись.

    Raises:
        ValueError: При неизвестном ключе/значении или некорректном транспорте.
    """
    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"unknown integration keys: {sorted(unknown)}")
    name = raw.get("name")
    if not name:
        raise ValueError("integration record requires 'name'")
    name = str(name)
    try:
        kind = IntegrationKind(
            str(raw.get("kind", base.kind.value if base else "tool"))
        )
        category = SignalCategory(
            str(raw.get("category", base.category.value if base else "exteroceptive"))
        )
        provenance = Provenance(
            str(raw.get("provenance", base.provenance.value if base else "community"))
        )
    except ValueError as exc:
        raise ValueError(f"integration {name!r}: {exc}") from exc

    if "transport" in raw or base is None:
        transport = _build_transport(raw, name)
    else:
        transport = base.transport

    return IntegrationSpec(
        name=name,
        kind=kind,
        transport=transport,
        category=category,
        provenance=provenance,
        reversible=bool(raw.get("reversible", base.reversible if base else True)),
        period=int(raw.get("period", base.period if base else 1)),
        enabled=bool(raw.get("enabled", base.enabled if base else True)),
        tools=tuple(str(t) for t in raw.get("tools", base.tools if base else ())),
        tool_args=_tool_args_from_toml(raw, base),
        rank=_rank_from_toml(raw, base),
    )


def _rank_from_toml(raw: dict[str, Any], base: IntegrationSpec | None) -> float | None:
    """Собрать ``rank₀`` из TOML (fail-fast по типу/знаку).

    Формат: ``rank = 3.0`` (скаляр > 0). Отсутствует → наследуется из базы
    (или None). Явный ``rank`` включает веса каналов (BACKLOG).

    Args:
        raw: TOML-запись.
        base: Базовая запись для наследования (None → без наследования).

    Returns:
        Ранг канала (> 0) или None (не объявлен).

    Raises:
        TypeError: Если ``rank`` не число.
        ValueError: Если ``rank`` <= 0.
    """
    if "rank" not in raw:
        return base.rank if base is not None else None
    value = raw["rank"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"'rank' must be a number, got {type(value).__name__}")
    rank = float(value)
    if rank <= 0.0:
        raise ValueError(f"'rank' must be > 0, got {rank}")
    return rank


def _tool_args_from_toml(
    raw: dict[str, Any], base: IntegrationSpec | None
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Собрать аргументы тулов из TOML (fail-fast по типам).

    Формат: ``tool_args = { tool_name = { key = "value" } }``.

    Args:
        raw: TOML-запись.
        base: Базовая запись для наследования (None → без наследования).

    Returns:
        Кортеж ``(имя тула, ((ключ, значение), ...))``.

    Raises:
        ValueError: Если ``tool_args`` не таблица таблиц.
    """
    if "tool_args" in raw:
        table = raw["tool_args"]
    elif base is not None:
        return base.tool_args
    else:
        return ()
    if not isinstance(table, dict):
        raise TypeError("'tool_args' must be a table of tables")
    result: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for tool, args in table.items():
        if not isinstance(args, dict):
            raise TypeError(f"'tool_args.{tool}' must be a table")
        result.append((str(tool), tuple((str(k), str(v)) for k, v in args.items())))
    return tuple(result)


def load_integrations(override: Path | None = None) -> IntegrationRegistry:
    """Загрузить реестр: база + TOML-override (ADR-0011 §4).

    Args:
        override: Путь к TOML-файлу; None → только база.

    Returns:
        IntegrationRegistry с применённым override.

    Raises:
        FileNotFoundError: Если override-файл не существует.
        ValueError: При неизвестном ключе/виде/транспорте или неверном
            значении (fail-fast до прогона).
    """
    base = {s.name: s for s in default_integrations()}
    if override is None:
        return IntegrationRegistry(tuple(base.values()))

    if not override.exists():
        raise FileNotFoundError(f"integrations override not found: {override}")

    with override.open("rb") as handle:
        data = tomllib.load(handle)

    unknown_top = set(data) - {"integrations"}
    if unknown_top:
        raise ValueError(f"unknown top-level keys: {sorted(unknown_top)}")

    records = data.get("integrations", [])
    if not isinstance(records, list):
        raise TypeError("'integrations' must be a list of tables")

    for raw in records:
        existing = base.get(str(raw.get("name", "")))
        spec = _spec_from_toml(raw, existing)
        base[spec.name] = spec  # replace existing or append new
    logger.debug("integrations override applied: %s", override)
    return IntegrationRegistry(tuple(base.values()))
