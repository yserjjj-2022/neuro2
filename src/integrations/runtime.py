"""Runtime glue: catalog + MCP clients → probe transport (ADR-0011 §6, Shell).

Turns real MCP clients into a ``ProbeFn`` the effector can call: an affordance
name maps to a (client, tool) pair; the tool's textual output is hashed into a
fixed-dim float vector (the bus speaks numbers, tools speak text).

Kept separate from ``bridges.py`` (Core) because it does I/O.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping

from src.integrations.models import (
    IntegrationKind,
    StdioTransport,
)
from src.integrations.registry import IntegrationRegistry
from src.mcp.client import MCPClient, MCPClientError
from src.mcp.models import SignalCategory
from src.mcp.probe import Affordance, AffordanceMap

logger = logging.getLogger(__name__)

DEFAULT_PROBE_DIM = 4


def hash_text_to_vector(text: str, dim: int = DEFAULT_PROBE_DIM) -> tuple[float, ...]:
    """Детерминированно отобразить текст в вектор ``[0, 1]`` (чистая).

    Использует SHA-256: одинаковый текст → одинаковый вектор, разный — разный.

    Args:
        text: Текстовый вывод тула.
        dim: Размерность вектора (> 0).

    Returns:
        Кортеж длины ``dim`` со значениями в ``[0, 1]``.

    Raises:
        ValueError: Если dim <= 0.
    """
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Раскладываем байты digest в dim равномерных значений [0, 1].
    values: list[float] = []
    for i in range(dim):
        byte = digest[i % len(digest)]
        values.append(byte / 255.0)
    return tuple(values)


class ProbeTransport:
    """ProbeFn поверх реальных MCP-клиентов (Shell).

    Attributes:
        routes: Отображение имя аффорданса → (клиент, имя тула).
        dim: Размерность возвращаемого вектора.
    """

    def __init__(
        self,
        routes: Mapping[str, tuple[MCPClient, str, dict[str, str]]],
        *,
        dim: int = DEFAULT_PROBE_DIM,
    ) -> None:
        self.routes = dict(routes)
        self.dim = dim

    def __call__(self, affordance: Affordance) -> tuple[float, ...]:
        """Вызвать тул и вернуть хеш его вывода (никогда не бросает наружу).

        Args:
            affordance: Аффорданс для зондирования.

        Returns:
            Вектор ``dim``; при отсутствии маршрута/сбое — пустой кортеж.
        """
        route = self.routes.get(affordance.name)
        if route is None:
            logger.warning("no MCP route for affordance %r", affordance.name)
            return ()
        client, tool, arguments = route
        try:
            result = client.call_tool(tool, arguments or None)
        except MCPClientError as exc:
            logger.error("probe transport error: %s", exc)
            return ()
        if not result.success:
            return ()
        return hash_text_to_vector(result.text, self.dim)

    def affordances(self) -> AffordanceMap:
        """Карта аффордансов по фактическим тулам подключённых серверов.

        Returns:
            AffordanceMap с именами реальных тулов (обратимые, экстеро).
        """
        return AffordanceMap(
            tuple(
                Affordance(name=name, category=SignalCategory.EXTEROCEPTIVE)
                for name in self.routes
            )
        )


def connect_probe_transport(
    registry: IntegrationRegistry,
) -> tuple[ProbeTransport | None, tuple[MCPClient, ...]]:
    """Подключить реальные MCP-клиенты и собрать транспорт зондирования.

    Проходит по включённым TOOL-интеграциям с stdio-транспортом, подключает
    каждую через ``MCPClient``, читает ``tools/list`` и строит маршруты
    аффорданс → (клиент, тул). Сбой подключения отдельного сервера не роняет
    остальные (изоляция).

    Args:
        registry: Реестр интеграций.

    Returns:
        ``(transport, clients)``: транспорт (None, если ни один сервер не
        подключился) и список открытых клиентов для последующего ``close()``.
    """
    routes: dict[str, tuple[MCPClient, str, dict[str, str]]] = {}
    clients: list[MCPClient] = []
    for spec in registry.enabled():
        if spec.kind is not IntegrationKind.TOOL or not isinstance(
            spec.transport, StdioTransport
        ):
            continue
        client = MCPClient()
        # Наследуем окружение процесса (PATH, UV_*, токены) и накладываем
        # явные переменные интеграции сверху.
        env = dict(os.environ)
        env.update(spec.transport.env)
        try:
            client.connect_stdio(
                spec.transport.command,
                spec.transport.args,
                env,
            )
            tools = client.list_tools()
        except MCPClientError as exc:
            logger.error("integration %r connect failed: %s", spec.name, exc)
            client.close()
            continue
        clients.append(client)
        for tool in tools:
            if tool.name in routes:
                logger.warning("duplicate tool %r; keeping first", tool.name)
                continue
            routes[tool.name] = (client, tool.name, spec.args_for(tool.name))
    if not routes:
        return None, tuple(clients)
    return ProbeTransport(routes), tuple(clients)


__all__ = [
    "DEFAULT_PROBE_DIM",
    "ProbeTransport",
    "connect_probe_transport",
    "hash_text_to_vector",
]
