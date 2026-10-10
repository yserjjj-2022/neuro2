"""Provider factories: catalog spec + injected deps → ``SignalProvider``.

Shell-side glue that turns declarative SENSOR records into runtime providers.
Kept separate from ``bridges.py`` (pure catalog → affordance mapping) so the
Core bridge never depends on host/memory types.

A factory is a pure function of ``(spec, ctx)``: no I/O, no owned state. Heavy
dependencies (resource meter, message provider with a real embedder) are
injected **pre-built** through ``SensorContext``; the factory only wires them in,
honoring the spec's declaration (category). This keeps ``u(t)`` assembly driven
by the registry while the Shell still owns construction of the dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.host.sources import (
    BatteryProvider,
    CircadianProvider,
    CpuProvider,
    SignalProvider,
    UserMessageProvider,
)
from src.integrations.models import (
    IntegrationKind,
    IntegrationSpec,
    LocalTransport,
)

__all__ = [
    "SensorContext",
    "build_provider",
    "build_providers",
    "enabled_sensors",
]


@dataclass(frozen=True)
class SensorContext:
    """Injected dependencies for building SENSOR providers (Shell).

    Attributes:
        seed: Зерно детерминированных провайдеров (cpu, message-заглушка).
        message_dim: Размерность заглушки сообщения (если провайдер не передан).
        resource_provider: Готовый ресурсный провайдер (инъекция meter) или
            None (канал ``resources`` не собирается — как в ``default_providers``).
        message_provider: Готовый коммуникативный провайдер (эмбеддер) или
            None → заглушка ``UserMessageProvider``.
    """

    seed: int = 0
    message_dim: int = 8
    resource_provider: SignalProvider | None = None
    message_provider: SignalProvider | None = None


def _build_local(spec: IntegrationSpec, ctx: SensorContext) -> SignalProvider:
    """Собрать встроенный провайдер по имени ``LocalTransport.provider``.

    Args:
        spec: SENSOR-запись с ``LocalTransport``.
        ctx: Инъектированные зависимости.

    Returns:
        Готовый провайдер.

    Raises:
        NotImplementedError: Для неизвестного имени или отсутствующей
            зависимости (resources без ``ctx.resource_provider``).
    """
    assert isinstance(spec.transport, LocalTransport)
    name = spec.transport.provider
    if name == "circadian":
        return CircadianProvider(category=spec.category)
    if name == "battery":
        return BatteryProvider(category=spec.category)
    if name == "cpu":
        return CpuProvider(seed=ctx.seed, category=spec.category)
    if name == "message":
        if ctx.message_provider is not None:
            return ctx.message_provider
        return UserMessageProvider(
            embedding_dim=ctx.message_dim,
            seed=ctx.seed,
            category=spec.category,
        )
    if name == "resources":
        if ctx.resource_provider is None:
            raise NotImplementedError(
                "resources provider requires an injected meter "
                "(SensorContext.resource_provider)"
            )
        return ctx.resource_provider
    raise NotImplementedError(
        f"local provider {name!r} not buildable here: {spec.name}"
    )


def build_provider(spec: IntegrationSpec, ctx: SensorContext) -> SignalProvider:
    """Построить сенсорный провайдер из записи каталога (Shell-glue).

    Args:
        spec: Запись интеграции вида SENSOR.
        ctx: Инъектированные зависимости (seed, meter, message provider).

    Returns:
        ``SignalProvider`` для укладки на шину ``u(t)``.

    Raises:
        ValueError: Если ``spec`` не SENSOR.
        NotImplementedError: Для MCP-сенсоров (нужен клиент) или неизвестного
            локального провайдера.
    """
    if spec.kind is not IntegrationKind.SENSOR:
        raise ValueError(
            f"build_provider expects SENSOR, got {spec.kind.value!r} ({spec.name})"
        )
    if isinstance(spec.transport, LocalTransport):
        return _build_local(spec, ctx)
    raise NotImplementedError(f"MCP sensor provider not implemented yet: {spec.name}")


def enabled_sensors(
    specs: Sequence[IntegrationSpec],
) -> tuple[IntegrationSpec, ...]:
    """Включённые SENSOR-записи в порядке объявления (порядок укладки).

    Args:
        specs: Записи каталога.

    Returns:
        Кортеж включённых SENSOR-записей.
    """
    return tuple(s for s in specs if s.kind is IntegrationKind.SENSOR and s.enabled)


def build_providers(
    specs: Sequence[IntegrationSpec], ctx: SensorContext
) -> list[SignalProvider]:
    """Собрать список провайдеров из включённых SENSOR-записей (Shell-glue).

    Порядок провайдеров = порядок записей → порядок сегментов шины.

    Args:
        specs: Записи каталога.
        ctx: Инъектированные зависимости.

    Returns:
        Провайдеры в порядке укладки на шину.

    Raises:
        ValueError: Если список пуст (шина не собирается).
    """
    providers = [build_provider(spec, ctx) for spec in enabled_sensors(specs)]
    if not providers:
        raise ValueError("no enabled SENSOR integrations: bus would be empty")
    return providers
