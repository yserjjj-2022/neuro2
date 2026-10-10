"""Signal providers and the sensory bus — the host's "sense organs".

Every provider turns one aspect of reality (time of day, battery, CPU load,
incoming text) into a numeric vector and returns it as a ``SignalSource``.
Mock and real providers share the SAME contract, so the host loop never
changes when a mock is swapped for a real integration (datetime/psutil/MCP).

Design notes:
    - ``SignalBus`` owns the providers and assembles the wide bus ``u(t)``
      by concatenation, keeping a segment map ``[(name, offset, dim, period)]``.
      The map is the foundation of width scaling: in Phase 2 columns will read
      only their own slice (``ColumnConfig.reads``) instead of the whole bus.
    - ``period`` (in ticks) supports the "slow tick" from the manifest §3.Ж:
      heavy sources (market, news) refresh once per K ticks, not every tick.
    - Providers are deterministic functions of ``(tick, now)``: same inputs →
      same ``SignalSource``. The loop owns the wall clock and passes it in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from src.mcp import SignalCategory, SignalRegistry, SignalSource
from src.mcp.models import Vector


@runtime_checkable
class SignalProvider(Protocol):
    """Контракт источника сигнала (mock или реальный).

    Реализация обязана быть детерминированной функцией от ``(tick, now)``:
    одинаковые аргументы → одинаковый ``SignalSource``. Никакого скрытого
    состояния между вызовами.

    Attributes:
        tag: Уникальный идентификатор источника ("battery", "weather", ...).
        category: Категория сигнала (extero-/intero-/communicative).
        dim: Размерность вектора данных.
        period: Как часто обновляется сигнал, в тиках (1 = каждый тик).
    """

    @property
    def tag(self) -> str: ...

    @property
    def category(self) -> SignalCategory: ...

    @property
    def dim(self) -> int: ...

    @property
    def period(self) -> int: ...

    def read(self, tick: int, now: float) -> SignalSource:
        """Прочитать сигнал на данном тике.

        Args:
            tick: Номер тика хоста (начинается с 0).
            now: Wall-clock время в секундах (владеет loop, не провайдер).

        Returns:
            SignalSource с категорией, вектором данных и метаданными.
        """
        ...


@dataclass(frozen=True)
class CircadianProvider:
    """Циркадное время суток (экстероцептивный).

    Кодирует фазу суток гармониками ``[sin(2πh/24), cos(2πh/24)]``,
    где ``h = (now / 3600) % 24``. Непрерывное время — манифест §3.Б.

    Attributes:
        tag: Идентификатор источника.
        category: Экстероцептивный — контекст внешнего мира.
        period: Обновляется каждый тик (дёшево, 2 числа).
    """

    tag: str = "circadian"
    category: SignalCategory = SignalCategory.EXTEROCEPTIVE
    period: int = 1

    @property
    def dim(self) -> int:
        """Размерность: sin + cos = 2."""
        return 2

    def read(self, tick: int, now: float) -> SignalSource:
        """Фаза суток как гармоники sin/cos.

        Args:
            tick: Номер тика (не влияет — время берётся из ``now``).
            now: Wall-clock время в секундах.

        Returns:
            SignalSource shape=(2,) с [sin, cos] фазы суток.
        """
        hour = (now / 3600.0) % 24.0
        angle = 2.0 * np.pi * hour / 24.0
        data = np.array([np.sin(angle), np.cos(angle)], dtype=np.float64)
        return SignalSource(category=self.category, data=data, tag=self.tag)


@dataclass(frozen=True)
class BatteryProvider:
    """Заряд батареи (интероцептивный, mock).

    Линейный разряд: ``level = max(0, start_level - drain_per_tick · tick)``.
    ``severity = 1 - level``; при ``level <= 0.1`` severity ≥ 0.9 и
    ``SignalSource`` автоматически помечает сигнал как ``is_reflex``.

    Attributes:
        start_level: Начальный заряд, [0, 1].
        drain_per_tick: Разряд за тик.
        tag: Идентификатор источника.
        category: Интероцептивный — состояние самой системы.
        period: Обновляется каждый тик.
    """

    start_level: float = 1.0
    drain_per_tick: float = 0.001
    tag: str = "battery"
    category: SignalCategory = SignalCategory.INTEROCEPTIVE
    period: int = 1

    @property
    def dim(self) -> int:
        """Размерность: [level] = 1."""
        return 1

    def read(self, tick: int, now: float) -> SignalSource:
        """Уровень заряда и severity на данном тике.

        Args:
            tick: Номер тика (определяет разряд).
            now: Wall-clock время (не используется).

        Returns:
            SignalSource shape=(1,) с [level]; severity = 1 - level.
        """
        level = float(max(0.0, self.start_level - self.drain_per_tick * tick))
        severity = float(min(1.0, 1.0 - level))
        data = np.array([level], dtype=np.float64)
        return SignalSource(
            category=self.category,
            data=data,
            severity=severity,
            tag=self.tag,
        )


@dataclass(frozen=True)
class CpuProvider:
    """Загрузка CPU (интероцептивный, mock).

    Детерминированный случайный блуг: ``load = clip(base + N(0, noise), 0, 1)``
    с генератором, засеянным ``seed + tick``. ``severity = load``.

    Attributes:
        base_load: Базовая загрузка, [0, 1].
        noise: Стандартное отклонение шума.
        seed: Зерно генератора (детерминизм).
        tag: Идентификатор источника.
        category: Интероцептивный.
        period: Обновляется каждый тик.
    """

    base_load: float = 0.3
    noise: float = 0.05
    seed: int = 0
    tag: str = "cpu"
    category: SignalCategory = SignalCategory.INTEROCEPTIVE
    period: int = 1

    @property
    def dim(self) -> int:
        """Размерность: [load] = 1."""
        return 1

    def read(self, tick: int, now: float) -> SignalSource:
        """Загрузка CPU и severity на данном тике.

        Args:
            tick: Номер тика (определяет выборку шума).
            now: Wall-clock время (не используется).

        Returns:
            SignalSource shape=(1,) с [load]; severity = load.
        """
        rng = np.random.default_rng(self.seed + tick)
        load = float(np.clip(self.base_load + rng.normal(0.0, self.noise), 0.0, 1.0))
        data = np.array([load], dtype=np.float64)
        return SignalSource(
            category=self.category,
            data=data,
            severity=load,
            tag=self.tag,
        )


@dataclass(frozen=True)
class UserMessageProvider:
    """Сообщение собеседника (коммуникативный, mock).

    Заглушка: без эмбеддера текст нельзя положить в числовую шину, поэтому
    отдаётся фиксированный детерминированный вектор размерности
    ``embedding_dim``. Маршрут коммуникации существует и тестируется;
    реальный эмбеддер подменит провайдера без смены интерфейса.

    Attributes:
        embedding_dim: Размерность вектора-заглушки.
        seed: Зерно генератора (вектор стабилен между тиками).
        tag: Идентификатор источника.
        category: Коммуникативный.
        period: Обновляется каждый тик.
    """

    embedding_dim: int = 8
    seed: int = 0
    tag: str = "user_message"
    category: SignalCategory = SignalCategory.COMMUNICATIVE
    period: int = 1

    @property
    def dim(self) -> int:
        """Размерность: embedding_dim."""
        return self.embedding_dim

    def read(self, tick: int, now: float) -> SignalSource:
        """Фиксированный вектор-заглушка сообщения.

        Args:
            tick: Номер тика (не используется — сообщение стабильно).
            now: Wall-clock время (не используется).

        Returns:
            SignalSource shape=(embedding_dim,) — детерминированный вектор.
        """
        rng = np.random.default_rng(self.seed)
        data = rng.standard_normal(self.embedding_dim).astype(np.float64)
        return SignalSource(category=self.category, data=data, tag=self.tag)


@dataclass(frozen=True)
class ConstantProvider:
    """Постоянный вход — тестовый генератор для проверки сходимости.

    Стабильный ``u(t)`` → EMA-состояние колонок сходится, ``e(t) → 0``,
    значит ``F(t) → 0`` и ``active_columns → 0`` (эксперимент №1).

    Attributes:
        value: Константный вектор данных.
        tag: Идентификатор источника.
        category: По умолчанию экстероцептивный.
        period: Обновляется каждый тик.
        severity: Severity сигнала (0.0 по умолчанию).
    """

    value: tuple[float, ...]
    tag: str = "constant"
    category: SignalCategory = SignalCategory.EXTEROCEPTIVE
    period: int = 1
    severity: float = 0.0

    @property
    def dim(self) -> int:
        """Размерность: len(value)."""
        return len(self.value)

    def read(self, tick: int, now: float) -> SignalSource:
        """Вернуть константный вектор.

        Args:
            tick: Номер тика (не используется).
            now: Wall-clock время (не используется).

        Returns:
            SignalSource с неизменным вектором данных.
        """
        data = np.array(self.value, dtype=np.float64)
        return SignalSource(
            category=self.category,
            data=data,
            severity=self.severity,
            tag=self.tag,
        )


@dataclass(frozen=True)
class StepProvider:
    """Ступенчатый вход — тестовый генератор для проверки реакции.

    На тике ``step_at`` значение скачком меняется с ``before`` на ``after``.
    Хост должен «вздрогнуть»: ``F(t)`` растёт, ``valence < 0``, затем
    система снова сходится (эксперимент №2).

    Attributes:
        before: Вектор до скачка.
        after: Вектор после скачка.
        step_at: Номер тика, на котором происходит скачок.
        tag: Идентификатор источника.
        category: По умолчанию экстероцептивный.
        period: Обновляется каждый тик.
    """

    before: tuple[float, ...]
    after: tuple[float, ...]
    step_at: int
    tag: str = "step"
    category: SignalCategory = SignalCategory.EXTEROCEPTIVE
    period: int = 1

    def __post_init__(self) -> None:
        if len(self.before) != len(self.after):
            raise ValueError(
                f"before/after dim mismatch: {len(self.before)} != {len(self.after)}"
            )
        if self.step_at < 0:
            raise ValueError(f"step_at must be >= 0, got {self.step_at}")

    @property
    def dim(self) -> int:
        """Размерность: len(before) == len(after)."""
        return len(self.before)

    def read(self, tick: int, now: float) -> SignalSource:
        """Вернуть before до скачка и after после.

        Args:
            tick: Номер тика (определяет сторону скачка).
            now: Wall-clock время (не используется).

        Returns:
            SignalSource с before при tick < step_at, иначе after.
        """
        value = self.after if tick >= self.step_at else self.before
        data = np.array(value, dtype=np.float64)
        return SignalSource(category=self.category, data=data, tag=self.tag)


@dataclass(frozen=True)
class NoisyProvider:
    """Детерминированный белый шум — тестовый генератор для стресса.

    Каждый тик — новая выборка ``N(0, scale)`` из генератора, засеянного
    ``seed + tick``. При стабильном шуме аллостатический стресс накапливается
    (эксперимент №4).

    Attributes:
        dim: Размерность вектора шума.
        scale: Стандартное отклонение шума.
        seed: Зерно генератора (детерминизм).
        tag: Идентификатор источника.
        category: По умолчанию экстероцептивный.
        period: Обновляется каждый тик.
    """

    dim: int
    scale: float = 1.0
    seed: int = 0
    tag: str = "noise"
    category: SignalCategory = SignalCategory.EXTEROCEPTIVE
    period: int = 1

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim must be > 0, got {self.dim}")

    def read(self, tick: int, now: float) -> SignalSource:
        """Выборка белого шума на данном тике.

        Args:
            tick: Номер тика (определяет выборку).
            now: Wall-clock время (не используется).

        Returns:
            SignalSource shape=(dim,) с шумом N(0, scale).
        """
        rng = np.random.default_rng(self.seed + tick)
        data = (rng.standard_normal(self.dim) * self.scale).astype(np.float64)
        return SignalSource(category=self.category, data=data, tag=self.tag)


def default_providers(
    message_dim: int = 8,
    seed: int = 0,
    resource_provider: SignalProvider | None = None,
    message_provider: SignalProvider | None = None,
) -> list[SignalProvider]:
    """Стандартный набор источников для host loop.

    Состав: circadian(2) + battery(1) + cpu(1) + message + resources(2, если
    передан). Итоговая ширина шины: 6 + message_dim при наличии ресурсного
    провайдера (по умолчанию 14).

    ``message_provider`` (S2: ``TextMessageProvider`` с эмбеддером) заменяет
    вектор-заглушку ``UserMessageProvider`` без смены интерфейса.

    Args:
        message_dim: Размерность заглушки сообщения (если message_provider
            не передан).
        seed: Зерно детерминированных генераторов (cpu, message).
        resource_provider: Ресурсный провайдер (инъекция meter).
            None → ресурсный канал не добавляется (шина 12).
        message_provider: Готовый коммуникативный провайдер (S2).
            None → ``UserMessageProvider(message_dim)`` (заглушка).

    Returns:
        Список провайдеров в порядке укладки на шину.
    """
    if message_provider is None:
        message_provider = UserMessageProvider(embedding_dim=message_dim, seed=seed)
    providers: list[SignalProvider] = [
        CircadianProvider(),
        BatteryProvider(),
        CpuProvider(seed=seed),
        message_provider,
    ]
    if resource_provider is not None:
        providers.append(resource_provider)
    return providers


def tags_above_threshold(
    errors: Vector,
    segments: tuple[BusSegment, ...],
    threshold: float,
) -> list[str]:
    """Теги сегментов шины с агрегированной по колонкам ошибкой выше порога.

    Чистая функция: не мутирует входы.

    Args:
        errors: Ошибки колонок, shape=(n_columns, bus_dim).
        segments: Карта сегментов шины (name, offset, dim, period).
        threshold: Порог ‖e‖² сегмента.

    Returns:
        Список тегов в порядке сегментов (только превысившие порог).
    """
    if errors.ndim != 2:
        raise ValueError(f"errors must be 2D (n_columns, bus_dim), got {errors.ndim}D")

    # Агрегируем ‖e‖² по колонкам (axis=0) для каждого канала шины
    per_channel = np.sum(errors**2, axis=0)
    tags: list[str] = []
    for segment in segments:
        window = per_channel[segment.offset : segment.offset + segment.dim]
        if float(np.sum(window)) > threshold:
            tags.append(segment.name)
    return tags


@dataclass(frozen=True)
class BusSegment:
    """Сегмент шины: какой источник занимает какой срез вектора ``u(t)``.

    Attributes:
        name: Тег источника.
        offset: Начало сегмента в векторе шины.
        dim: Длина сегмента.
        period: Период обновления источника в тиках.
    """

    name: str
    offset: int
    dim: int
    period: int


DEFAULT_CHANNEL_RANK = 1.0


def channel_importance(
    segments: Sequence[BusSegment],
    ranks: Mapping[str, float],
    *,
    total_dim: int | None = None,
    default_rank: float = DEFAULT_CHANNEL_RANK,
) -> Vector:
    """Развернуть важность каналов в per-component веса ``wᵢ``.

    Честная обработка сигналов (BACKLOG): важность канала ``rank`` — свойство
    органа (скаляр), а его ширина ``dim`` — скрытый вес. Поэтому вес канала
    распределяется по его компонентам как ``rank / dim``: суммарный вклад
    канала в F не зависит от его размерности.

    Чистая функция: не мутирует входы.

    Args:
        segments: Карта сегментов шины (name/offset/dim/period).
        ranks: Важность каналов по тегу (``rank₀``). Отсутствующий тег →
            ``default_rank``. Значение должно быть > 0.
        total_dim: Полная ширина вектора. None → по последнему сегменту.
        default_rank: Важность канала без явного ранга (> 0).

    Returns:
        Вектор важности shape=(total_dim,) с ``rank/dim`` по срезам сегментов.

    Raises:
        ValueError: Если default_rank <= 0, ранг <= 0, dim <= 0 или
            ``total_dim`` меньше покрытия сегментов.
    """
    if default_rank <= 0.0:
        raise ValueError(f"default_rank must be > 0, got {default_rank}")
    for name, rank in ranks.items():
        if rank <= 0.0:
            raise ValueError(f"rank must be > 0 for {name!r}, got {rank}")

    if total_dim is None:
        total_dim = max((s.offset + s.dim for s in segments), default=0)
    if total_dim < 0:
        raise ValueError(f"total_dim must be >= 0, got {total_dim}")

    weights = np.ones(total_dim, dtype=np.float64)
    for segment in segments:
        if segment.dim <= 0:
            raise ValueError(
                f"segment {segment.name!r} has dim <= 0: {segment.dim}"
            )
        if segment.offset + segment.dim > total_dim:
            raise ValueError(
                f"segment {segment.name!r} exceeds total_dim {total_dim}: "
                f"{segment.offset + segment.dim}"
            )
        rank = float(ranks.get(segment.name, default_rank))
        weights[segment.offset : segment.offset + segment.dim] = rank / segment.dim
    return weights


class SignalBus:
    """Imperative Shell: владеет провайдерами и собирает шину ``u(t)``.

    Держит карту сегментов ``[(name, offset, dim, period)]`` — фундамент
    width scaling: в Фазе 2 колонка сможет читать только свой срез шины,
    а не весь вектор (``ColumnConfig.reads``).

    Агрегация делегируется ``SignalRegistry`` (контракт ``src/mcp/``):
    каждый тик источники перерегистрируются, затем конкатенируются.

    Attributes:
        segments: Карта сегментов шины.
        bus_dim: Полная ширина шины (сумма dim всех источников).
        last_signals: Сигналы последнего ``step()`` (для телеметрии).
    """

    def __init__(self, providers: list[SignalProvider]) -> None:
        """Создать шину из списка провайдеров.

        Args:
            providers: Источники сигналов. Порядок задаёт укладку на шину.

        Raises:
            ValueError: Если список пуст, теги не уникальны, dim <= 0
                или period < 1.
        """
        if not providers:
            raise ValueError("providers list must not be empty")

        seen: set[str] = set()
        for provider in providers:
            if provider.tag in seen:
                raise ValueError(f"duplicate provider tag: {provider.tag!r}")
            seen.add(provider.tag)
            if provider.dim <= 0:
                raise ValueError(
                    f"provider {provider.tag!r} has dim <= 0: {provider.dim}"
                )
            if provider.period < 1:
                raise ValueError(
                    f"provider {provider.tag!r} has period < 1: {provider.period}"
                )

        self._providers: list[SignalProvider] = list(providers)
        self._segments = self._build_segments(self._providers)
        self._bus_dim = sum(p.dim for p in self._providers)
        self._registry = SignalRegistry()
        self._cache: dict[str, SignalSource] = {}
        self._last_signals: list[SignalSource] = []

    @staticmethod
    def _build_segments(providers: list[SignalProvider]) -> tuple[BusSegment, ...]:
        """Построить карту сегментов шины.

        Args:
            providers: Провайдеры в порядке укладки.

        Returns:
            Кортеж BusSegment с накопленным offset.
        """
        segments: list[BusSegment] = []
        offset = 0
        for provider in providers:
            segments.append(
                BusSegment(
                    name=provider.tag,
                    offset=offset,
                    dim=provider.dim,
                    period=provider.period,
                )
            )
            offset += provider.dim
        return tuple(segments)

    def step(self, tick: int, now: float) -> Vector:
        """Собрать шину ``u(t)`` на данном тике.

        Провайдеры с ``period > 1`` читаются не каждый тик: последнее
        значение кэшируется и переиспользуется (медленный такт, §3.Ж).

        Args:
            tick: Номер тика хоста (начинается с 0).
            now: Wall-clock время в секундах.

        Returns:
            Агрегированный вектор шины shape=(bus_dim,).

        Raises:
            ValueError: Если сигнал провайдера не совпал с объявленным dim.
        """
        signals: list[SignalSource] = []
        for provider in self._providers:
            cached = self._cache.get(provider.tag)
            if tick % provider.period == 0 or cached is None:
                signal = provider.read(tick, now)
                if signal.data.shape != (provider.dim,):
                    raise ValueError(
                        f"provider {provider.tag!r} returned shape "
                        f"{signal.data.shape}, expected ({provider.dim},)"
                    )
                self._cache[provider.tag] = signal
                cached = signal
            signals.append(cached)

        self._registry.clear()
        self._registry.register_many(signals)
        self._last_signals = signals

        aggregated = self._registry.aggregate()
        if aggregated is None:
            raise ValueError("empty bus: aggregate() returned None")
        return aggregated

    @property
    def segments(self) -> tuple[BusSegment, ...]:
        """Карта сегментов шины (name, offset, dim, period)."""
        return self._segments

    @property
    def bus_dim(self) -> int:
        """Полная ширина шины (сумма dim всех источников)."""
        return self._bus_dim

    @property
    def providers(self) -> list[SignalProvider]:
        """Список провайдеров в порядке укладки (read-only view)."""
        return list(self._providers)

    @property
    def last_signals(self) -> list[SignalSource]:
        """Сигналы последнего ``step()`` (пусто до первого вызова)."""
        return list(self._last_signals)
