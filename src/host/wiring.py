"""Wiring — connects independently developed modules into a working pipeline.

Builds the full per-tick pipeline: CMC → voting → attractors → energy.
``CMCPipeline.tick(u, precision, dt)`` runs one complete host tick and returns
``TickOutcome`` — pure composition, no I/O. Telemetry lives in the host loop
(it has the full context: bus, resources, clock).

This is the ONLY place that knows the concrete field names of
FreeEnergyResult (result.f, result.valence, ...), isolating interface desync.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.attractors import TaskAttractor
from src.core.cmc import CMCEnsemble, ColumnConfig
from src.core.cmc.models import Vector
from src.core.energy import EnergyObserver, FreeEnergyCalculator, FreeEnergyResult
from src.core.voting import VotingManager
from src.host.sources import BusSegment
from src.telemetry import TelemetryLogger, TelemetryWriter


@dataclass(frozen=True)
class TickOutcome:
    """Снимок результата одного тика конвейера (без I/O).

    Attributes:
        result: Аффективные метрики (F, valence, stress, gamma).
        active_tags: Теги сегментов шины с ошибкой выше порога.
        reflex_tags: Теги критических сигналов текущего тика.
        activities: Активности колонок (‖e‖²) — вход selfcontrol (S6).
        switched: Сменился ли аттрактор на этом тике (S6).
    """

    result: FreeEnergyResult
    active_tags: tuple[str, ...]
    reflex_tags: tuple[str, ...]
    activities: Vector | None = None
    switched: bool = False


@dataclass(frozen=True)
class CMCPipeline:
    """Композиция per-tick: CMC → voting → attractors → energy (без I/O).

    Attributes:
        ensemble: Ансамбль колонок (производитель e(t) и активностей).
        voting: k-WTA по активностям колонок.
        attractor: TaskAttractor — динамика выбора задачи.
        observer: EnergyObserver с DI sink (по умолчанию None).
        active_threshold: Порог для определения активных сегментов шины.
    """

    ensemble: CMCEnsemble
    voting: VotingManager
    attractor: TaskAttractor
    observer: EnergyObserver
    active_threshold: float = 1e-8

    def tick(
        self,
        u: Vector,
        precision: Vector,
        dt: float,
        segments: tuple[BusSegment, ...] = (),
        reflex_tags: tuple[str, ...] = (),
    ) -> TickOutcome:
        """Один полный тик: CMC → voting/attractors → energy.

        Args:
            u: Вход ансамбля L4, shape == (input_dim,).
            precision: Вектор точности γ, shape == raveled errors
                (N_columns * input_dim).
            dt: Шаг интегрирования в секундах (> 0).
            segments: Карта сегментов шины для active_tags.
            reflex_tags: Теги критических сигналов текущего тика.

        Returns:
            TickOutcome — метрики и теги текущего тика.

        Raises:
            ValueError: Если u.shape != (input_dim,) или
                precision.shape != (N_columns * input_dim,) — fail-fast.
        """
        out = self.ensemble.step(u)

        expected = np.ravel(out.errors).shape
        if precision.shape != expected:
            raise ValueError(
                f"Shape mismatch: precision {precision.shape} != "
                f"raveled errors {expected} (N_columns * input_dim)"
            )

        # Активности колонок = ‖e‖² по строкам errors → вход для k-WTA
        activities = np.sum(out.errors**2, axis=1)
        self.voting.vote(activities)
        prev_mask = self.attractor.current_mask
        self.attractor.tick(activities)
        switched = _attractor_switched(prev_mask, self.attractor.current_mask)

        active_tags: tuple[str, ...] = ()
        if segments:
            active_tags = tuple(
                _segment_tags_above(out.errors, segments, self.active_threshold)
            )

        result = self.observer.observe(np.ravel(out.errors), precision, dt)
        return TickOutcome(
            result=result,
            active_tags=active_tags,
            reflex_tags=reflex_tags,
            activities=activities,
            switched=switched,
        )


def _attractor_switched(prev_mask: Vector | None, cur_mask: Vector | None) -> bool:
    """Сменился ли аттрактор между тиками (по маске победителя, S6).

    Args:
        prev_mask: Маска до tick() (None на первом тике).
        cur_mask: Маска после tick().

    Returns:
        True, если индекс победителя изменился.
    """
    if prev_mask is None or cur_mask is None:
        return False
    return int(np.argmax(prev_mask)) != int(np.argmax(cur_mask))


def _segment_tags_above(
    errors: Vector,
    segments: tuple[BusSegment, ...],
    threshold: float,
) -> list[str]:
    """Теги сегментов с агрегированной по колонкам ‖e‖² выше порога.

    Args:
        errors: Ошибки колонок, shape=(n_columns, bus_dim).
        segments: Карта сегментов шины.
        threshold: Порог.

    Returns:
        Список тегов в порядке сегментов.
    """
    per_channel = np.sum(errors**2, axis=0)
    tags: list[str] = []
    for segment in segments:
        window = per_channel[segment.offset : segment.offset + segment.dim]
        if float(np.sum(window)) > threshold:
            tags.append(segment.name)
    return tags


def build_energy_pipeline(log_path: Path) -> EnergyObserver:
    """Собирает observer, подключённый к файловому логгеру (legacy).

    Args:
        log_path: Путь к JSONL-файлу для записи телеметрии.

    Returns:
        Настроенный EnergyObserver с sink, который пишет в указанный файл.
    """
    writer = TelemetryWriter(log_path=log_path)
    telemetry_logger = TelemetryLogger(writer=writer, phase="phase1", mode="free")

    def sink(result: FreeEnergyResult) -> None:
        telemetry_logger.log(
            free_energy=result.f,
            valence=result.valence,
            allostatic_stress=result.allostatic_stress,
            gamma=result.gamma,
        )

    return EnergyObserver(calculator=FreeEnergyCalculator(), sink=sink)


def build_cmc_pipeline(
    columns: list[ColumnConfig],
    k: int,
    log_path: Path,
    active_threshold: float = 1e-8,
    attractor: TaskAttractor | None = None,
    calculator: FreeEnergyCalculator | None = None,
) -> CMCPipeline:
    """Собирает per-tick конвейер: CMC → voting → attractors → energy.

    Телеметрия больше не входит в pipeline (переехала в host loop): pipeline
    — чистая композиция, возвращает TickOutcome.

    Args:
        columns: Конфигурации колонок ансамбля (единые input_dim/state_dim).
        k: Число победителей k-WTA (1 = hard-WTA).
        log_path: Путь к JSONL-файлу телеметрии (не используется здесь;
            сохранён для обратной совместимости сигнатуры — loop владеет
            writer'ом).
        active_threshold: Порог активности колонки/сегмента.
        attractor: Готовый TaskAttractor (из config). None → дефолт.
        calculator: Готовый FreeEnergyCalculator (из config). None → дефолт.

    Returns:
        CMCPipeline — готовый к tick(u, precision, dt).
    """
    del log_path  # телеметрия переехала в loop
    ensemble = CMCEnsemble(columns=columns, active_threshold=active_threshold)
    voting = VotingManager(k=k)
    if attractor is None:
        attractor = TaskAttractor(n_tasks=len(columns))
    if calculator is None:
        calculator = FreeEnergyCalculator()

    observer = EnergyObserver(calculator=calculator, sink=None)
    return CMCPipeline(
        ensemble=ensemble,
        voting=voting,
        attractor=attractor,
        observer=observer,
        active_threshold=active_threshold,
    )
