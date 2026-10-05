"""Named configuration presets — deterministic HostConfig factories (S7-B).

A preset is a named snapshot of the configuration tied to a scenario (ADR-0010
§6). Base presets are Python factories (type-safe, deterministic); a partial
TOML override (stdlib ``tomllib``, no new dependency) may be layered on top via
:func:`load_preset`.

Base invariants (all presets): ``clock_mode="synthetic"``, ``llm_mode="fake"``,
``embedder_mode="fake"`` and a fixed ``seed`` — so the same input yields the same
fingerprint (ADR-0010 §6, VALIDATION §5). Real LLMs/embedders are a separate
override, never used for fingerprint comparison.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.config.params import (
    AutonomyConfig,
    HostConfig,
    MemoryConfig,
    PolicyConfig,
    SocialConfig,
    SpeechConfig,
)
from src.core.homeostasis import Setpoint


def _deterministic_base() -> HostConfig:
    """Общая детерминированная база всех пресетов."""
    return HostConfig(
        seed=0,
        clock_mode="synthetic",
        precision_mode="ones",
        memory=MemoryConfig(embedder_mode="fake"),
        speech=SpeechConfig(llm_mode="fake"),
    )


def baseline() -> HostConfig:
    """Канонический пресет: всё дефолтное, детерминизм (эталон отпечатка)."""
    return _deterministic_base()


def stress() -> HostConfig:
    """Низкие сетпоинты и частый рефлекс: throttle/escape hatch/гомеостаз."""
    return replace(
        _deterministic_base(),
        homeostasis=dataclasses.replace(
            HostConfig().homeostasis,
            setpoints=(
                Setpoint(tag="battery", comfort=0.3, critical=0.5),
                Setpoint(tag="resources", comfort=0.3, critical=0.5),
                Setpoint(tag="cpu", comfort=0.5, critical=0.7),
            ),
            reflex_threshold=0.5,
            escape_hatch_ticks=2,
        ),
    )


def dialogue() -> HostConfig:
    """Речь + память + policy + ToM (fake): тон/уместность/узнавание."""
    return replace(
        _deterministic_base(),
        speech=SpeechConfig(enabled=True, llm_mode="fake", f_threshold=0.5),
        policy=PolicyConfig(enabled=True, mode="free"),
        social=SocialConfig(enabled=True),
    )


def autonomy() -> HostConfig:
    """Автономия + самоконтроль + ночная консолидация: наблюдаемые/сброс."""
    return replace(
        _deterministic_base(),
        autonomy=AutonomyConfig(
            enabled=True,
            consolidate_every_ticks=50,
            consolidate_min_episodes=5,
        ),
    )


def long_horizon() -> HostConfig:
    """Длинный горизонт (synthetic): тихий дрейф/устойчивость (C10)."""
    return replace(
        _deterministic_base(),
        max_ticks=1500,
        autonomy=AutonomyConfig(enabled=True),
    )


def cooperative() -> HostConfig:
    """Социальный контур: режим cooperative + ToM + Joint Agency (в чате)."""
    return replace(
        dialogue(),
        policy=PolicyConfig(enabled=True, mode="cooperative"),
    )


_PRESETS = {
    "baseline": baseline,
    "stress": stress,
    "dialogue": dialogue,
    "autonomy": autonomy,
    "long-horizon": long_horizon,
    "cooperative": cooperative,
}


def _merge(config: HostConfig, data: dict[str, Any], path: str = "") -> HostConfig:
    """Рекурсивно наложить TOML-данные на конфиг (fail-fast).

    Args:
        config: Базовый конфиг (frozen dataclass).
        data: Словарь из TOML (секции → вложенные dataclass'ы).
        path: Префикс пути для сообщений об ошибке.

    Returns:
        Новый HostConfig с перезаписанными полями.

    Raises:
        ValueError: Если ключ неизвестен, секция наложена на не-dataclass или
            значение не проходит валидацию dataclass.
    """
    fields = {f.name for f in dataclasses.fields(config)}
    updates: dict[str, Any] = {}
    for key, value in data.items():
        if key not in fields:
            raise ValueError(f"unknown preset key {path + key!r}")
        current = getattr(config, key)
        if isinstance(value, dict):
            if not dataclasses.is_dataclass(current) or isinstance(current, type):
                raise ValueError(
                    f"preset key {path + key!r} is not an overridable section"
                )
            updates[key] = _merge(current, value, path=f"{path}{key}.")
        else:
            updates[key] = value
    return replace(config, **updates)


def load_preset(name: str, *, override: Path | None = None) -> HostConfig:
    """Собрать пресет по имени и наложить TOML-override (fail-fast).

    Args:
        name: Имя пресета (``baseline``, ``stress``, ``dialogue``, ``autonomy``,
            ``long-horizon``, ``cooperative``).
        override: Путь к TOML-файлу частичной перезаписи (None → только база).

    Returns:
        Готовый HostConfig.

    Raises:
        ValueError: Если пресет неизвестен, override содержит неизвестный ключ
            или недопустимое значение.
    """
    if name not in _PRESETS:
        raise ValueError(
            f"unknown preset {name!r}; known: {sorted(_PRESETS)}"
        )
    config = _PRESETS[name]()
    if override is not None:
        with open(override, "rb") as handle:
            data = tomllib.load(handle)
        config = _merge(config, data)
    return config


def available_presets() -> tuple[str, ...]:
    """Имена доступных пресетов (детерминированный порядок).

    Returns:
        Кортеж имён.
    """
    return tuple(_PRESETS)
