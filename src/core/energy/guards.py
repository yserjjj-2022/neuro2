"""Guards for host integrity — fail-fast on non-finite affective state.

S1: превращаем «тихое продолжение с мусором» в явную ошибку. HostLoop
вызывает ``check_finite`` после каждого тика; нарушение → HostIntegrityError,
перехватывается в main → graceful shutdown.
"""

from __future__ import annotations

import math

from .models import FreeEnergyResult


class HostIntegrityError(RuntimeError):
    """Нарушение целостности состояния хоста (NaN/inf в аффективных метриках).

    Отличается от ValueError (ошибка входа): это признак того, что
    вычисление разошлось и продолжать нельзя — нужен graceful shutdown.
    """


def check_finite(result: FreeEnergyResult) -> None:
    """Проверить конечность полей FreeEnergyResult (fail-fast).

    Args:
        result: Результат тика.

    Raises:
        HostIntegrityError: Если f/valence/allostatic_stress/gamma не конечны
            (NaN или inf) — с указанием конкретного поля.
    """
    values = {
        "f": result.f,
        "valence": result.valence,
        "allostatic_stress": result.allostatic_stress,
        "gamma": result.gamma,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise HostIntegrityError(
                f"non-finite {name}: {value!r} — host state integrity violated"
            )
