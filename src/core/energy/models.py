from dataclasses import dataclass


@dataclass(frozen=True)
class FreeEnergyResult:
    """Результат расчёта свободной энергии."""

    f: float  # Свободная энергия F(t) ≥ 0
    valence: float  # Валентность -dF/dt (сглаженная)
    allostatic_stress: float  # Интеграл F(t) по времени (затухающий)
    gamma: float  # Precision weighting γ


@dataclass(frozen=True)
class EnergyState:
    """Иммутабельный снимок состояния аффективного контура.

    Аналог ColumnState (cmc) и Episode (memory) — frozen dataclass как
    снимок состояния в момент t. Заменяет пару prev_f/prev_stress:
    observer владеет одним EnergyState вместо разрозненных полей.

    Attributes:
        f: F(t-1) — предыдущее значение свободной энергии.
        stress: allostatic_stress(t-1) — предыдущий интеграл.
        valence: valence(t-1) — предыдущая сглаженная валентность
            (для экспоненциального сглаживания по времени).
    """

    f: float = 0.0
    stress: float = 0.0
    valence: float = 0.0
