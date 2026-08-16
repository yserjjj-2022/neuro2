from dataclasses import dataclass


@dataclass(frozen=True)
class FreeEnergyResult:
    """Результат расчёта свободной энергии."""
    f: float                    # Свободная энергия F(t) ≥ 0
    valence: float              # Валентность -dF/dt
    allostatic_stress: float    # Интеграл F(t) по времени (затухающий)
    gamma: float                # Precision weighting γ
