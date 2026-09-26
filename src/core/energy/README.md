# Free Energy & Affective Signals

Аффективный контур хоста: F(t), валентность (−dF/dt, сглаженная),
аллостатический стресс (утечка + интеграл), точность γ (обратная дисперсия).

S1: единая временная база (секунды), человеческие константы
(`valence_tau≈1 с`, `stress_leak≈0.01/с`), `gamma_max=10`.

- Core: `FreeEnergyCalculator`, `inverse_variance`
- Shell: `EnergyObserver` (владеет `EnergyState`), `PrecisionEstimator`, `DriftDetector`
- Guards: `HostIntegrityError`, `check_finite`

См. `SPEC.md` (формулы) и ADR-0006 (временные шкалы).
