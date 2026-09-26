# Config

Единый источник настраиваемых параметров хоста (CONSTITUTION §2.2):
`dt` (10 Гц, эмоц. контур), человеческие константы аффекта
(`valence_tau≈1 с`, `stress_leak≈0.01/с`), `gamma_max`, `time_scale`,
пороги дрейфа, колонки.

`HostConfig` собирает `EnergyConfig` + `ColumnParams` + `AttractorConfig` и
прокидывается в `build_host_loop`. Значения — стартовые (калибровка S1–S2).

См. `SPEC.md`, `PLAN.md` и ADR-0006 (временные шкалы).