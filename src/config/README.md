# Config

Единый источник настраиваемых параметров хоста (CONSTITUTION §2.2):
`dt`, пороги (`active_threshold`, `basin_threshold`, `dominance`), `alpha`,
`dwell`, `precision_mode`, набор колонок.

`HostConfig` собирает `EnergyConfig` + `ColumnParams` + `AttractorConfig` и
прокидывается в `build_host_loop`. Значения — стартовые (тестовые),
калибровка по телеметрии — Фаза 2/3.

См. `SPEC.md` и `PLAN.md`.