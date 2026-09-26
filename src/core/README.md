# core

Колоночное ядро хоста. Все модули следуют Functional Core / Imperative Shell
(ADR-0004): чистое ядро + shell с состоянием и I/O.

| Модуль | Роль | Core | Shell |
|---|---|---|---|
| cmc | Колоночная динамика L4→L5/6→L2/3 | `column_step` | `CMCEnsemble` |
| energy | F(t), valence, stress, γ | `FreeEnergyCalculator` | `EnergyObserver` |
| voting | Латеральное торможение k-WTA | `kwta` | `VotingManager` |
| attractors | Аттрактор задачи (STP, гистерезис) | `compute_dwell`, `check_*` | `TaskAttractor` |

Поток одного тика: `u(t)` → cmc → {voting, attractors} → energy → telemetry.