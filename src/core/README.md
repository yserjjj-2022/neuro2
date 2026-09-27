# core

Колоночное ядро хоста. Все модули следуют Functional Core / Imperative Shell
(ADR-0004): чистое ядро + shell с состоянием и I/O.

| Модуль | Роль | Core | Shell |
|---|---|---|---|
| cmc | Колоночная динамика L4→L5/6→L2/3 | `column_step` | `CMCEnsemble` |
| energy | F(t), valence, stress, γ | `FreeEnergyCalculator` | `EnergyObserver` |
| voting | Латеральное торможение k-WTA | `kwta` | `VotingManager` |
| attractors | Аттрактор задачи (STP, гистерезис) | `compute_dwell`, `check_*` | `TaskAttractor` |
| homeostasis | Сетепоинты интеро-каналов (S4) | `setpoint_deviation` | `Homeostat` |
| policy | Выбор действия + трасса (S4) | `evaluate_candidates`, `select_action` | — |

`cmc` также экспортирует `attention_gate`/`apply_attention` — пред-колоночный
барьер внимания (S4, ADR-0005 §2).

Поток одного тика: `u(t)` → cmc → {voting, attractors} → energy → telemetry.
Гомеостаз и policy (S4) работают над сигналами шины и состоянием тика.