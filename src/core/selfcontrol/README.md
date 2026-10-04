# selfcontrol — самоконтроль хоста (S6)

Метакогнитивные наблюдаемые, детектор тихого дрейфа (critical slowing down),
классификатор «развитие vs дрейф» и протокол сброса.

## Зачем

Манифест §6.4: накопительная память и эволюционирующие приоры могут незаметно
увести поведение от нормы (**тихий дрейф**). Нужен слой самодиагностики
first-class: заметить приближение к смене режима и инициировать управляемый
сброс. INTENT §4: guard различает **развитие** (core сохранён, трассируемо,
когерентно) и **дрейф** (иначе), а не запрещает изменение.

## Состав

- `models.py` — `Metacognition`, `CriticalSlowingDown`, `ChangeKind`,
  `ChangeAssessment`, `ResetLevel`, `ResetPlan` (Core, frozen).
- `compute.py` — чистые функции: `compute_conflict`,
  `compute_metastability`, `compute_saturation`, `critical_slowing_down`,
  `classify_change`, `plan_reset`.
- `monitor.py` — `SelfMonitor` (Shell): кольцевые буферы F/смен, наблюдаемые +
  план сброса на каждом тике.

## Ключевые принципы

- **Триггер — не голый стресс** (BACKLOG): critical slowing down (рост
  дисперсии И lag-1 автокорреляции) — ранний признак смены режима.
- **Три механизма** (не смешивать): SOFT (adaptive reset), FREEZE (regime
  shift), HARD (catastrophic drift — аварийная остановка, не «к норме»).
- **Core не сбрасывается:** reset меняет установки/приоритеты, якоря — нет.
- **Read-only:** наблюдаемые не влияют на F того же тика (анти-circularity).

Решения — ADR-0009. Спека — `SPEC.md`, стадия — `stages/S6_SPEC.md`.
