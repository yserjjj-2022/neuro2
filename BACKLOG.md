# Бэклог идей и задач

## Как заполнять
- Формат: `[ID] Заголовок | Контекст | Приоритет (P0-P2) | Дата`
- P0 — критично для текущей задачи
- P1 — улучшит архитектуру/производительность
- P2 — интересные идеи, отложенные
- **Правило:** идея → в бэклог → обсуждение → только потом в работу.

## Записи

Статус: ✅ — сделано, ⏳ — в работе, ⛔ — заблокировано, (пусто) — не начато.

### Сделано

| ID | Заголовок | Контекст | Приоритет | Дата |
|----|-----------|----------|-----------|------|
| [Phase1][energy] Task 1 ✅ | Реализовать models.py: FreeEnergyResult | Frozen dataclass: f, valence, allostatic_stress, gamma | P0 | 2025-01-16 |
| [Phase1][energy] Task 2 ✅ | Реализовать calculator.py: FreeEnergyCalculator | Stateless: validate shapes → clip precision → F(t) → valence/stress/gamma | P0 | 2025-01-16 |
| [Phase1][energy] Task 3 ✅ | Реализовать tests/test_energy_calculator.py | 7 unit-тестов: formula, shape mismatch, empty, clip, valence, stress decay | P0 | 2025-01-16 |
| [Phase1][energy] Task 4 ✅ | Реализовать observer.py: EnergyObserver | Shell с состоянием (_prev_f, _prev_stress), DI через sink | P0 | 2025-01-16 |
| [Phase1][energy] Task 5 ✅ | Реализовать tests/test_energy_observer.py | 3 unit-теста: no sink, with sink, state preservation | P0 | 2025-01-16 |
| [Phase1][energy] Task 6 ✅ | Обновить __init__.py | Re-export сущностей energy | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 1 ✅ | Реализовать models.py: TelemetryEvent | Frozen dataclass: timestamp, free_energy, valence, allostatic_stress, active_columns, phase, mode | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 2 ✅ | Реализовать serialize.py: serialize_event | Чистое ядро: json.dumps(allow_nan=False), без I/O | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 3 ✅ | Реализовать tests/test_telemetry_serialize.py | 2 unit-теста: valid event, NaN → ValueError | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 4 ✅ | Реализовать writer.py: TelemetryWriter | Shell: делегирует serialize_event, flush после каждой записи | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 5 ✅ | Реализовать tests/test_telemetry_writer.py | 2 unit-теста: file creation, JSONL format | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 6 ✅ | Реализовать logger.py: TelemetryLogger + SupportsWrite | Shell с DI через Protocol, phase/mode в __init__, crash-safety | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 7 ✅ | Реализовать tests/test_telemetry_logger.py | 2 unit-теста: mock writer, swallows errors | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 8 ✅ | Обновить __init__.py | Re-export сущностей telemetry | P0 | 2025-01-16 |
| [Phase1][integration] Task 1 ✅ | Связка EnergyObserver + TelemetryLogger | sink=lambda r: telemetry_logger.log(r.f, r.valence, r.allostatic_stress) | P0 | 2025-01-16 |
| [Phase3][cmc] Task 1 ✅ | Прокинуть active_columns из cmc в telemetry | CMCPipeline.tick() — sink берёт active_columns из ensemble.active | P1 | 2025-01-16 |
| [Phase3][cmc] Task 2 ✅ | Обновить src/core/__init__.py при реализации cmc | Удалён Column, re-export новых сущностей, docstring обновлён | P0 | 2026-08-16 |
| [Phase1][voting] Task 1 ✅ | Реализовать models.py: VotingResult | Frozen dataclass: indices (k,), mask (N,), scores (k,) | P0 | 2026-08-16 |
| [Phase1][voting] Task 2 ✅ | Реализовать kwta.py: kwta() | Чистая функция: top-k stable argsort, ties → меньший индекс | P0 | 2026-08-16 |
| [Phase1][voting] Task 3 ✅ | Реализовать tests/test_voting_kwta.py | 11 unit-тестов | P0 | 2026-08-16 |
| [Phase1][voting] Task 4 ✅ | Реализовать manager.py: VotingManager | Shell: k, vote() делегирует kwta, last cache, set_k() | P0 | 2026-08-16 |
| [Phase1][voting] Task 5 ✅ | Реализовать tests/test_voting_manager.py | 7 unit-тестов | P0 | 2026-08-16 |
| [Phase1][voting] Task 6 ✅ | Интеграционный тест cmc → voting | Активности ‖e‖² → scores → kwta → победители | P0 | 2026-08-16 |
| [Phase1][voting] Task 7 ✅ | Обновить __init__.py (voting + core) | Re-export kwta, VotingManager, VotingResult | P0 | 2026-08-16 |
| [Phase2][cross-module] Threshold Calibration ✅ | active_threshold: 0.0 → 1e-8 | EMA не сходится к точному 0.0 в float64 | P1 | 2026-08-17 |
| [Phase2][sensory] Signal category taxonomy ✅ | SignalCategory + SignalSource + SignalRegistry | Разведение extero/intero/communicative, reflex-path при severity ≥ 0.9 | P0 | 2026-08-17 |
| [Phase2][architecture] Voting vs Attractors ✅ | Зафиксирован статус voting в Phase 2 | voting — параллельный потребитель scores (телеметрия), не звено пайплайна | P1 | 2026-08-17 |
| [Phase1][memory] Episodic memory ✅ | MemoryStore (SQLite + sqlite-vec) | store/recall, дедуп SHA-256, транзакции, persistence | P0 | 2026-08-16 |
| [Phase1][cmc] Canonical Microcircuits ✅ | column_step + CMCEnsemble | α-EMA L4→L5/6→L2/3, active по ‖e‖² | P0 | 2026-08-16 |
| [Phase2][attractors] Task attractors ✅ | TaskAttractor (STP, Kubota & Aihara) | dwell, basin, immediate switch, convergence safety | P0 | 2026-08-17 |
| [Phase1][host] CMCPipeline ✅ | Сборка per-tick пайплайна | cmc → voting/attractors → energy → telemetry | P0 | 2026-08-16 |
| [Phase1][host] Host loop + `__main__.py` ✅ | Цикл по dt, CLI, graceful shutdown | HostLoop + build_host_loop, SIGINT, `--ticks/--dt/--log` | P0 | 2026-09-26 |
| [Phase1][sensors] Мок-провайдеры сигналов ✅ | src/host/sources.py | Circadian/Battery/Cpu/UserMessage + Constant/Step/Noisy | P0 | 2026-09-26 |
| [Phase1][sensors] Карта сегментов шины ✅ | BusSegment [(name, offset, dim, period)] | Фундамент width scaling и проекций колонок (Фаза 2) | P0 | 2026-09-26 |
| [Phase1][config] Вынос параметров ✅ | src/config/params.py | HostConfig/EnergyConfig/ColumnParams/AttractorConfig, DI через build() | P1 | 2026-09-26 |

### В работе / запланировано

> **Порядок работ определён ADR-0005** (`adr/0005-build-order-fep-fallback-and-guardrails.md`).
> Стадии S1–S6 — см. `BUILD_ROADMAP.md`, ворота — `VALIDATION.md`.
> Ниже сгруппировано по стадиям.

#### S1. Честные сигналы (фундамент) ✅ ЗАВЕРШЕНО 2026-09-26

| ID | Заголовок | Контекст | Приоритет | Дата |
|----|-----------|----------|-----------|------|
| [S1][time] Единая временная база ✅ | tick ↔ секунды; decay/интегралы в секундах | ADR-0006; `dt=0.1` (10 Гц) | P0 | 2026-09-26 |
| [S1][energy] Валидная valence ✅ | Сглаженная EMA, `valence_tau=1.0` | значимых смен знака ~649 → 1 | P0 | 2026-09-26 |
| [S1][energy] Настоящая γ ✅ | Обратная дисперсия, `gamma_max=10` | 1e6 взрывал F; калибровка | P0 | 2026-09-26 |
| [S1][telemetry] Расширение TelemetryEvent ✅ | 15 полей: tick/tags/reflex/bus_dim/gamma/resources/drift | Наблюдаемость | P0 | 2026-09-26 |
| [S1][sensors] ResourceProvider ✅ | Латентность + RSS (intero) | «Тахикардия/одышка» | P0 | 2026-09-26 |
| [S1][safety] Guard NaN/inf + детектор дрейфа ✅ | `HostIntegrityError`/`check_finite` + `DriftDetector` | Заготовка самоконтроля | P1 | 2026-09-26 |
| [S1][validation] Поведенческий регресс ✅ | C1–C6 + `behavioral_fingerprint` | 286 тестов | P0 | 2026-09-26 |
| [S1][arch] ADR-0006: непрерывность + временные шкалы ✅ | Эмоц./рацион./симуляция | Фундаментальное свойство | P0 | 2026-09-26 |

#### S2. Непрерывность

| ID | Заголовок | Контекст | Приоритет | Дата |
|----|-----------|----------|-----------|------|
| [S2][memory] Эмбеддер (API) | Текст → вектор; keystone | Разблокирует memory-wiring и коммуникативный вход | P0 | 2026-09-26 |
| [S2][memory] Эпизоды на значимых событиях | Всплеск F, reflex → store | Источник content/embedding появляется | P0 | 2026-09-26 |
| [S2][memory] Recall → приор/точность | Прошлое влияет на поведение | Проверка «узнавания» после перезапуска | P0 | 2026-09-26 |

#### S3. Голос

| ID | Заголовок | Контекст | Приоритет | Дата |
|----|-----------|----------|-----------|------|
| [S3][speech] Intent-Frame | Из (F, valence, аттрактор, прецеденты) | Речь Б1 (ADR-0003) | P0 | 2026-09-26 |
| [S3][speech] Event-triggered LLM | Вызов только при F > порога | Экологическая рациональность (§3.Е) | P0 | 2026-09-26 |
| [S3][speech] Диалоговый стенд | CLI-чат + HITL-протокол | Валидация тона/уместности | P0 | 2026-09-26 |

#### S4. Воля

| ID | Заголовок | Контекст | Приоритет | Дата |
|----|-----------|----------|-----------|------|
| [S4][policy] Action Selection layer | Выбор действия + причинная трассировка | goal-directed test + explainability (§3.И) | P0 | 2026-09-26 |
| [S4][sensory] Reflex path | Обход cmc/attractors, реакция за 1 тик | Критические интеро-сигналы (§3.К) | P0 | 2026-09-26 |
| [S4][homeostasis] Сетпоинты/корневые приоры | Гомеостаз + постоянные ценности | Манифест §3.В | P0 | 2026-09-26 |
| [S4][energy] γ как пред-колоночный барьер | Управление предсказанием | Решение ADR-0005 §2 (этап 2) | P1 | 2026-09-26 |
| [S4][control] Канал горячего тестирования | status/pause/step/inject/set/snapshot/freeze/kill | Оперативный контроль + HITL | P0 | 2026-09-26 |
| [S4][safety] Capability gate | Единая точка side-effect: tier/HITL/бюджет | Гардрейлы (ADR-0005 §9) | P0 | 2026-09-26 |
| [S4][metacognition] Канал состояний ризонинга (открытый дизайн) | Метакогнитивные наблюдаемые как отдельный сигнальный канал: конфликт колонок (разброс scores/энтропия маски), метастабильность аттрактора (history, частота переключений), эпистемическая неопределённость (разброс предсказаний), насыщение/тренд F. НЕ дублировать F/valence/stress/gamma. Вопрос: четвёртая категория сигналов (`METACOGNITIVE`) или подкатегория intero — разные маршруты (гомеостаз→рефлекс vs метакогниция→эпистемическое действие). Риск circularity: read-only наблюдение, влияние на следующий тик/отдельный слой, не в F того же тика. Потребители — S4 (policy) и S6 (самоконтроль). Кандидат на отдельный ADR при проектировании S4. | P1 | 2026-09-26 |

#### S5. Социальность

| ID | Заголовок | Контекст | Приоритет | Дата |
|----|-----------|----------|-----------|------|
| [S5][tm] Модель партнёра | partner_trust, partner_state | Манифест §3.Г | P1 | 2026-09-26 |
| [S5][tm] Vigilance Gate | Новые утверждения как гипотезы | Защита от эпистемического дрейфа | P1 | 2026-09-26 |
| [S5][tm] Joint Agency | Совместные целевые аттракторы | Сотрудничество, а не исполнение | P1 | 2026-09-26 |

#### S6. Автономия

| ID | Заголовок | Контекст | Приоритет | Дата |
|----|-----------|----------|-----------|------|
| [S6][memory] Ночная консолидация | Pruning + Structure Learning | Манифест §3.Д, §4 | P1 | 2026-09-26 |
| [S6][policy] Эпистемический драйв | Исследование неопределённости | Манифест §3.В | P1 | 2026-09-26 |
| [S6][policy] Полный дискретный слой | Sparse-факторизация (pymdp) | Решение ADR-0005 §3 (этап 2) | P1 | 2026-09-26 |
| [S6][safety] Детектор дрейфа + протокол сброса | Самоконтроль first-class | Манифест §6.4 | P1 | 2026-09-26 |

#### Прочее / техдолг

| ID | Заголовок | Контекст | Приоритет | Дата |
|----|-----------|----------|-----------|------|
| [Phase2][sensors] Медленный такт источников | period per channel | Поле `period` есть в SignalBus | P2 | 2026-09-26 |
| [Phase2][sensors] Проекции колонок | ColumnConfig.reads | O(Σ dim(slice)) вместо O(N·D) | P2 | 2026-09-26 |
| [Phase2][sensors] Реальные интеграции | datetime → psutil → MCP | После замыкания контура | P2 | 2026-09-26 |
| [Phase2][tech-debt] Purity Test для Calculator | Одинаковый вход → одинаковый выход | Прямой тест свойства | P2 | 2025-01-16 |
| [Phase2][tech-debt] Восстановление Observer при перезапуске | prev_f/prev_stress из memory | Интеграция с последним эпизодом | P2 | 2025-01-16 |
| [Phase1][memory-wiring] ⛔ Blocked | Wiring memory в host loop | Снимается в S2 (эмбеддер) | P1 | 2026-08-16 |

### Фаза 3+ (архитектурные пробелы — покрыты стадиями выше)

| ID | Заголовок | Контекст | Приоритет | Дата |
|----|-----------|----------|-----------|------|
| [Phase3][policy] Action Selection layer | EFE-based выбор действия по текущему аттрактору | → S4 | P1 | 2026-08-17 |
| [Phase3][sensory] Reflex path | Обход cmc/attractors для критических сигналов | → S4 | P1 | 2026-08-17 |
| [Phase3][policy] Trust calibration guardrail | Причинная прослеживаемость вместо «ощущения понятности» | → S4/S5 | P1 | 2026-08-17 |
