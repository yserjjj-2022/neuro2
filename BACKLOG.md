# Бэклог идей и задач

## Как заполнять
- Формат: `[ID] Заголовок | Контекст | Приоритет (P0-P2) | Дата`
- P0 — критично для текущей задачи
- P1 — улучшит архитектуру/производительность
- P2 — интересные идеи, отложенные
- **Правило:** идея → в бэклог → обсуждение → только потом в работу.

## Записи

| ID | Заголовок | Контекст | Приоритет | Дата |
|----|-----------|----------|-----------|------|
| [Phase1][energy] Task 1 | Реализовать models.py: FreeEnergyResult | Создать frozen dataclass с полями f, valence, stress, gamma | P0 | 2025-01-16 |
| [Phase1][energy] Task 2 | Реализовать calculator.py: FreeEnergyCalculator | Stateless калькулятор: validate shapes → clip precision → F(t) → valence/stress/gamma | P0 | 2025-01-16 |
| [Phase1][energy] Task 3 | Реализовать tests/test_energy_calculator.py | 7 unit-тестов: formula, shape mismatch, empty arrays, clip, valence, stress decay | P0 | 2025-01-16 |
| [Phase1][energy] Task 4 | Реализовать observer.py: EnergyObserver | Shell с состоянием (self._prev_f, self._prev_stress), DI через sink, observe() | P0 | 2025-01-16 |
| [Phase1][energy] Task 5 | Реализовать tests/test_energy_observer.py | 3 unit-теста: no sink, with sink, state preservation between calls | P0 | 2025-01-16 |
| [Phase1][energy] Task 6 | Обновить __init__.py | Re-export FreeEnergyResult, FreeEnergyCalculator, EnergyObserver | P0 | 2025-01-16 |
| [Phase2][tech-debt] Task 1 | Добавить Purity Test для Calculator | Прямой тест архитектурного свойства: одинаковый вход → одинаковый выход, порядок не важен | P2 | 2025-01-16 |
| [Phase2][tech-debt] Task 2 | Восстановление состояния Observer при перезапуске | Интеграция с memory для загрузки prev_f/prev_stress из последнего эпизода | P2 | 2025-01-16 |
| [Phase1][telemetry] Task 1 | Реализовать models.py: TelemetryEvent | Создать frozen dataclass с полями timestamp, free_energy, valence, stress, active_columns, phase, mode | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 2 | Реализовать serialize.py: serialize_event | Чистое ядро: json.dumps(allow_nan=False), без I/O | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 3 | Реализовать tests/test_telemetry_serialize.py | 2 unit-теста: valid event, NaN raises ValueError | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 4 | Реализовать writer.py: TelemetryWriter | Shell: делегирует serialize_event, flush после каждой записи | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 5 | Реализовать tests/test_telemetry_writer.py | 2 unit-теста: file creation, JSONL format | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 6 | Реализовать logger.py: TelemetryLogger + SupportsWrite | Shell с DI через Protocol, phase/mode в __init__, crash-safety | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 7 | Реализовать tests/test_telemetry_logger.py | 2 unit-теста: mock writer, swallows errors | P0 | 2025-01-16 |
| [Phase1][telemetry] Task 8 | Обновить __init__.py | Re-export TelemetryEvent, serialize_event, TelemetryWriter, TelemetryLogger | P0 | 2025-01-16 |
| [Phase1][integration] Task 1 | Связка EnergyObserver + TelemetryLogger | Правильный адаптер: `sink=lambda r: telemetry_logger.log(r.f, r.valence, r.allostatic_stress)` — active_columns=0 по умолчанию. r.f (НЕ r.free_energy!) | P0 | 2025-01-16 |
| [Phase3][cmc] Task 1 | Прокинуть реальный active_columns из src/core/cmc/ в telemetry_logger.log() | РЕШЕНО (2026-08-16): CMCPipeline.tick() в src/host/wiring.py — sink берёт active_columns из ensemble.active. | P1 | 2025-01-16 |
| [Phase3][cmc] Task 2 | Обновить src/core/__init__.py при реализации cmc | Удалить `from .cmc import Column`, re-export новые сущности (ColumnConfig, ColumnState, column_step, CMCEnsemble, EnsembleOutput). Одновременно с удалением Column из src/core/cmc/__init__.py — иначе импорт пакета сломается. Обновить docstring (убрать «SIMD-оптимизация», описать FC/IS). | P0 | 2026-08-16 |
| [Phase1][memory-wiring] Blocked | Wiring memory в host loop | ЗАБЛОКИРОВАНО: нет источника content/embedding для Episode. Energy/telemetry не производят ни текстового описания, ни эмбеддинга; модуль-эмбеддер в roadmap не заявлен (после memory идут cmc и voting). Минимальный стаб — build_memory_store() создаёт MemoryStore без автоматического вызова store(). Переоценить при планировании Фазы 2/cmc. | P1 | 2026-08-16 |
| [Phase1][voting] Task 1 | Реализовать models.py: VotingResult | Frozen dataclass: indices (k,), mask (N,) float64, scores (k,). Type aliases: Vector, IndexVector (идиома из memory/serialize.py). | P0 | 2026-08-16 |
| [Phase1][voting] Task 2 | Реализовать kwta.py: kwta() | Чистая функция: top-k через stable argsort, ties → меньший индекс. Fail-fast валидация k/scores. | P0 | 2026-08-16 |
| [Phase1][voting] Task 3 | Реализовать tests/test_voting_kwta.py | 11 unit-тестов: basic, top-k property, k=1, k=N, ties, invalid k ×2, empty/2D scores, purity, mask.sum | P0 | 2026-08-16 |
| [Phase1][voting] Task 4 | Реализовать manager.py: VotingManager | Shell: k (tuning), vote() делегирует kwta, last cache, set_k(). k=1 по умолчанию. | P0 | 2026-08-16 |
| [Phase1][voting] Task 5 | Реализовать tests/test_voting_manager.py | 7 unit-тестов: default k, invalid init, vote, k>N, last none/after, set_k | P0 | 2026-08-16 |
| [Phase1][voting] Task 6 | Интеграционный тест cmc → voting | Активности ‖e‖² из CMCEnsemble → scores → kwta → победители. Сходимость → нулевые scores → ties. | P0 | 2026-08-16 |
| [Phase1][voting] Task 7 | Обновить __init__.py (voting + core) | Re-export kwta, VotingManager, VotingResult. Добавить voting в src/core/__init__.py. | P0 | 2026-08-16 |
| [Phase2][cross-module] Threshold Calibration | Вынести active_threshold из теста в конфигурацию модуля | EMA никогда не сходится к точному 0.0 (свойство рекуррентного фильтра, не баг) → дефолт 0.0 даёт ложноположительные active_columns при любой длительной сходимости; нужно явное значение (1e-8 или адаптивный) в публичном API; паттерн повторился на трёх уровнях: cmc, voting, energy — при калибровке проверить все три модуля, иначе телеметрия Фаза 2 даст ложные сигналы нестабильности | P1 | 2026-08-17 |
| [Phase2][architecture] Voting vs Attractors | Зафиксировать статус voting в Phase 2 | voting.vote() вызывается в CMCPipeline.tick(), но .last не потребляется attractors (attractor читает scores напрямую). Voting существует для телеметрии/логирования, не как звено пайплайна. Обновить документацию. | P1 | 2026-08-17 |
| [Phase2][sensory] Signal category taxonomy | Явное разведение extero-/intero-/коммуникативных сигналов в контракте SignalSource | Выходит из обсуждения: источники принципиально разнородны (погода vs батарея vs текст), требуют разных маршрутов обработки. Блокирует дизайн registry. | P0 | 2026-08-17 |
| [Phase3][policy] Action Selection layer | EFE-based выбор действия по текущему аттрактору; критерий приёмки — чувствительность к изменению ценности исхода (goal-directed test); explainability — обязательный инвариант, не опция | Архитектурный пробел: есть восприятие (attractors), нет действия (policy). Не заменяет attractors, идёт после. | P1 | 2026-08-17 |
| [Phase3][sensory] Reflex path | Обход cmc/attractors для критических/интероцептивных сигналов; зависит от готового SignalSource/registry контракта | Динамика attractors (dwell, basin) — вредна для сигналов вроде критической батареи. Нужен параллельный маршрут за один тик. | P1 | 2026-08-17 |
| [Phase3][policy] Trust calibration guardrail | Явный принцип: не оптимизировать под антропоморфные сигналы доверия в отрыве от объяснимости; риск — слой, обученный «звучать убедительно» без причинной трассировки решения | Анти-паттерн из манифеста §3.И: тёплый тон ≠ надёжность. Policy-слой должен оптимизироваться под причинную прослеживаемость, не под «ощущение понятности». | P1 | 2026-08-17 |
