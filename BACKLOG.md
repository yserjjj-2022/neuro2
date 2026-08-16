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
