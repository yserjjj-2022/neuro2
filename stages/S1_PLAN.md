# PLAN.md — S1: Честные сигналы

Реализация `stages/S1_SPEC.md`. Порядок — от чистого ядра к обвязке, каждый
шаг с тестами. Модульные SPEC (`src/core/energy/SPEC.md`,
`src/telemetry/SPEC.md`, `src/host/SPEC.md`) обновляются по ходу.

## Файлы

### Новые

1. `src/core/energy/models.py` — добавить `EnergyState` (frozen)
2. `src/core/energy/precision.py` — `inverse_variance` (Core) + `PrecisionEstimator` (Shell)
3. `src/core/energy/guards.py` — `HostIntegrityError`, `check_finite`
4. `src/core/energy/drift.py` — `DriftDetector` (заготовка)
5. `src/host/resources.py` — `ResourceMeter` + `ResourceProvider`
6. `src/tests/test_energy_precision.py`
7. `src/tests/test_energy_guards.py`
8. `src/tests/test_host_resources.py`
9. `src/tests/test_behavioral_regress.py`
10. `stages/S1_PLAN.md` — этот файл

### Изменяемые

1. `src/core/energy/calculator.py` — новая сигнатура (`EnergyState`, `dt`),
   сглаженная valence, экспоненциальная утечка стресса
2. `src/core/energy/observer.py` — владеет `EnergyState`
3. `src/core/energy/__init__.py` — re-export новых сущностей
4. `src/core/energy/SPEC.md` — обновить формулы/интерфейс
5. `src/telemetry/models.py` — 8 новых полей
6. `src/telemetry/logger.py` — новые параметры `log()`
7. `src/telemetry/SPEC.md` — обновить
8. `src/host/wiring.py` — `CMCPipeline.tick(u, precision, dt)`; `active_tags`/`reflex_tags`
9. `src/host/loop.py` — `clock_mode`, `dt` от часов, `PrecisionEstimator`,
   ресурсы, guard, drift
10. `src/host/sources.py` — `ResourceProvider` в дефолтном наборе (bus_dim 12→14)
11. `src/config/params.py` — новые параметры (см. ниже)
12. `src/__main__.py` — `--clock-mode`
13. `src/host/SPEC.md` — обновить
14. `src/config/SPEC.md` — обновить
15. `src/tests/test_energy_calculator.py`, `test_energy_observer.py`,
    `test_integration_*`, `test_telemetry_*`, `test_host_*`, `test_config.py` —
    адаптация под новые сигнатуры

## Зависимости

- **Внешние:** `numpy`; stdlib `resource`, `time`, `collections.deque`.
- **Внутренние:** energy ← cmc (через wiring); telemetry ← energy;
  host ← всё.

## Порядок реализации

### Шаг 1. Energy core: EnergyState + calculator

1. `EnergyState(f: float=0, stress: float=0, valence: float=0)` в `models.py`.
2. `FreeEnergyCalculator.__init__(stress_leak_per_sec, valence_tau, gamma_base)`.
3. `compute(error, precision, state, dt)`:
   - валидация shape; `dt > 0` → иначе ValueError;
   - `f = 0.5·Σγe²` (пусто → 0.0);
   - `valence_raw = −(f − state.f)/dt`; EMA: `a = 1−exp(−dt/valence_tau)`;
   - `stress = state.stress·exp(−λ·dt) + f·dt`;
   - `gamma = mean(precision)`.
4. Тесты: формула F; valence сглажена; stress-утечка; `dt` разный → та же
   кривая при synthetic-масштабе; пустые; clip precision.

### Шаг 2. Energy shell: Observer

1. `EnergyObserver` хранит `_state: EnergyState`.
2. `observe(error, precision, dt)` → `compute`, обновить state, sink.
3. Тесты: state переносится между вызовами; sink вызывается.

### Шаг 3. Precision

1. `inverse_variance(samples, eps, gamma_max)` — чистая.
2. `PrecisionEstimator(dim, window, eps, gamma_max)` — deque + `update(u)`.
3. Тесты: константа → max γ; шумный канал → γ ниже; окно ограничено;
   первый сэмпл; purity `inverse_variance`.

### Шаг 4. Guards + Drift

1. `HostIntegrityError`; `check_finite(result)`.
2. `DriftDetector(f_threshold, stress_threshold, hold_ticks).update(result)`.
3. Тесты: NaN/inf → raise; норма → ok; drift срабатывает после hold.

### Шаг 5. Telemetry

1. `TelemetryEvent`: +`tick, gamma, active_tags, reflex_tags, bus_dim,
   latency_ms, rss_mb, drift` (15 полей).
2. `TelemetryLogger.log(...)` — новые параметры с дефолтами.
3. Тесты: сериализация 15 полей; NaN → ValueError; обратная совместимость.

### Шаг 6. ResourceProvider

1. `ResourceMeter` (`record_tick`, `last_latency_s`, `current_rss_mb`).
2. `ResourceProvider` (dim=2, intero, severity=max(норм)).
3. Тесты: норма → низкий severity; перегруз латентности → severity высокий;
   severity ≥ 0.9 → is_reflex; RSS читается.

### Шаг 7. Wiring: dt + tags

1. `CMCPipeline.tick(u, precision, dt)`.
2. `CMCEnsemble.last_errors` (свойство) для посегментной активности.
3. `_tags_above_threshold(errors, segments, threshold)` — чистая функция:
   `active_tags` по карте сегментов.
4. `reflex_tags` из `bus.last_signals`.
5. Тесты: tags считаются корректно; dt пробрасывается.

### Шаг 8. Loop: time base + ресурсы + guard + drift

1. `clock_mode` (`synthetic`/`wall`), `paced`; `dt` от часов.
2. `PrecisionEstimator` → `precision`; `precision_mode` (`variance`/`ones`).
3. Ресурсы: `perf_counter` вокруг тика → meter → `ResourceProvider`.
4. Guard после тика; `DriftDetector` → `drift`.
5. Телеметрия: новые поля заполняются.
6. Тесты: synthetic детерминирован; wall `dt>0`; guard роняет прогон;
   `ones` baseline работает.

### Шаг 9. Config + CLI

1. `EnergyConfig`: `stress_leak_per_sec`, `valence_tau` (вместо `stress_decay`).
2. `HostConfig`: `clock_mode`, `paced`, `precision_window`, `tick_budget_ms`,
   `rss_budget_mb`, `drift_f_threshold`, `drift_stress_threshold`.
3. `--clock-mode` в CLI.
4. Тесты валидации.

### Шаг 10. Адаптация тестов + регресс

1. Обновить существующие тесты под новые сигнатуры.
2. `test_behavioral_regress.py`: C1–C6 + инварианты + `behavioral_fingerprint`.
3. Обновить SPEC/README/SPECS/BACKLOG.

## План тестов

| Тест | Покрытие | Инвариант |
|---|---|---|
| `test_calculator_dt_scaling` | valence/stress от dt | зависит от секунд |
| `test_valence_smoothing` | EMA valence | дребезг убран |
| `test_stress_leak` | экспоненциальная утечка | затухает |
| `test_inverse_variance_*` | γ = 1/var | purity, clip |
| `test_precision_estimator_window` | окно | ограничено |
| `test_check_finite_raises` | NaN/inf | HostIntegrityError |
| `test_drift_detector_*` | порог + hold | флаг |
| `test_telemetry_fields` | 15 полей | сериализация |
| `test_resource_provider_*` | латентность/RSS | severity, reflex |
| `test_tags_above_threshold` | карта сегментов | CSV |
| `test_loop_synthetic_deterministic` | synthetic | replay |
| `test_loop_wall_dt_positive` | wall | dt>0 |
| `test_behavioral_c1..c6` | сценарии | инварианты организма |

## Заметки

- **Обратная совместимость сигнатур**: `build_cmc_pipeline` принимает
  `energy_config`/`precision_estimator` опционально; старые вызовы обновляются
  в одном PR (breaking change внутри S1 допустим — код не в проде).
- **bus_dim 12 → 14**: `ResourceProvider` добавляет 2; тесты, где хардкод 12,
  обновить.
- **`last_errors`**: хранить последний `EnsembleOutput.errors` в ensemble
  (или отдавать из `tick`) — минимальная правка.
- **`ResourceProvider` — недетерминирован**: в behavioral-тестах использовать
  fake `ResourceMeter` (инъекция).
- **`ones` baseline**: держим как фоллбэк FEP (ADR-0005 §4) для сравнения
  на воротах.
- **Миграция параметра**: `stress_decay` (per-tick) → `stress_leak_per_sec`
  (per-sec); маппинг дефолта: `0.99/тик @100Гц` ↔ `λ≈1.0/с`.
