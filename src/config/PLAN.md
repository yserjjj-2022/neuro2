# PLAN.md — src/config

## Файлы

1. `src/config/params.py` — `EnergyConfig`, `ColumnParams`, `AttractorConfig`, `HostConfig` ✅
2. `src/config/__init__.py` — re-exports ✅
3. `src/config/SPEC.md` — этот файл ✅
4. `src/tests/test_config.py` ✅

## Зависимости

**Внутренние:** `core/energy`, `core/cmc`, `core/attractors` (в `build()`).
**Стандартная библиотека:** `dataclasses`.

## Порядок реализации (S1, выполнено)

1. `EnergyConfig`: человеческие константы (`stress_leak=0.01`, `valence_tau=1.0`).
2. `HostConfig`: `dt=0.1`, `gamma_max`, `precision_eps`, `time_scale`,
   пороги дрейфа 100/50.
3. Проброс в `build_host_loop` (loop.py).
4. Валидация новых полей.
5. Тесты.

## Заметки

- **Размерности из шины**: `ColumnParams.build(bus_dim, bus_dim)`.
- **Обратная совместимость**: `build_cmc_pipeline` принимает attractor/calculator опционально.
- **Калибровка S1**: `gamma_max=10` (не 1e6), `valence_tau=1.0`, `stress_leak=0.01`.
- **time_scale**: 1.0 = жизнь, >1 = симуляция (ADR-0006).
