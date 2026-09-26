# PLAN.md — src/config

## Файлы

1. `src/config/params.py` — `EnergyConfig`, `ColumnParams`, `AttractorConfig`, `HostConfig` ✅
2. `src/config/__init__.py` — re-exports ✅
3. `src/config/SPEC.md` — спецификация ✅
4. `src/tests/test_config.py` — тесты валидации и `build()` ✅

## Зависимости

**Внутренние:** `src/core/energy`, `src/core/cmc`, `src/core/attractors`
(только в `build()`-методах, не в данных).

**Стандартная библиотека:** `dataclasses`.

## Порядок реализации (выполнено)

1. `EnergyConfig.build()` → `FreeEnergyCalculator`.
2. `ColumnParams.build(input_dim, state_dim)` → `ColumnConfig`.
3. `AttractorConfig.build(n_tasks)` → `TaskAttractor`.
4. `HostConfig` с вложенными конфигами + валидация в `__post_init__`.
5. Прокинуть в `build_host_loop` (loop.py) и `build_cmc_pipeline` (wiring.py).
6. Добавить `dominance_threshold` в `TaskAttractor.__init__` (был хардкод-дефолт).
7. Тесты `test_config.py`.

## Заметки

- **Размерности из шины**: `ColumnParams.build` вызывается с `bus_dim`,
  поэтому ширина не хардкодится в конфиге.
- **Обратная совместимость**: `build_cmc_pipeline` принимает
  `attractor`/`calculator` опционально — старые вызовы (без конфига) работают.
- **precision_mode**: значение валидируется в `HostConfig` и в `HostLoop`;
  `variance` — задел Фазы 2 (BACKLOG `[Phase2][energy]`).
