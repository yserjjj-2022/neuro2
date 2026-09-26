# PLAN.md — src/host

## Файлы

1. `src/host/sources.py` — провайдеры + `SignalBus` + карта сегментов ✅
2. `src/host/loop.py` — `HostLoop` + `build_host_loop` ✅
3. `src/host/wiring.py` — добавлен `attractor`/`calculator` (DI из config), `writer`, `close()` ✅
4. `src/__main__.py` — CLI + graceful shutdown ✅
5. `src/host/SPEC.md` — этот файл ✅
6. `src/tests/test_host_sources.py` — 39 тестов провайдеров/шины ✅
7. `src/tests/test_host_loop.py` — эксперименты #1–#5 + механика ✅

## Зависимости

**Внутренние:** `src/mcp` (SignalSource/Registry), `src/core/cmc`,
`src/core/attractors`, `src/core/energy`, `src/telemetry`, `src/config`.

**Внешние:** `numpy`. **Стандартная библиотека:** `argparse`, `signal`,
`time`, `dataclasses`, `pathlib`, `logging`.

## Порядок реализации (выполнено)

1. `sources.py`: Protocol + 7 провайдеров + `SignalBus` + `default_providers`.
2. `loop.py`: `HostLoop` (precision/run/step_once/close) + `build_host_loop`.
3. `wiring.py`: DI `attractor`/`calculator`, `writer` в `CMCPipeline`, `close()`.
4. `__main__.py`: argparse → `HostConfig` → `build_host_loop`; SIGINT.
5. Тесты: `test_host_sources.py`, `test_host_loop.py`.
6. `src/config/`: `HostConfig`/`EnergyConfig`/`ColumnParams`/`AttractorConfig`;
   `dominance_threshold` прокинут в `TaskAttractor`.
7. Документация: SPEC/README/BACKLOG.

## Экспериментальный протокол

Запуск → метрики из JSONL → вывод → следующий шаг.

| # | Сценарий | Гипотеза | Критерий |
|---|---|---|---|
| 1 | `ConstantProvider` | EMA сходится | `F→0`, `active→0` |
| 2 | `StepProvider` | хост «вздрагивает» | `F↑`, `valence<0`, затем сходимость |
| 3 | `BatteryProvider` разряд | reflex-контракт | `is_reflex=True` при severity ≥ 0.9 |
| 4 | `NoisyProvider` | стресс копится | `allostatic_stress` растёт |
| 5 | Дефолтный набор | стабильность | N тиков → N событий, без shape mismatch |

## Заметки

- **width scaling**: карта сегментов готова; проекции колонок (`reads`) —
  Фаза 2, без изменения провайдеров.
- **DI конфига**: `build_cmc_pipeline` принимает готовые `attractor`/`calculator`
  (опционально) — старые вызовы без них не ломаются.
- **precision**: `ones` baseline; `variance` явно бросает `NotImplementedError`
  как задел Фазы 2 (BACKLOG `[Phase2][energy]`).
