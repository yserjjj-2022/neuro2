# PLAN.md — src/host

## Файлы

1. `src/host/sources.py` — провайдеры, `SignalBus`, карта сегментов, `tags_above_threshold` ✅
2. `src/host/wiring.py` — `TickOutcome`, `CMCPipeline` (без I/O) ✅
3. `src/host/resources.py` — `ResourceMeter`, `ResourceProvider` ✅
4. `src/host/loop.py` — `HostLoop`, `build_host_loop(config, meter)` ✅
5. `src/__main__.py` — CLI ✅
6. `src/host/SPEC.md` — этот файл ✅
7. `src/tests/test_host_sources.py` ✅
8. `src/tests/test_host_loop.py` ✅
9. `src/tests/test_host_resources.py` ✅
10. `src/tests/test_behavioral_regress.py` — C1–C6 + фингерпринт ✅

## Зависимости

**Внутренние:** `mcp`, `core/*`, `telemetry`, `config`.
**Внешние:** `numpy`. **Стандартная библиотека:** `argparse`, `signal`, `time`,
`resource`, `dataclasses`, `pathlib`, `logging`, `collections`.

## Порядок реализации (S1, выполнено)

1. Провайдеры + `SignalBus` + карта сегментов + медленный такт.
2. `tags_above_threshold` (чистая).
3. `TickOutcome`; `CMCPipeline.tick(u, precision, dt, segments, reflex_tags)`.
4. `ResourceMeter`/`ResourceProvider` (intero, dim=2).
5. `HostLoop`: `_time_for_tick` (synthetic/wall × time_scale), precision,
   ресурсы, guard, drift, телеметрия.
6. `build_host_loop(config, meter=None)`; `--clock-mode/--paced`.
7. Тесты + поведенческий регресс.

## Экспериментальный протокол (C1–C6)

| # | Сценарий | Критерий |
|---|---|---|
| C1 | Constant | F→0, без NaN |
| C2 | Step | F↑, valence<0, сходимость |
| C3 | Battery | reflex в логе |
| C4 | Noise | стресс копится, без runaway |
| C5 | Default bus | N тиков → N событий |
| C6 | Resources | latency/rss в логе |

## Заметки

- **Телеметрия в loop**: pipeline чистый, loop владеет writer'ом.
- **Инъекция meter**: fake для replay; реальный RSS нестабилен (ADR-0006).
- **bus_dim 14**: resources добавляет 2 канала.
- **Регресс-фингерпринт**: значимые смены valence (|v|>1), не микро-шум.
