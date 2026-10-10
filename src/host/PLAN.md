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

## S8 этап 6: эффекторы + executor (Shell)

### Файлы

1. `src/host/effectors.py` — `Effector` Protocol, `DeferredEffector`,
   `ToolEffector`, `SpeechEffector`
2. `src/host/executor.py` — `ActuatorExecutor`, `ExecutorOutcome`, `node_at`
3. `src/tests/test_actuation_executor.py` — эффекторы + executor

### Порядок

1. `Effector` Protocol (`goal`/`start`/`poll`/`preempt`) + `WorkFn`.
2. `DeferredEffector`: синхронная работа, отложенная на `latency_ticks`
   (не блокирует тик); `poll` → `Running` → результат (кэш) → `Preempted`.
3. `ToolEffector` над `ProbeEffector` (через gate); `SpeechEffector` над
   инъецированным `speak`. Отказ/сбой → `Failure`.
4. `ActuatorExecutor`: опрос эффекторов → `action_status`; tick BT; разрешение
   бегущего листа по `TickMemory.running_path`; преемпция (смена goal →
   `preempt`); старт нового; `done` без переигрывания.
5. `ExecutorOutcome`: статус, `running_goal`, `impatience`, `completed` (→ шина).
6. Тесты: нет блокировки тика; завершение на след. тике; `preempt` → `Preempted`
   → `Failure`; результат в `completed`; нетерпение растёт; `done`; детерминизм.

### Зависимости

`host.executor` ← `core.actuation`, `host.effectors`; `host.effectors` ←
`host.probe` (ProbeEffector), `core.actuation` (Actuation/Result). Без цикла.

### Заметки этапа 6

- **FC/IS:** эффекторы/executor — Shell (I/O и время); BT/генератор — Core.
- **Планировщик завершений** — детерминированный (`latency_ticks`), аналог
  `DeterministicMeter`; реальные вызовы секундами — этап 7 (нетерпение в шину).
- **Преемпция:** смена бегущего `goal` → `preempt` старого эффектора; прерванный
  шаг **не** успешен (fail-safe).
- **Wiring в loop/телеметрия** — этап 7, не здесь (этот этап самодостаточен).
