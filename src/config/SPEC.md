# SPEC.md — src/config

## Назначение

Единый источник настраиваемых параметров хоста (CONSTITUTION §2.2: все
временные величины и пороги живут здесь, не хардкодятся в логике).

Значения — **стартовые** (подобраны под тесты), не откалиброваны. Калибровка
порогов по телеметрии — Фаза 2/3 (BACKLOG).

## Публичный интерфейс

### EnergyConfig → FreeEnergyCalculator

| Поле | Дефолт | Смысл |
|---|---|---|
| `dt` | 0.01 | шаг интегрирования для валентности |
| `stress_decay` | 0.99 | затухание аллостатического стресса |
| `gamma_base` | 1.0 | γ для пустого входа |

### ColumnParams → ColumnConfig

| Поле | Дефолт | Смысл |
|---|---|---|
| `specialization` | "general" | тег специализации |
| `alpha` | 0.1 | скорость обновления состояния |

`build(input_dim, state_dim)` — размерности берутся из фактической ширины шины.

### AttractorConfig → TaskAttractor

| Поле | Дефолт | Смысл |
|---|---|---|
| `base_dwell` | 5 | жёсткий пол удержания |
| `dwell_slope` | 2.0 | прирост dwell при выигрыше |
| `plasticity_gain` | 0.1 | прирост устойчивости (STP) |
| `basin_threshold` | 0.15 | порог бассейна притяжения |
| `convergence_threshold` | 1e-8 | порог сходимости EMA |
| `dominance_threshold` | 0.3 | порог явного превосходства |

### HostConfig

| Поле | Дефолт | Смысл |
|---|---|---|
| `dt` | 0.01 | шаг loop (0 → без пауз) |
| `max_ticks` | 100 | число тиков (0 → бесконечно) |
| `k` | 2 | победители k-WTA |
| `seed` | 0 | зерно провайдеров |
| `message_dim` | 8 | размерность заглушки сообщения |
| `active_threshold` | 1e-8 | порог активности колонки |
| `precision_mode` | "ones" | "ones" (Фаза 1) / "variance" (Фаза 2) |
| `log_path` | "host_telemetry.jsonl" | JSONL телеметрии |
| `columns` | 3 (tone/rhythm/meaning) | параметры колонок |
| `energy` | EnergyConfig() | параметры energy |
| `attractor` | AttractorConfig() | параметры аттрактора |

## Инварианты

1. **Валидация fail-fast**: `k < 1`, `message_dim <= 0`, `dt < 0`,
   `precision_mode ∉ {ones, variance}`, `len(columns) < k` → `ValueError`.
2. **Immutable**: все dataclass — `frozen`.
3. **DI через `build()`**: модули получают параметры из конфига, не из дефолтов
   конструкторов.

## Open Questions

| Вопрос | Статус | Решение |
|--------|--------|---------|
| Формат (dataclass vs dict/YAML) | Решено | frozen dataclass — типобезопасность, DI |
| Значения порогов | Открыто | Стартовые; калибровка — Фаза 2/3 |
| `variance` precision | Отложено | Задел `precision_mode`; реализация — Фаза 2 |
