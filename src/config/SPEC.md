# SPEC.md — src/config

## Назначение

Единый источник настраиваемых параметров хоста (CONSTITUTION §2.2).
Значения — стартовые (калибровка по телеметрии, S1–S2).

## Публичный интерфейс

### EnergyConfig → FreeEnergyCalculator

| Поле | Дефолт | Смысл |
|---|---|---|
| `stress_leak_per_sec` | 0.01 | λ, утечка стресса (настроение, минуты) |
| `valence_tau` | 1.0 | τ, сглаживание valence (эмоция, с) |
| `gamma_base` | 1.0 | γ для пустого входа |

### ColumnParams → ColumnConfig

| Поле | Дефолт | Смысл |
|---|---|---|
| `specialization` | "general" | тег |
| `alpha` | 0.1 | скорость обновления |

### AttractorConfig → TaskAttractor

| Поле | Дефолт |
|---|---|
| `base_dwell` | 5 |
| `dwell_slope` | 2.0 |
| `plasticity_gain` | 0.1 |
| `basin_threshold` | 0.15 |
| `convergence_threshold` | 1e-8 |
| `dominance_threshold` | 0.3 |

### MemoryConfig (S2) → эмбеддер + MemoryStore + MemoryRouter

| Поле | Дефолт | Смысл |
|---|---|---|
| `enabled` | True | Включить память в loop |
| `embedder_mode` | "auto" | auto (ключ→api, иначе fake), fake, api |
| `embedding_dim` | 8 | Размерность fake-эмбеддера / коммуникативного входа |
| `db_path` | "host_memory.db" | Файл БД памяти |
| `episode_spike_threshold` | 1.0 | Порог всплеска F для эпизода |
| `recall_limit` | 1 | Сколько эпизодов извлекать |
| `prior_dim` | 4 | Размерность приора в шине |

API-настройки эмбеддера — в окружении (`.env`), не в `MemoryConfig`:
`EMBEDDER_API_KEY`, `EMBEDDER_BASE_URL`, `EMBEDDER_MODEL`, `EMBEDDER_DIM`.

### HostConfig

| Поле | Дефолт | Смысл |
|---|---|---|
| `dt` | 0.1 | шаг (10 Гц, эмоц. контур, ADR-0006) |
| `max_ticks` | 100 | 0 → бесконечно |
| `k` | 2 | победители k-WTA |
| `seed` | 0 | зерно провайдеров |
| `active_threshold` | 1e-8 | порог активности |
| `precision_mode` | "variance" | γ=1/var или "ones" |
| `precision_window` | 50 | окно дисперсии |
| `gamma_max` | 10.0 | потолок γ |
| `precision_eps` | 1e-6 | регуляризация γ |
| `clock_mode` | "synthetic" | время |
| `paced` | False | спать между тиками |
| `time_scale` | 1.0 | множитель субъективного времени |
| `tick_budget_ms` | 50.0 | бюджет тика |
| `rss_budget_mb` | 1024.0 | бюджет памяти |
| `drift_f_threshold` | 100.0 | порог F дрейфа |
| `drift_stress_threshold` | 50.0 | порог стресса дрейфа |
| `log_path` | "host_telemetry.jsonl" | JSONL |
| `columns` | 3 (tone/rhythm/meaning) | колонки |
| `energy` | EnergyConfig() | energy |
| `attractor` | AttractorConfig() | аттрактор |
| `memory` | MemoryConfig() | память + эмбеддер |

## Инварианты

1. Валидация fail-fast: `k<1`, `dt<=0`,
   `precision_mode∉{ones,variance}`, `precision_window<1`, `gamma_max<=0`,
   `precision_eps<=0`, `clock_mode∉{synthetic,wall}`, `time_scale<=0`,
   бюджеты ≤0, `len(columns)<k` → ValueError.
   `MemoryConfig`: `embedder_mode∉{auto,fake,api}`, `embedding_dim<=0`,
   `prior_dim<=0`, `recall_limit<1`, `episode_spike_threshold<0` → ValueError.
2. Все dataclass — frozen.
3. DI через `build()`.

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| Значения порогов | Открыто | Стартовые; калибровка S1–S2 |
| `time_scale` дефолт | Решено | 1.0 (жизнь) |
| `gamma_max` | Решено | 10.0 |
