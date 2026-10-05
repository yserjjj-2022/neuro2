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
| `speech` | SpeechConfig() | речь + LLM (S3) |
| `homeostasis` | HomeostasisConfig() | гомеостаз + throttle (S4) |
| `policy` | PolicyConfig() | выбор действия (S4) |
| `social` | SocialConfig() | ToM (S5) |
| `autonomy` | AutonomyConfig() | автономия (S6) |

### SpeechConfig (S3)

| Поле | Дефолт | Смысл |
|---|---|---|
| `enabled` | False | Включать речь (поднимается `--chat`) |
| `llm_mode` | "auto" | auto (ключ→api, иначе fake), fake, api |
| `f_threshold` | 1.0 | Порог F для инициативы |
| `recall_limit` | 3 | Прецедентов в Intent-Frame |
| `default_register` | "brief" | Речевой режим (длина ответа) |
| `history_turns` | 20 | Глубина истории диалога |
| `temperature` | 0.7 | Температура генерации |
| `style` | "neutral" | Дефолтный стиль (S5 — из характера) |
| `reasoning` | False | Reasoning у LLM (ADR-0007) |

### HomeostasisConfig (S4)

| Поле | Дефолт | Смысл |
|---|---|---|
| `setpoints` | battery/resources/cpu | Сетепоинты каналов |
| `reflex_threshold` | 0.9 | Порог критического сигнала |
| `throttle_k_scale` | 0.5 | Множитель k при throttle |
| `throttle_dt_scale` | 2.0 | Множитель dt при throttle |

### PolicyConfig (S4)

| Поле | Дефолт | Смысл |
|---|---|---|
| `enabled` | True | Включать policy (иначе S3-поведение) |
| `preferences` | Preferences() | Предпочитаемые исходы |
| `mode` | "free" | Режим хоста (game/cooperative/free) |
| `attention_gate` | False | Пред-колоночная γ (проход 2) |

### SocialConfig (S5)

| Поле | Дефолт | Смысл |
|---|---|---|
| `enabled` | False | Включать ToM (иначе S4-совместимость) |
| `match_threshold` | 0.75 | Порог косинуса для узнавания |
| `signature_learning_rate` | 0.2 | Скорость обновления сигнатуры |
| `trust_gain` | 0.1 | Прирост доверия при согласии |
| `trust_decay` | 0.01 | Утечка доверия |
| `conflict_threshold` | 0.6 | Порог рассогласования (Vigilance) |
| `pause_tau_s` | 5.0 | Постоянная нормировки паузы |
| `identify_threshold` | 0.7 | Порог uncertainty для мягкого интента |

### AutonomyConfig (S6)

| Поле | Дефолт | Смысл |
|---|---|---|
| `enabled` | False | Включать автономию (иначе S5-совместимость) |
| `metacog_window` | 50 | Окно наблюдаемых/CSD, тики |
| `csd_variance_gain` | 1.0 | Вес дисперсионной компоненты CSD |
| `csd_autocorr_gain` | 1.0 | Вес автокорреляционной компоненты CSD |
| `csd_warning_threshold` | 0.6 | Порог slowing для warning |
| `reset_soft_threshold` | 0.5 | Порог slowing для SOFT-сброса |
| `consolidate_min_weight` | 0.1 | Порог веса эпизода для pruning |
| `schema_threshold` | 0.8 | Порог косинуса для схем |
| `max_schemas` | 8 | Максимум схем |
| `recency_tau_s` | 86400.0 | Постоянная свежести эпизода, с |
| `explore_threshold` | 0.6 | Порог неопределённости для EXPLORE |
| `factor_learning_rate` | 0.3 | Скорость обновления факторов |

## Пресеты (S7-B)

```python
def baseline() -> HostConfig: ...
def stress() -> HostConfig: ...
def dialogue() -> HostConfig: ...
def autonomy() -> HostConfig: ...
def long_horizon() -> HostConfig: ...
def cooperative() -> HostConfig: ...

def load_preset(name: str, *, override: Path | None = None) -> HostConfig: ...
def available_presets() -> tuple[str, ...]: ...
```

Python-база (типобезопасные фабрики) + TOML-override (stdlib `tomllib`).
Инварианты базы: `clock_mode="synthetic"`, `llm_mode="fake"`,
`embedder_mode="fake"`, фиксированный `seed` → воспроизводимость отпечатков.
Override — частичная перезапись полей/секций; неизвестный ключ, секция не на
dataclass или недопустимое значение → `ValueError` (fail-fast, до прогона).
CLI: `--preset NAME [--preset-file PATH]`.

| Пресет | Что включает |
|---|---|
| `baseline` | всё дефолтное, детерминизм (эталон отпечатка) |
| `stress` | низкие сетпоинты, частый рефлекс, короткий escape hatch |
| `dialogue` | speech+memory+policy+social (fake) |
| `autonomy` | autonomy+selfcontrol+ночная консолидация |
| `long-horizon` | большой `max_ticks`, synthetic (C10) |
| `cooperative` | `mode="cooperative"` + ToM |

## Инварианты

1. Валидация fail-fast: `k<1`, `dt<=0`,
   `precision_mode∉{ones,variance}`, `precision_window<1`, `gamma_max<=0`,
   `precision_eps<=0`, `clock_mode∉{synthetic,wall}`, `time_scale<=0`,
   бюджеты ≤0, `len(columns)<k` → ValueError.
   `MemoryConfig`: `embedder_mode∉{auto,fake,api}`, `embedding_dim<=0`,
   `prior_dim<=0`, `recall_limit<1`, `episode_spike_threshold<0` → ValueError.
   `SpeechConfig`: `llm_mode∉{auto,fake,api}`, `default_register∉{brief,terse,
   normal,story}`, `f_threshold<0`, `recall_limit<1`, `history_turns<0`,
   `temperature∉[0,2]` → ValueError.
   `HomeostasisConfig`: пустые `setpoints`, `reflex_threshold∉[0,1]`,
   `throttle_k_scale∉(0,1]`, `throttle_dt_scale<1` → ValueError.
   `PolicyConfig`: `mode∉{game,cooperative,free}` → ValueError.
   `SocialConfig`: `match_threshold`/`conflict_threshold`/`identify_threshold`
   ∉[0,1], `signature_learning_rate`∉(0,1], `trust_gain`/`trust_decay`<0,
   `pause_tau_s<=0` → ValueError.
   `AutonomyConfig`: `metacog_window<1`, `csd_*_gain<0`,
   `csd_warning_threshold`/`reset_soft_threshold`/`schema_threshold`/
   `explore_threshold`∉[0,1], `consolidate_min_weight<0`, `max_schemas<0`,
   `recency_tau_s<=0`, `factor_learning_rate`∉(0,1] → ValueError.
2. Все dataclass — frozen.
3. DI через `build()`.

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| Значения порогов | Открыто | Стартовые; калибровка S1–S2 |
| `time_scale` дефолт | Решено | 1.0 (жизнь) |
| `gamma_max` | Решено | 10.0 |
