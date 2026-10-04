# SPEC.md — S7: HITL-диагностика

Стадия S7 из `BUILD_ROADMAP.md` (§3). Цель — **воспроизводимая и нормируемая
диалоговая диагностика хоста человеком (HITL)**: формальный sensitivity-harness
для чувствительности к ручкам + структурированный, но адаптивный диалоговый
протокол с категориальными вердиктами. Решения — ADR-0010.

Предпосылки: S1–S6 замкнуты (включая проход 1 автономии). Ворота — VALIDATION
§6 (HITL) и §5 (отпечаток прогона). Существующие инструменты: `ChatSession`
(S3), `format_status` (S3), `ControlChannel` (S4), `CapabilityGate` (S4),
`SelfMonitor`/`ResetPlan` (S6), `behavioral_fingerprint` (в тестах).

## Область (входит)

1. **Sensitivity-harness** (S7-A): формальный тест чувствительности к ручкам —
   матрица возмущений + проверка инвариантов направления; Core-fingerprint.
2. **Пресеты** (S7-B): Python-база + TOML-override.
3. **Диагностическая сессия** (S7-C): движок проб с мобильным ветвлением по
   ответам; CLI `--diagnose`; структурированный журнал вердиктов.
4. **Самоотчёт сброса** (S7-D): диалоговый интент «я поплыл, нужен сброс» +
   HITL-подтверждение исполнения через gate.
5. **Операторский протокол** (S7-E): `stages/S7_HITL_PROTOCOL.md` — чек-лист,
   дерево проб, границы.

## Явно НЕ входит

- **Длинный горизонт C10 / ночной цикл / MCP** — S6 проход 2 (автономия).
- **Автоматическая корреляция «диалог → машинный тест»** — офлайн-мост, позже.
- **Оценка «сознания»** — HITL судит только наблюдаемое поведение (манифест §3.И).
- **Реальные LLM/эмбеддеры в пресетах** — базовые пресеты детерминированы
  (fake); реальные — отдельный override, не для сравнения отпечатков.
- **Калибровка порогов** — BACKLOG `[S4][policy]`; S7 проверяет инварианты
  направления, а не точные значения.

## Решения стадии

Зафиксированы в `adr/0010-hitl-diagnostic-protocol-and-sensitivity-harness.md`:

- **A. HITL-диагностика — отдельный этап S7**, не проход 2 S6.
- **B. Две части: формальный тест (числа/ручки) и HITL-диалог (категории).**
- **C. Ручки крутятся только в формальном harness; в диалоге заморожены.**
- **D. Harness хардкодит матрицу, проверяет инварианты направления.**
- **E. Вердикт = категория + комментарий + числовой snapshot (не оценивается).**
- **F. Мобильное ветвление по ответам (не по ручкам).**
- **G. Пресеты — гибрид: Python-база + TOML-override.**
- **H. Harness — и pytest, и CLI.**
- **I. Самоотчёт сброса — интент + HITL-подтверждение (fail-safe deny).**

## 1. Sensitivity-harness (S7-A)

### 1.1 Fingerprint (Core, вынос из тестов)

```python
@dataclass(frozen=True)
class BehavioralFingerprint:
    """Компактный числовой отпечаток прогона (VALIDATION §5).

    Attributes:
        f_profile: Сводка F (mean/std/max).
        reflex_count: Число reflex-событий.
        stress_peaks: Пики стресса.
        active_fraction: Доля активных каналов.
        resource_alarms: Ресурсные алярмы.
        talk_rate: Доля тиков с репликой, [0, 1].
    """

def behavioral_fingerprint(events: Sequence[TelemetryEvent]) -> BehavioralFingerprint:
    """Собрать отпечаток из телеметрии (чистая)."""

def fingerprint_distance(a: BehavioralFingerprint, b: BehavioralFingerprint) -> float:
    """Нормированное расстояние между отпечатками (чистая), >= 0."""
```

`behavioral_fingerprint` переносится из `src/tests/test_behavioral_regress.py` в
`src/host/fingerprint.py` (Core); тесты импортируют оттуда (обратная
совместимость через re-export в тестовом хелпере).

### 1.2 Матрица чувствительности (Core)

```python
@dataclass(frozen=True)
class SensitivityCase:
    """Одна ячейка матрицы: ручка, значения, ожидаемое направление.

    Attributes:
        knob: Имя ручки (например "f_threshold").
        values: Возмущаемые значения.
        metric: Метрика отпечатка (например "talk_rate").
        direction: "nondecreasing" | "nonincreasing" | "bounded".
    """

def build_sensitivity_matrix() -> tuple[SensitivityCase, ...]:
    """Хардкод матрицы возмущений (какие ручки/диапазоны/направления)."""

def check_direction(
    metric_values: Sequence[float], *, direction: str, tol: float = 1e-9
) -> bool:
    """Проверить инвариант направления метрики по значениям ручки (чистая)."""
```

Хардкодятся **ручки и диапазоны**; проверяются **инварианты**:
- `f_threshold ↑` → `talk_rate` не растёт (`nonincreasing`);
- `silent_stress_gain ↑` → `talk_rate` не растёт;
- `explore_threshold ↑` → доля `EXPLORE` не растёт;
- `alert_deviation ↑` → число throttle/инициатив не растёт;
- любой отпечаток: метрики в границах (`bounded`), тот же seed → тот же отпечаток.

### 1.3 Shell + CLI

```python
class SensitivityRunner:
    """Shell: прогон harness-кейсов поверх HostLoop, сбор отпечатков."""
    def run_case(self, case: SensitivityCase, *, seed: int, ticks: int) -> tuple[float, ...]: ...
    def run_all(self, *, seed: int, ticks: int) -> list[SensitivityResult]: ...
```

- pytest: `src/tests/test_sensitivity.py` — гейт в CI.
- CLI: `python -m src --sensitivity` — печатает матрицу (ручка → метрики →
  вердикт направления) для ручного просмотра.

## 2. Пресеты (S7-B)

```python
# src/config/presets.py (Python-база)
def baseline() -> HostConfig: ...
def stress() -> HostConfig: ...
def dialogue() -> HostConfig: ...
def autonomy() -> HostConfig: ...
def long_horizon() -> HostConfig: ...
def cooperative() -> HostConfig: ...

def load_preset(name: str, *, override: Path | None = None) -> HostConfig:
    """База по имени + TOML-override (stdlib tomllib), с валидацией."""
```

- **База** (`presets.py`): детерминированные `HostConfig`-фабрики. Общие
  инварианты базы: `clock_mode="synthetic"`, `llm_mode="fake"`,
  `embedder_mode="fake"`, фиксированный `seed`.
- **Override** (`configs/*.toml`): частичная перезапись полей поверх базы;
  неизвестные ключи/недопустимые значения → `ValueError` (fail-fast).
- Примеры: `configs/dialogue.toml`, `configs/stress.toml`.
- CLI: `--preset NAME [--preset-file PATH]`.

Таблица пресетов:

| Пресет | Что включает | Что проверяет |
|---|---|---|
| `baseline` | всё дефолтное, детерминизм | канонический отпечаток-эталон |
| `stress` | низкие сетпоинты, частый reflex | throttle/escape hatch/гомеостаз |
| `dialogue` | speech+memory+policy+social, fake | тон/уместность/ToM |
| `autonomy` | autonomy+selfcontrol+consolidation | наблюдаемые/сброс/консолидация |
| `long-horizon` | большой `max_ticks`, synthetic | тихий дрейф/устойчивость (C10) |
| `cooperative` | `mode="cooperative"` + Joint Agency | социальный контур |

## 3. Диагностическая сессия (S7-C)

### 3.1 Модель пробы (Core)

```python
class Verdict(Enum):
    MATCHES = "matches"          # соответствует
    PARTIAL = "partial"          # частично
    MISMATCH = "mismatch"        # не соответствует

@dataclass(frozen=True)
class Probe:
    """Одна диагностическая проба.

    Attributes:
        id: Идентификатор (например "S3.tone.01").
        stage: Покрываемая стадия (S3/S4/S5/S6).
        preset: Имя пресета.
        setup: Скрипт событий/сообщений (seed, ticks, messages).
        question: Вопрос наблюдателю (формулировка для человека).
        branches: Правило ветвления Verdict → id следующей пробы.
        fallback: Проба по умолчанию, если ветвление не задано.
    """

@dataclass(frozen=True)
class ProbeResult:
    probe_id: str
    verdict: Verdict
    comment: str
    snapshot: DiagnosticSnapshot
    next_probe: str | None
```

### 3.2 Ветвление (Core)

```python
def next_probe(probe: Probe, verdict: Verdict) -> str | None:
    """Выбрать следующую пробу по категориальному ответу (чистая)."""
```

### 3.3 Диагностический журнал (Shell)

```python
@dataclass(frozen=True)
class DiagnosticSnapshot:
    """Числовой снимок состояния на момент вердикта (не оценивается)."""
    tick: int
    f: float
    valence: float
    stress: float
    gamma: float
    task: str
    partner_trust: float
    partner_uncertainty: float
    metacog_conflict: float
    reset_level: str
    change_kind: str

class DiagnosticSession:
    """Shell: ведёт пробу → снимок → вопрос → вердикт → ветвление → журнал."""
    def snapshot(self) -> DiagnosticSnapshot: ...
    def run_probe(self, probe: Probe) -> ProbeResult: ...
    def run(self, *, start: str, max_probes: int = 0) -> list[ProbeResult]: ...
```

- Ввод/вывод — инъекция (`input_fn`/`output_fn`), как в `ChatSession`.
- Журнал — отдельный JSONL (`--diagnose-log`): `{probe_id, verdict, comment,
  snapshot, next_probe}`. Вердикт привязан к тику/состоянию.
- CLI: `python -m src --diagnose [--preset NAME]`.

### 3.4 Дерево проб (данные)

Дерево проб — данные (`src/host/probes.py` или `configs/probes.toml`): список
`Probe` с ветвлениями. Начальный набор по стадиям:

- **S3 (тон/уместность):** тон следует знаку/величине valence; ответ уместен
  контексту; правка запоминается и влияет на следующий ответ.
- **S4 (воля):** рефлекс срабатывает за 1 тик; escape hatch под удержанным
  throttle; explainability (трасса видна).
- **S5 (социальность):** узнавание по повторной реплике; противоречие
  маркируется гипотезой; вокатив по объявленному имени.
- **S6 (автономия):** наблюдаемые метакогниции видны; сброс триггерится по
  slowing down; консолидация явная и логируемая; самоотчёт (S7-D).

## 4. Самоотчёт сброса (S7-D)

```python
def reset_self_report(*, plan: ResetPlan, task: str) -> str | None:
    """Сформулировать самоотчёт при triggered=True (чистая); None если нет."""
```

- Триггер: `SelfMonitor.observe` → `ResetPlan.triggered` (SOFT/FREEZE).
- Речевой интент `report_reset` (отдельно от `INITIATIVE`).
- Исполнение сброса — через `CapabilityGate` (необратимо → `hitl_token`;
  таймаут → fail-safe deny, ADR-0005 §9).
- `HARD` — аварийная остановка, самоотчёт обязателен, сброс не исполняется
  автоматически.

## 5. Операторский протокол (S7-E)

`stages/S7_HITL_PROTOCOL.md` — развёрнутый документ:

1. **Подготовка:** какой пресет, какой seed, что детерминировано.
2. **Порядок прогонов:** baseline → dialogue → stress → autonomy → cooperative.
3. **Дерево проб:** какие вопросы, в каком порядке, как ветвятся.
4. **Что смотреть:** чек-лист по стадиям (наблюдаемое поведение).
5. **Как нормировать:** категории, snapshot, что сравнивать с эталоном.
6. **Границы:** не оцениваем «сознание»; причинная прослеживаемость вместо
   «ощущения понятности»; антропоморфная самообманка (манифест §3.И).
7. **Мост в машинный тест:** как находка становится новым `test_sensitivity`-
   кейсом.

## 6. Конфигурация и CLI

- `--preset NAME`, `--preset-file PATH` (S7-B).
- `--sensitivity` (S7-A: печать матрицы).
- `--diagnose`, `--diagnose-log PATH` (S7-C).
- `--probes PATH` (S7-C: переопределение дерева проб).

Новых полей `HostConfig` не требуется: диагностика — надстройка над loop.
Возможен `DiagnosticConfig` (frozen) для путей/лимитов.

## Инварианты S7

1. **Ручки не крутятся в диалоге:** диагностическая сессия не меняет
   `HostConfig`; ветвление — только по вердиктам.
2. **Детерминизм базы:** базовый пресет фиксирует seed/synthetic/fake →
   одинаковый отпечаток при одинаковом входе.
3. **Fail-fast override:** неизвестный ключ/недопустимое значение TOML →
   `ValueError` до прогона.
4. **Вердикт привязан к состоянию:** каждый `ProbeResult` несёт `snapshot`.
5. **Категории, не числа:** человек не выставляет числовые оценки.
6. **LLM не судит:** диагностика и harness — детерминированная математика
   (ADR-0007); реальный LLM — только в отдельных прогонах, не в отпечатках.
7. **Fail-safe deny:** исполнение сброса без HITL-токена → отказ.
8. **Обратная совместимость:** без флагов S7 контур S6 идентичен.

## Критерии приёмки (S7)

- [ ] `behavioral_fingerprint` вынесен в Core, тесты импортируют оттуда
- [ ] sensitivity-матрица проверяет инварианты направления
- [ ] harness доступен как pytest-гейт и как CLI `--sensitivity`
- [ ] пресеты: Python-база + TOML-override с fail-fast валидацией
- [ ] диагностическая сессия: проба → snapshot → вердикт → ветвление → журнал
- [ ] дерево проб покрывает S3–S6
- [ ] самоотчёт сброса + HITL-подтверждение (fail-safe deny)
- [ ] операторский протокол написан и пройден вручную
- [ ] без флагов S7 контур S6 идентичен
- [ ] ruff/тесты зелёные; mypy strict для новых Core-модулей

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| S7 отдельно от S6 | Решено | отдельный этап (ADR-0010 §1) |
| Ручки в диалоге | Решено | не крутятся; только в harness (ADR-0010 §3) |
| Формат пресетов | Решено | гибрид Python+TOML (ADR-0010 §6) |
| Шкала вердикта | Решено | 3 категории (ADR-0010 §4) |
| Harness-входы | Решено | pytest + CLI (ADR-0010 §7) |
| Самоотчёт сброса | Решено | интент + HITL-подтверждение (ADR-0010 §8) |
| Набор проб и пороги ветвления | Открыто | уточняются при реализации протокола |
| Автокорреляция «диалог → тест» | Отложено | офлайн-мост, позже |
| Формат дерева проб (Python vs TOML) | Открыто | предварительно Python (`probes.py`) |

## Implementation Notes

1. **FC/IS:** `fingerprint.py` (Core), `presets.py` (Core+Shell),
   `diagnostic.py` (Shell), `probes.py` (данные). Core — чистые функции.
2. **Без новых зависимостей:** TOML — stdlib `tomllib` (Python 3.11+).
3. **Обратная совместимость:** `behavioral_fingerprint` re-export; без флагов
   S7 — контур S6.
4. **Инъекция I/O:** `input_fn`/`output_fn`/`clock` — как в `ChatSession`.
5. **Reference:** `ChatSession` (Shell с инъекцией), `ControlChannel`
   (status/snapshot), `CapabilityGate` (HITL-токен), `SelfMonitor`/`ResetPlan`
   (S6), `behavioral_fingerprint` (тесты).
