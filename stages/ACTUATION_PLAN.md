# ACTUATION_PLAN — Секвенирование актуаций (сложные реакции)

План спринта: научить хост **сложной реакции** — одной цели, достигаемой
**несколькими актуациями** (в т.ч. не-текстовыми, через MCP), и короткой
цепочкой из 2–3 шагов. Кандидат на стадию **S8** и на **ADR-0012**
(секвенирование актуаций); до кода — ADR + модульные SPEC (CONSTITUTION §5).

Предпосылки: S3 (речь), S4 (policy, `CapabilityGate`), S6 (MCP-зондирование,
`ProbeEffector`, `AffordanceMap`), S7 (b-тест, §7). Замыкает долг
`[INT][action] ACTION-вид (исполнители)` (BACKLOG, ADR-0011 §2) и открывает
`MCP` как **действие**, а не только как эпистемическое зондирование.

> **Почему отдельный спринт.** Речь и зондирование сейчас — две несвязанные
> ветки актуации с разными решателями (`select_action` vs `select_affordance`).
> Секвенирование — не «ещё один эффектор», а **слой над эффекторами**
> (executive), которого в архитектуре нет. Он достаточно крупный, чтобы не
> примешивать его к S5/S6.

## Что решаем (из обсуждения)

### 1. Representation — Behavior Tree сразу

Берём **BT**, а не teleo-reactive (T-R). Обоснование: T-R — вырожденный BT
(priority `Fallback` из листьев «условие→действие»), а миграция T-R → BT
механическая. Ключевой аргумент: **`Running`** (действие длится > 1 тика) и
**память бегущего узла** — самая дорогая часть BT, и она **нужна нам
независимо** из-за асинхронности MCP. После этого BT добавляет лишь
`Sequence` + обход дерева.

- **Минимальный словарь узлов:** `Sequence`, `Fallback`, листья
  `Action`/`Condition`; статусы `Running`/`Success`/`Failure`. `Parallel` и
  декораторы (retry/invert/timeout) — **не** вводим до реального кейса.
- **Tick с корня каждый цикл** (closed-loop): реактивность by construction;
  `Running` позволяет преемптить текущий лист более приоритетным.
- Литература: Iovino et al., arXiv 2405.16137 (BT > FSM/HFSM по модульности и
  реактивности; CC=1); Colledanchise & Ögren; backchained BT (arXiv 2101.01964).

### 2. Closed-loop

План живёт как **активная подцель + правило переключения**, перерешаемое
каждый тик, а не как застывший список. Наш tick — reactive-слой; deliberative
(генератор) перестраивает подцель при изменении состояния. Родственно
teleo-reactive (Nilsson 1994), situated automata (Kaelbling 1990),
universal plans (Schoppers 1987).

### 3. Ограничение «сверху» — горизонт обязательства

Не «длина плана ≤ N вообще», а **receding horizon**: за один deliberative-цикл
коммитимся на короткую глубину (2–3), дальше перерешаем. Защита от
**chattering** (Iovino et al.: бесконечное переключение узлов при неудачном
порядке предусловий). Опора: bounded rationality (Simon), resource-bounded
agents (Zilberstein 1996; Russell & Wefald), span-of-control в RCS/4D-RCS
(Albus).

### 4. Единый контракт «текст vs тул» — общий статус, разные payload

Речь и тул сводим к одному интерфейсу **на уровне статуса и результата**, а
payload остаётся разным (речь семантична/дешёва; тул даёт данные):

```python
class ActuationKind(Enum): SPEAK; INVOKE_TOOL

@dataclass(frozen=True)
class Actuation: kind; goal; payload            # payload: текст | (tool, args)
@dataclass(frozen=True)
class ActuationResult: status; data             # status: Running|Success|Failure|Preempted
class Effector(Protocol):                       # Shell-реализация
    def start(self, act, *, gate) -> Handle      # асинхронный старт
```

`SpeechController` и `ProbeEffector` становятся двумя эффекторами за общим
контрактом. Идиома — ROS Action Server (goal/feedback/result/**preempt**),
лист BT (Running/Success/Failure), option (Sutton/Precup/Singh 1999).

## Контракт действия: кондишены (решение)

**Кондишены — транскрипция закономерностей, а не авторская политика.** Это
снимает возражение «ещё один способ запрограммировать поведение»:
закономерность **опровержима против мира**, предпочтение — нет.

| Слой | Природа | Где живёт |
|---|---|---|
| Цель / preference | наше решение (ADR-0008) | `Preferences` |
| Жёсткий кондишен | закономерность мира (опровержимо) | `Guard` |
| Мягкий кондишен | видовая склонность («сложилось так») | матрица видовых предпочтений |
| Когда/в каком порядке | **выводится** из цели + закономерностей | генератор |

### Форма: градуированные именованные факты

Факт — чистая функция `состояние → [0, 1]`, именованная, по одной
ответственности, из **закрытого словаря** (комбинировать можно только
объявленные факты). Один механизм обслуживает жёсткое и мягкое:

- **жёсткий precondition** → `Guard`: `факт ≥ порог` → узел-`Condition`,
  гейтит (закономерность мира);
- **мягкий precondition** → `Regularity`: степень факта → **стоимость**
  действия, не гейт (видовая склонность);
- **мягкий postcondition** → вероятность успеха → **порядок/скидка** веток;
- **postcondition (эффект)** → символьный (`факт := значение`) или данные
  (`payload → в шину`).

Мягкие регулярности живут в **версионируемой матрице видовых предпочтений**
рядом с `Preferences` — меняются осознанно, как «генетика вида».

**Отклонено:** полноценный PDDL / логика первого порядка (нужен символьный
мир, которого нет; огрубляет непрерывное состояние; ломает replay) и
свободные предикаты-строки (неограниченное авторство = замаскированное
программирование поведения).

**Коллизия имён:** `Precondition` уже занят в `behavioral_chain`
(born/primed/matured). Для предусловий действий используем `Guard` (жёсткое)
и `Regularity` (мягкое).

**Критерий честности:** жёсткий кондишен должен быть опровержим против мира
(тул *действительно* требует сети?); мягкий — нет, он видовая склонность,
поэтому идёт в стоимость, а не в запрет. Эффект наблюдаем в телеметрии
(открывает путь к индукции кондишенов из опыта в зрелой версии).

## Асинхронность (обязательна)

Без async тики «ничего не стоят»: реальный MCP-вызов длится секунды и не
может блокировать `step_once`.

- **Тик не блокируется:** эффектор возвращает **handle/future**, не результат.
- **Лист → `Running`:** пока future не завершён, подцель бежит, дерево
  тикается, реактивный слой живёт.
- **Завершение → в шину:** результат подаётся в сигнальную шину на следующем
  тике (замыкает контур — важно для развития, а не скрипта).
- **Преемпция через gate:** право отозвано (`revoke`) → долгий шаг
  прерывается (`Preempted`), а не «доигрывается вслепую».
- **Детерминизм/replay (§2.8):** порядок/тайминг завершения future
  недетерминированы → для тестов нужен **детерминированный планировщик
  завершений** (аналог `DeterministicMeter`). Самое коварное место спринта.

## Генерация vs авторство

Дерево **не пишем руками** — оно **выводится** backward chaining от цели по
`postconditions` (достигает цель?) и `preconditions` (что должно быть
истинно?) действий, до уже истинного. Родственно backchained BT с гарантиями
сходимости.

- **Авторство** (избегаем): руками писать «если uncertainty > 0.7 → спросить
  имя», «сначала search, потом speak» = программирование поведения
  (`INTENT.md` §2).
- **Генерация** (берём): описываем только факты о действиях (guard/effect);
  задаём цель; алгоритм собирает цепочку. Поведение **выводится**.
- **Runtime**, а не compile-time: каждый цикл бэкчейним от активной цели по
  текущему состоянию (closed-loop; цель может стать неактуальной).
- **LLM — только преобразователь аргументов/формулировки**, не планировщик
  (ADR-0007). Аргументы тула — выход LLM → **мемоизируются в трассу**
  (replay воспроизводит записанные аргументы, не переспрашивая модель).

**Что остаётся нашим** (честный остаток): цель/предпочтения; *какие*
закономерности транскрибировали (можем ошибиться — это неверная модель мира,
опровержимая и чинибельная); сам алгоритм генератора.

## Файлы

### Новые

1. `src/core/actuation/` — `models.py` (`Fact`, `Guard`, `Regularity`,
   `Effect`, `Actuation`, `ActuationKind`, `ActuationStatus`,
   `ActuationResult`, `Leaf`, `Sequence`, `Fallback`, `Node`, `Goal`),
   `compute.py` (`evaluate_fact`, `guard_holds`, `regularity_cost`,
   `order_children`, `backward_chain`), `SPEC.md`, `README.md`
2. `src/host/executor.py` — `ActuatorExecutor` (Shell): tick BT над `HostLoop`,
   async-планировщик завершений, преемпция через gate, результат → в шину
3. `src/host/effectors.py` — `Effector` Protocol + адаптеры (`SpeechEffector`
   над `SpeechController`, `ToolEffector` над `ProbeEffector`)
4. `src/config/actuation.py` — `ActuationConfig`; матрица видовых предпочтений
5. `src/tests/test_actuation_facts.py`
6. `src/tests/test_actuation_bt.py`
7. `src/tests/test_actuation_generator.py`
8. `src/tests/test_actuation_executor.py`
9. `src/tests/test_actuation_async.py`

### Изменяемые

1. `src/mcp/probe.py` — `Affordance` расширяется контрактом `guard`/`effect`
   (сейчас: только `name`/`category`/`reversible`/`dim`)
2. `src/host/probe.py` — `ProbeEffector` реализует `Effector`-контракт
   (`start`/handle вместо синхронного `probe`)
3. `src/speech/controller.py` — `SpeechController` за `Effector` (адаптер)
4. `src/host/gate.py` — преемпция (revoke → `Preempted`); без смены правил
5. `src/host/loop.py` — wiring executor, телеметрия актуаций
6. `src/telemetry/models.py` + `logger.py` — поля актуаций (шаги, статусы,
   tool-вызовы, преемпции)
7. `src/config/params.py`, `src/config/__init__.py` — `ActuationConfig`
8. `src/host/behavioral_chain.py` — звено 4 обобщается (см. §7)
9. `src/host/SPEC.md`, `src/mcp/SPEC.md`, `src/speech/SPEC.md`, `src/config/SPEC.md`
10. `src/__main__.py` — флаг включения секвенирования
11. `BUILD_ROADMAP.md`, `VALIDATION.md`, `SPECS.md`, `BACKLOG.md`, `README.md`
    — синк; `adr/0012-*.md`

## Зависимости

- **Внешние:** новых нет (stdlib + `numpy`; `mcp` уже есть).
- **Внутренние:** `actuation` ← `numpy`, `policy` (Action), `mcp.probe`
  (Affordance) — **только через Protocol/структурные типы** (без цикла);
  `host.executor` ← `actuation`, `gate`, `loop`, `effectors`;
  `effectors` ← `speech`, `host.probe`; `config.actuation` ← `policy`
  (Preferences-совместимость).

## Порядок работ

**Контракт → async → BT-исполнитель → генератор → §7.** Контракт и async —
фундамент; исполнитель — поверх них; генератор — последним (без контракта ему
не из чего строить).

### Шаг 0. ADR-0012 + SPEC

1. ADR-0012: representation = BT (минимальный словарь), контракт
   `Guard`/`Regularity`/эффекты, единый `Effector` со статусом, async,
   runtime-генератор, горизонт. Append-only.
2. Модульные SPEC (`core/actuation`, `host/executor`, `config/actuation`) +
   README до кода (CONSTITUTION §5.1).

### Шаг 1. Core: факты и кондишены

1. `Fact` (frozen): имя из закрытого словаря; `evaluate_fact(fact, state) -> float`
   ∈ [0, 1] (чистая).
2. `Guard(fact, threshold)`; `guard_holds(guard, state) -> bool` (чистая).
3. `Regularity(fact, weight)`; `regularity_cost(regularities, state) -> float`
   (чистая; мягкое → стоимость).
4. Тесты: границы, пустой вход, детерминизм, монотонность, закрытый словарь
   (неизвестный факт → ошибка).

### Шаг 2. Core: эффекты и действия

1. `Effect` — символьный (факт-дельта) или данные (dim payload).
2. `Actuation(kind, goal, payload)`; `ActuationKind`; `ActuationStatus`
   (`Running`/`Success`/`Failure`/`Preempted`); `ActuationResult(status, data)`.
3. `Affordance` расширяется `guard`/`effect` (в `mcp.probe`), совместимо
   (дефолты → прежнее поведение).
4. Тесты: валидация payload/status; обратная совместимость `Affordance`.

### Шаг 3. Core: BT (узлы, tick)

1. `Node` (Protocol): `tick(state) -> status`; листья `Leaf`
   (`Condition`/`Action`), `Sequence`, `Fallback` (priority).
2. Tick с корня; `Running` помнит бегущего ребёнка; преемпция приоритетом.
3. `order_children(children, state)` — сортировка по `regularity_cost`
   (мягкие предпочтения), детерминированный тай-брейк.
4. Тесты: Sequence/Fallback семантика; Running не перезапускает лист;
   преемпция; порядок по стоимости; детерминизм; отсутствие chattering на
   лимите горизонта.

### Шаг 4. Core: runtime-генератор

1. `backward_chain(goal, affordances, state, *, max_depth) -> Node` — от цели
   по `effect`/`guard`; упор в уже истинное; `max_depth` (горизонт 2–3).
2. Runtime-вызов каждый цикл от активной цели (closed-loop).
3. LLM не участвует; аргументы — отдельно (мемоизация в Shell).
4. Тесты: цель достижима → цепочка; недостижима → безопасный отказ;
   `max_depth` обрезает; детерминизм; ablation «убери effect → цепочка
   схлопывается».

### Шаг 5. Shell: эффекторы + executor + async

1. `Effector` Protocol (`start`/handle); `SpeechEffector` над
   `SpeechController`, `ToolEffector` над `ProbeEffector` — за общим контрактом.
2. `ActuatorExecutor`: tick BT над `HostLoop`; `Running` → next tick; результат
   → в шину; преемпция через gate.
3. **Детерминированный планировщик завершений** (для тестов/replay).
4. Тесты: длинный шаг не блокирует тик; завершение приходит на след. тике;
   revoke → `Preempted`; результат в шине; детерминизм при фикс. планировщике.

### Шаг 6. Конфиг, телеметрия, wiring

1. `ActuationConfig` (frozen) + валидация; поле в `HostConfig`; матрица
   видовых предпочтений.
2. Телеметрия: +поля (шаги, статусы, tool-вызовы, преемпции, аргументы).
3. `HostLoop` wiring; CLI-флаг включения; дефолт `enabled=False` →
   S7-совместимость (речь как была).
4. Тесты: конфиг валидация; телеметрия сериализуется; `enabled=False` →
   контур идентичен; CLI-флаг.

### Шаг 7. §7: обобщение звена 4 + ablation

1. Звено 4: «LLM вызван ⇔ решение говорить» → **«каждый шаг актуации прошёл
   gate и завершился ⇔ решён»** (инвариант, выводим точно).
2. Звено 5: форма результата соответствует интенту шага (для тула — против
   ожидаемой схемы).
3. Ablation: выключи генератор → цепочка схлопывается до одной актуации.
4. Fidelity: ось «аргументы тула следуют из подцели».
5. Эталон наблюдаемых: доля многошаговых реакций / доля tool-актуаций —
   наблюдаемые (калибруются, не проверяются).

### Шаг 8. Валидация

1. Интеграционный прогон: сложная реакция (2–3 шага) end-to-end на mock-MCP.
2. Синк `BUILD_ROADMAP.md`, `VALIDATION.md`, `SPECS.md`, `BACKLOG.md`,
   `README.md`.

## План тестов

| Тест | Покрытие | Инвариант |
|---|---|---|
| `test_actuation_facts_*` | Core факты/guard/regularity | границы, закрытый словарь |
| `test_actuation_bt_*` | Core BT | Sequence/Fallback, Running, преемпция |
| `test_actuation_generator_*` | Core генератор | цепочка, горизонт, отказ |
| `test_actuation_executor_*` | Shell executor | шаг не блокирует тик, gate |
| `test_actuation_async_*` | Shell async | завершение, преемпция, детерминизм |
| `test_actuation_config_*` | конфиг | валидация |
| `test_telemetry_actuation_*` | телеметрия | сериализация |
| `test_actuation_disabled_compat` | совместимость | S7 идентичен |
| `test_actuation_ablation_generator` | §7 | цепочка схлопывается |

## Заметки

- **FC/IS (ADR-0004):** факты, BT-узлы, генератор — Core (чистые);
  executor, эффекторы, async — Shell.
- **LLM не судит и не планирует (ADR-0007):** только аргументы/формулировка.
- **Fail-safe:** сбой шага → прервать остаток цепочки, не пропускать
  (пропуск = молчаливое выполнение «в обход»). HITL — **на шаге**, не оптом.
- **Детерминизм (§2.8):** структура дерева = причинная трасса; аргументы LLM
  мемоизируются; async-завершения детерминированы в тестах.
- **Границы плана:** горизонт 2–3 — защита от chattering и от «агента»;
  не реактивный агент на 20 шагов.
- **Не трогаем финализированные S7-тесты.**
- **Reference:** `CapabilityGate`, `AffordanceMap`/`ProbeEffector`,
  `SpeechController`, `DeterministicMeter` (образец детерминизма).

## Открытые вопросы (до кода)

1. **Форма `Effect`:** символьная факт-дельта vs данные-payload — совмещать
   оба или начать с одного?
2. **Планировщик завершений:** виртуальные часы vs очередь завершений — что
   проще для replay.
3. **`Parallel`:** точно ли не нужен в первой версии (или нужен для
   «говорить, пока идёт тул»)?
4. **Мемоизация аргументов:** в телеметрию или в отдельную трассу replay.

## Статус

План зафиксирован (обсуждение 2026-10-08). **Не реализован.**
Синхронизирован с: `BUILD_ROADMAP.md` (спина S1–S8 + §1/§3),
`BACKLOG.md` (секция «S8. Секвенирование актуаций»),
`SPECS.md` (запланированные модули `core/actuation`/`host/executor`/
`config/actuation` + порядок стадий), `VALIDATION.md` (§7.10 — планируемое
расширение), `README.md` (навигация + статус). Ждёт готовности начать
(шаг 0 — ADR-0012 + SPEC).
