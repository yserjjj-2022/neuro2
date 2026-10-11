"""GoalDirector — Shell, замыкающий целевой контур в runtime (S8, проход 2).

Владелец **целевого контура в рабочем цикле** (ADR-0014/0015). Наблюдает снимок
фактов через ``Environment``, формирует цель (``select_goal``), строит/ведёт
цепочку по цели (``backward_chain`` + ``ActuatorExecutor`` через
``HostLoop.tick_actuation``), проверяет достижение новым наблюдением
(``goal_reached``). Core-функции чистые; здесь — I/O и жизненный цикл.

Жизненный цикл (``stages/S8_PASS2_SPEC.md`` §7):

- **Одно действие в полёте:** пока дерево ``RUNNING``, директор его не
  пересобирает; повторный запуск запрещён до терминального статуса.
- **Периодическое наблюдение:** среда переспрашивается после терминального
  статуса; наблюдение продолжается и после исчерпания ``max_attempts`` (иначе
  хост не обнаружит изменение среды). Запрет — на **запуск действий**.
- **Сброс счётчика попыток:** по **значимому** изменению факта цели или явному
  обновлению цели, не по любому изменению снимка.
- **Преемпция ≠ отмена:** смена цели преемптит текущее дерево (новое — вместо
  старого, без отката внешних шагов); отсутствие цели (все исходы достигнуты)
  останавливает запуски.
- **Граница отмены:** преемпция отменяет **ещё не отправленный** вызов (работа
  эффектора ещё не начата) — она не гарантирует остановку уже выполняющейся
  внешней операции. ``DeferredEffector`` выполняет работу синхронно внутри
  ``poll``; настоящая асинхронная отмена — отдельный шаг.

Functional Core / Imperative Shell (ADR-0004): решения (что делать) — чистые
Core-функции; I/O (наблюдение, исполнение) — здесь.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from src.config import OptionDeclaration
from src.core.actuation import (
    ActuationPreferences,
    Goal,
    Guard,
    Node,
    NodeKind,
    NodeStatus,
    Option,
    OptionContext,
    OptionTrace,
    OptionWindow,
    PreferredOutcome,
    backward_chain,
    build_options,
    evaluate_fact,
    goal_reached,
    select_goal,
    select_option,
)
from src.host.executor import ExecutorOutcome
from src.mcp.probe import Affordance, AffordanceMap

# Порог «значимого» изменения факта цели (защита от шума снимка).
_SIGNIFICANCE_EPS = 1e-9


@runtime_checkable
class Environment(Protocol):
    """Источник снимка фактов мира (Shell).

    Снимок — ``Mapping[str, float]`` (имя факта → значение ∈ [0, 1]); неизвестные
    факты деградируют к ``fact.default`` в Core (тотальность).
    """

    def observe(self) -> Mapping[str, float]:
        """Вернуть текущий снимок фактов мира."""
        ...


@runtime_checkable
class _HostLike(Protocol):
    """Минимальный контракт хоста, нужный директору (без цикла импортов)."""

    probe_effector: Any
    executor: Any

    def active_task(self) -> str:
        """Тег активной специализации колонки (трассировка)."""
        ...

    def tick_actuation(
        self, root: Node, facts: Mapping[str, float] | None = None
    ) -> ExecutorOutcome | None:
        """Провести тик секвенирования актуаций (Shell)."""
        ...


@dataclass(frozen=True)
class GoalAttempt:
    """Итог одного тика директора (для телеметрии/тестов).

    Три уровня трассы **разделены** (победитель глобального скоринга ≠ выбранная
    цепочка ≠ фактически запущенный лист — их совпадение не гарантировано):

    Attributes:
        goal: Выбранная цель-исход (None — все исходы удовлетворены/нет исходов).
        reason: Почему цель выбрана/не выбрана (аудит); при провальном терминале
            включает причину отказа (Gate deny / отказ сервиса, §11).
        trace: Трасса **глобального** скоринга окна (кандидаты + выбранная); None —
            не оценивалось на этом тике.
        call_status: Статус последнего шага (``running``/``success``/``failure``).
        observed_after: Снимок фактов после терминального статуса (иначе текущий).
        achieved: Достигнута ли цель по наблюдению (успех цели, не вызова).
        attempts: Число попыток текущей цели.
        chain_goal: Имя корня выведенной цепочки (опция, достигающая конечной
            цели), "" если дерево не строилось.
        planned_first_action: Первый ``Action`` дерева в порядке обхода —
            **структурный план** (что могло бы запуститься), не факт запуска.
            Невыполненная предпосылка оставляет это имя, но запуска нет.
        launched_goal: Идентификатор **фактически запущенного** листа (событие
            запуска от исполнителя), "" если запуска не было.
    """

    goal: Goal | None
    reason: str
    trace: OptionTrace | None
    call_status: str
    observed_after: Mapping[str, float]
    achieved: bool
    attempts: int
    chain_goal: str = ""
    launched_goal: str = ""
    planned_first_action: str = ""


def _first_action(node: Node) -> Node | None:
    """Первый ``Action``-лист в порядке обхода дерева (чистая).

    Генератор строит ``Sequence(subtree, action)`` — исполняемый лист идёт после
    подцели, поэтому при успешном обходе фактически запускается **первый**
    ``Action``. Возвращает None, если дерево — чистый ``Condition`` (безопасный
    отказ): тогда запуска нет.

    Это **структурный план** (что могло бы запуститься), а не факт запуска:
    невыполненная предпосылка (``Condition`` провалился) останавливает обход
    раньше, поэтому фактический запуск берётся из события исполнителя
    (``ExecutorOutcome.started_goal``).

    Args:
        node: Корень выведенного дерева.

    Returns:
        Первый ``Action``-узел или None.
    """
    if node.kind is NodeKind.ACTION:
        return node
    for child in node.children:
        found = _first_action(child)
        if found is not None:
            return found
    return None


def build_goal_window(
    affordances: AffordanceMap | Sequence[Affordance],
    declarations: Sequence[OptionDeclaration] = (),
) -> OptionWindow:
    """Построить окно опций и наложить объявленную семантику (чистая).

    Окно собирается из аффордансов (``build_options``); каждая
    ``OptionDeclaration`` накладывается на опцию **по устойчивому ``option_id``**
    (``replace``), порядок окна сохраняется. Валидация fail-fast: каждый
    ``option_id`` разрешается в существующую опцию; дубликаты запрещены.

    Args:
        affordances: Карта аффордансов (или последовательность аффордансов).
        declarations: Объявленная семантика опций (effect/guard).

    Returns:
        OptionWindow с наложенными эффектами/гардами.

    Raises:
        ValueError: Если ``option_id`` декларации нет в окне или дубликат.
    """
    tools = (
        affordances.affordances
        if isinstance(affordances, AffordanceMap)
        else tuple(affordances)
    )
    window = build_options(tools)
    by_id = {decl.option_id: decl for decl in declarations}
    if len(by_id) != len(declarations):
        raise ValueError("duplicate option declaration id")
    options: list[Option] = []
    for option in window.options:
        declaration = by_id.get(option.id)
        if declaration is not None:
            option = replace(option, effect=declaration.effect, guard=declaration.guard)
        options.append(option)
    unknown = [
        decl.option_id for decl in declarations if window.find(decl.option_id) is None
    ]
    if unknown:
        raise ValueError(f"option declarations not in window: {unknown}")
    return OptionWindow(tuple(options))


class GoalDirector:
    """Shell: ведёт целевой контур по тикам (S8, проход 2).

    Attributes:
        outcomes: Предпочтительные исходы (что считаем желательным).
        environment: Источник снимка фактов мира.
        preferences: Веса скорера окна.
        max_attempts: Ограничение повторов попытки (>= 1).
        deadline_ticks: Тайм-аут цепочки → преемпция → FAILURE.
        cooldown_ticks: Пауза между попытками (анти-дребезг).
        max_depth: Горизонт генератора.
    """

    def __init__(
        self,
        *,
        loop: _HostLike,
        outcomes: Sequence[PreferredOutcome],
        declarations: Sequence[OptionDeclaration] = (),
        environment: Environment,
        preferences: ActuationPreferences | None = None,
        max_attempts: int = 3,
        deadline_ticks: int = 8,
        cooldown_ticks: int = 1,
        max_depth: int = 3,
    ) -> None:
        """Создать директора (окно мемоизируется из аффордансов хоста).

        Args:
            loop: Хост (``tick_actuation``/``active_task``/``probe_effector``).
            outcomes: Предпочтительные исходы.
            declarations: Объявленная семантика опций (effect/guard).
            environment: Источник снимка фактов.
            preferences: Веса скорера (None → дефолт ``ActuationPreferences``).
            max_attempts: Ограничение повторов (>= 1).
            deadline_ticks: Тайм-аут цепочки в тиках (>= 0; 0 → без тайм-аута).
            cooldown_ticks: Пауза между попытками (>= 0).
            max_depth: Горизонт генератора (>= 1).

        Raises:
            ValueError: Если ``max_attempts < 1``, тики отрицательны или
                декларация ссылается на отсутствующую опцию.
        """
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        if deadline_ticks < 0:
            raise ValueError(f"deadline_ticks must be >= 0, got {deadline_ticks}")
        if cooldown_ticks < 0:
            raise ValueError(f"cooldown_ticks must be >= 0, got {cooldown_ticks}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        self.loop = loop
        self.outcomes = tuple(outcomes)
        self.declarations = tuple(declarations)
        self.environment = environment
        self.preferences = (
            preferences if preferences is not None else ActuationPreferences()
        )
        self.max_attempts = max_attempts
        self.deadline_ticks = deadline_ticks
        self.cooldown_ticks = cooldown_ticks
        self.max_depth = max_depth
        probe = loop.probe_effector
        affordances = getattr(probe, "affordances", None) or AffordanceMap()
        self._window = build_goal_window(affordances, self.declarations)
        # Состояние жизненного цикла.
        self._goal: Goal | None = None
        self._tree: Node | None = None
        self._running = False
        self._attempts = 0
        self._exhausted = False
        self._cooldown = 0
        self._elapsed = 0
        self._goal_value: float | None = None
        self._trace: OptionTrace | None = None
        self._call_status = ""
        self._chain_goal = ""
        self._planned_first_action = ""
        self._launched_goal = ""

    @property
    def active(self) -> bool:
        """Есть ли выбранная цель или действие в полёте (для подавления S6)."""
        return self._goal is not None or self._running

    @property
    def window(self) -> OptionWindow:
        """Мемоизированный снимок окна опций (с наложенными декларациями)."""
        return self._window

    def tick(self, tick: int) -> GoalAttempt:
        """Провести тик целевого контура (наблюдение → цель → исполнение).

        Порядок важен (§7): **пока есть активная попытка**, сначала разрешается её
        жизненный цикл — проверка, не достигнута ли **её** цель новым наблюдением,
        опрос исполнения, фиксация завершения. Только затем выбирается следующая
        цель. Иначе достигнутая цель могла бы «исчезнуть» (её сменил бы
        ``select_goal``) до обработки терминального результата, и достижение не
        получило бы собственного события.

        Args:
            tick: Номер тика хоста (для трассы; логика от него не зависит).

        Returns:
            GoalAttempt — выбранная цель, причина, трассы, статус, наблюдение.
        """
        facts = dict(self.environment.observe())

        # 1. Активная попытка → сначала разрешаем её жизненный цикл.
        if self._running and self._goal is not None:
            active = self._goal
            # 1a. Цель уже достигнута новым наблюдением — в т.ч. если вызов изменил
            #     среду на прошлом тике, до терминального poll. Останавливаем дерево
            #     и фиксируем успех **этой** цели, а не молча меняем её другой.
            if goal_reached(active, facts):
                return self._achieve(active, facts, "goal achieved (in flight)")
            # 1b. Цель перевыбирается каждый тик: смена/снятие исхода оператором —
            #     преемпция (новое вместо старого) или отмена (пауза).
            goal = select_goal(self.outcomes, facts)
            if goal is None:
                self._abandon()
                return GoalAttempt(
                    None,
                    "cancelled (no unsatisfied outcome)",
                    self._trace,
                    self._call_status,
                    facts,
                    True,
                    self._attempts,
                    self._chain_goal,
                    self._launched_goal,
                    self._planned_first_action,
                )
            if goal != active:
                self._begin_goal(goal)  # преемптит активное дерево
                # Новая цель запускается на этом же тике (ниже).
            else:
                return self._advance(facts)

        # 2. Активной попытки нет → выбираем цель по текущему наблюдению.
        goal = select_goal(self.outcomes, facts)

        if goal is None:
            self._abandon()
            return GoalAttempt(
                None,
                "no unsatisfied outcome",
                self._trace,
                self._call_status,
                facts,
                True,
                self._attempts,
                self._chain_goal,
                self._launched_goal,
                self._planned_first_action,
            )

        if self._goal is None or goal != self._goal:
            self._begin_goal(goal)
        elif self._changed(goal, facts):
            # Значимое изменение факта цели → сброс попыток (не любой шум).
            self._attempts = 0
            self._exhausted = False
            self._cooldown = 0

        if self._exhausted:
            return GoalAttempt(
                goal,
                f"attempts exhausted ({self._attempts})",
                self._trace,
                self._call_status,
                facts,
                False,
                self._attempts,
                self._chain_goal,
                self._launched_goal,
                self._planned_first_action,
            )

        if self._cooldown > 0:
            self._cooldown -= 1
            return GoalAttempt(
                goal,
                f"cooldown ({self._cooldown} left)",
                self._trace,
                self._call_status,
                facts,
                False,
                self._attempts,
                self._chain_goal,
                self._launched_goal,
                self._planned_first_action,
            )

        return self._launch(goal, facts)

    def _begin_goal(self, goal: Goal) -> None:
        """Начать жизненный цикл новой цели (сброс попыток, преемпция)."""
        if self._running and self._goal is not None:
            # Преемпция: смена цели останавливает текущее дерево (без отката
            # внешних шагов) — новое вместо старого (ADR-0012 §7, §7 SPEC).
            self._preempt(self._goal)
        self._goal = goal
        self._tree = None
        self._running = False
        self._attempts = 0
        self._exhausted = False
        self._cooldown = 0
        self._elapsed = 0
        self._goal_value = None
        self._trace = None
        self._call_status = ""
        self._chain_goal = ""
        self._planned_first_action = ""
        self._launched_goal = ""

    def _changed(self, goal: Goal, facts: Mapping[str, float]) -> bool:
        """Изменилось ли значение факта цели значимо с прошлого наблюдения."""
        value = evaluate_fact(goal.fact, facts)
        if self._goal_value is None:
            self._goal_value = value
            return False
        changed = abs(value - self._goal_value) > _SIGNIFICANCE_EPS
        self._goal_value = value
        return changed

    def _launch(self, goal: Goal, facts: Mapping[str, float]) -> GoalAttempt:
        """Построить цепочку по цели и запустить исполнение (новый тик).

        Трассы **разделены**: ``_trace`` — победитель глобального скоринга окна
        (аудит конкуренции), ``_chain_goal`` — опция, достигающая конечной цели
        (корень выведенной цепочки), ``_planned_first_action`` — первый ``Action``
        дерева (структурный план), ``_launched_goal`` — **фактически запущенный**
        лист (из события запуска исполнителя). Их совпадение не гарантировано.

        Если исполнитель завершил дерево терминально, не запустив ни одного
        действия (невыполненная предпосылка ``Condition``), фактического запуска
        нет: фиксируем провал, а не «launched via».
        """
        context = OptionContext(
            uncertainty=0.0,
            task=self.loop.active_task(),
            goal=goal,
            facts=facts,
        )
        self._trace = select_option(self._window, context, self.preferences)
        tree = backward_chain(
            goal,
            self._window.options,
            facts,
            max_depth=self.max_depth,
            context=context,
            preferences=self.preferences,
        )
        leaf = _first_action(tree)
        self._chain_goal = tree.name
        self._planned_first_action = leaf.name if leaf is not None else ""
        self._launched_goal = ""
        # Повторная попытка: то же дерево структурно равно прошлому, а executor
        # помнит терминальный итог и не переигрывает листья. Явный сброс
        # заставляет дерево переиграться (fail-safe повтор, §7).
        executor = getattr(self.loop, "executor", None)
        if executor is not None:
            executor.reset()
        self._tree = tree
        self._running = True
        self._elapsed = 0
        self._call_status = ""
        if leaf is None:
            # Безопасный отказ: чистый Condition без действия — не запускаем.
            self._running = False
            self._tree = None
            self._finish_failed()
            return GoalAttempt(
                goal,
                "unreachable (no action)",
                self._trace,
                self._call_status,
                facts,
                False,
                self._attempts,
                self._chain_goal,
                self._launched_goal,
                self._planned_first_action,
            )
        outcome = self.loop.tick_actuation(tree, facts)
        if outcome is None:
            self._running = False
            self._tree = None
            self._finish_failed()
            return GoalAttempt(
                goal,
                "executor unavailable",
                self._trace,
                self._call_status,
                facts,
                False,
                self._attempts,
                self._chain_goal,
                self._launched_goal,
                self._planned_first_action,
            )
        self._call_status = outcome.status.value
        # Фактический запуск — только из события исполнителя, не из структуры.
        self._launched_goal = outcome.started_goal or ""
        if not self._launched_goal:
            # Дерево терминально на старте без запуска действия: невыполненная
            # предпосылка (Condition провалился) → честный провал попытки.
            self._running = False
            self._tree = None
            self._finish_failed()
            detail = f": {outcome.last_reason}" if outcome.last_reason else ""
            return GoalAttempt(
                goal,
                f"not launched (no action started){detail}",
                self._trace,
                self._call_status,
                facts,
                False,
                self._attempts,
                self._chain_goal,
                self._launched_goal,
                self._planned_first_action,
            )
        return GoalAttempt(
            goal,
            f"launched via {self._launched_goal}",
            self._trace,
            self._call_status,
            facts,
            False,
            self._attempts,
            self._chain_goal,
            self._launched_goal,
            self._planned_first_action,
        )

    def _achieve(
        self, goal: Goal, facts: Mapping[str, float], reason: str
    ) -> GoalAttempt:
        """Зафиксировать достижение цели по наблюдению (успех цели, не вызова).

        Останавливает активное дерево (преемпция, если что-то бежит) и сбрасывает
        счётчик попыток. Отдельный путь для случая «цель достигнута в полёте»: не
        дожидаясь терминального ``poll``, достижение получает собственное событие.
        """
        if self._running:
            self._preempt(goal)
        self._running = False
        self._tree = None
        self._attempts = 0
        self._exhausted = False
        self._cooldown = 0
        return GoalAttempt(
            goal,
            reason,
            self._trace,
            self._call_status,
            facts,
            True,
            self._attempts,
            self._chain_goal,
            self._launched_goal,
            self._planned_first_action,
        )

    def _advance(self, facts: Mapping[str, float]) -> GoalAttempt:
        """Продвинуть бегущее дерево на тик (deadline → преемпция).

        Вызывается, когда активная цель **не** достигнута (достижение в полёте
        обрабатывается в ``tick`` до этого пути).
        """
        goal = self._goal
        if goal is None:  # недостижимо: _advance вызывается только с активной целью
            return GoalAttempt(
                None,
                "no active goal",
                self._trace,
                self._call_status,
                facts,
                False,
                self._attempts,
                self._chain_goal,
                self._launched_goal,
                self._planned_first_action,
            )

        self._elapsed += 1
        if self.deadline_ticks and self._elapsed > self.deadline_ticks:
            self._preempt(goal, facts)
            self._finish_failed()
            return GoalAttempt(
                goal,
                f"deadline exceeded ({self.deadline_ticks})",
                self._trace,
                self._call_status,
                facts,
                False,
                self._attempts,
                self._chain_goal,
                self._launched_goal,
                self._planned_first_action,
            )

        tree = self._tree
        if tree is None:
            self._running = False
            self._finish_failed()
            return GoalAttempt(
                goal,
                "no running tree",
                self._trace,
                self._call_status,
                facts,
                False,
                self._attempts,
                self._chain_goal,
                self._launched_goal,
                self._planned_first_action,
            )
        outcome = self.loop.tick_actuation(tree, facts)
        if outcome is None:
            self._running = False
            self._tree = None
            self._finish_failed()
            return GoalAttempt(
                goal,
                "executor unavailable",
                self._trace,
                self._call_status,
                facts,
                False,
                self._attempts,
                self._chain_goal,
                self._launched_goal,
                self._planned_first_action,
            )

        self._call_status = outcome.status.value
        if outcome.started_goal:
            # Многошаговая цепочка: на этом тике фактически запущен следующий
            # лист — трасса запуска отражает последний реальный запуск.
            self._launched_goal = outcome.started_goal
        if outcome.status is NodeStatus.RUNNING:
            return GoalAttempt(
                goal,
                "running",
                self._trace,
                self._call_status,
                facts,
                False,
                self._attempts,
                self._chain_goal,
                self._launched_goal,
                self._planned_first_action,
            )

        # Терминальный статус → новое наблюдение и проверка успеха цели.
        self._running = False
        self._tree = None
        after = dict(self.environment.observe())
        if goal_reached(goal, after):
            self._attempts = 0
            self._exhausted = False
            self._cooldown = 0
            return GoalAttempt(
                goal,
                "goal achieved",
                self._trace,
                self._call_status,
                after,
                True,
                self._attempts,
                self._chain_goal,
                self._launched_goal,
                self._planned_first_action,
            )
        self._finish_failed()
        detail = f": {outcome.last_reason}" if outcome.last_reason else ""
        return GoalAttempt(
            goal,
            f"goal not reached{detail}",
            self._trace,
            self._call_status,
            after,
            False,
            self._attempts,
            self._chain_goal,
            self._launched_goal,
            self._planned_first_action,
        )

    def _finish_failed(self) -> None:
        """Учесть провальную попытку: повторы, cooldown, исчерпание."""
        self._running = False
        self._tree = None
        self._attempts += 1
        self._cooldown = self.cooldown_ticks
        if self._attempts >= self.max_attempts:
            self._exhausted = True

    def _preempt(self, goal: Goal, facts: Mapping[str, float] | None = None) -> None:
        """Преемптить бегущее дерево (fail-safe) через смену корня.

        Новый корень — ``Condition`` на неудовлетворённую цель: executor
        пересбрасывает дерево и преемптит бегущие эффекторы (ADR-0012 §7).
        """
        if not self._running:
            return
        stub = Node(
            NodeKind.CONDITION,
            name=goal.fact.name,
            guard=Guard(goal.fact, goal.value),
        )
        self.loop.tick_actuation(stub, facts)

    def _abandon(self) -> None:
        """Остановить целевой контур (все исходы достигнуты / нет исходов)."""
        if self._running and self._goal is not None:
            self._preempt(self._goal)
        self._goal = None
        self._tree = None
        self._running = False
        self._attempts = 0
        self._exhausted = False
        self._cooldown = 0
        self._elapsed = 0
        self._goal_value = None
        self._chain_goal = ""
        self._planned_first_action = ""
        self._launched_goal = ""
