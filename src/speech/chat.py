"""ChatSession — CLI dialogue stand over the host loop (Shell).

Runs the affective loop while the operator talks. Each input becomes a
communicative message; the loop processes it event-triggered; the
SpeechController produces a reply. The LLM call happens here (not in the tick),
so a slow reply never blocks the affective contour.

Commands: ``/clear`` (drop dialogue history), ``/quit`` (exit).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from src.config import PolicyConfig
from src.core.policy import Action, select_action
from src.host.gate import ActionRequest, CapabilityGate, CapabilityTier
from src.host.loop import HostLoop
from src.speech.controller import SpeechController
from src.speech.history import ConversationHistory
from src.speech.status import format_status

logger = logging.getLogger(__name__)

_CLEAR = "/clear"
_QUIT = "/quit"


class ChatSession:
    """Диалоговый стенд: ввод оператора → тики → реплика хоста.

    Attributes:
        loop: Host loop (аффективный контур).
        controller: Генератор реплик.
        history: История диалога.
        input_fn: Источник ввода (инъекция для тестов).
        output_fn: Приёмник вывода (инъекция для тестов).
        ticks_per_turn: Сколько тиков прогнать на реплику (сообщение «осмыслено»).
        show_status: Печатать строку состояния перед каждой репликой (HITL).
        policy: Параметры policy (S4); None → S3-поведение (should_speak).
        gate: Capability gate (S4): единая точка аудита side-effect.
    """

    def __init__(
        self,
        loop: HostLoop,
        controller: SpeechController,
        history: ConversationHistory | None = None,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        ticks_per_turn: int = 3,
        show_status: bool = False,
        policy: PolicyConfig | None = None,
        gate: CapabilityGate | None = None,
    ) -> None:
        if ticks_per_turn < 1:
            raise ValueError(f"ticks_per_turn must be >= 1, got {ticks_per_turn}")
        self.loop = loop
        self.controller = controller
        self.history = history if history is not None else ConversationHistory()
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.ticks_per_turn = ticks_per_turn
        self.show_status = show_status
        self.policy = policy
        self.gate = gate if gate is not None else CapabilityGate()
        self._tick = 0
        self._last_message = ""

    def _advance_loop(self, message: str) -> None:
        """Прокрутить тики, подав новое сообщение (если есть)."""
        if message and self.loop.message_provider is not None:
            self.loop.message_provider = type(self.loop.message_provider)(
                embedder=self.loop.message_provider.embedder,
                messages=((self._tick, message),),
            )
            self._last_message = message
        for _ in range(self.ticks_per_turn):
            self.loop.step_once(self._tick)
            self._tick += 1

    def _handle_turn(self, user_input: str) -> bool:
        """Обработать одну реплику оператора.

        Args:
            user_input: Ввод оператора.

        Returns:
            False, если нужно завершить сессию (/quit), иначе True.
        """
        if user_input == _QUIT:
            return False
        if user_input == _CLEAR:
            self.history.clear()
            self.output_fn("[история очищена]")
            return True
        if not user_input:
            return True

        self._advance_loop(user_input)
        self.history.add_user(user_input)

        outcome = self.loop.last_outcome
        f = outcome.result.f if outcome is not None else 0.0
        valence = outcome.result.valence if outcome is not None else 0.0
        stress = outcome.result.allostatic_stress if outcome is not None else 0.0
        gamma = outcome.result.gamma if outcome is not None else 0.0
        task = self._active_task()

        if self.show_status:
            self.output_fn(
                format_status(
                    f=f,
                    valence=valence,
                    stress=stress,
                    gamma=gamma,
                    task=task,
                    recall_hit=self.loop.last_memory_hit,
                    drift=self.loop.last_drift,
                )
            )

        goal, allow_speak = self._decide_goal(has_new_message=True)
        if not allow_speak:
            self.output_fn("[хост промолчал]")
            return True

        # Capability gate (S4): речь — T1 (обратимая), но проходит через
        # единую точку аудита side-effect (ADR-0005 §9).
        decision = self.gate.request(
            ActionRequest(
                name="speak",
                tier=CapabilityTier.T1,
                reversible=True,
                reason=goal or "respond",
            )
        )
        if not decision.allowed:
            self.output_fn("[хост промолчал]")
            return True

        reply = self.controller.respond(
            user_text=user_input,
            f=f,
            valence=valence,
            stress=stress,
            task=task,
            history=self.history.as_messages()[:-1],  # без текущей реплики
            new_message=True,
            goal=goal,
        )
        if reply is None:
            self.output_fn("[хост промолчал]")
            return True
        self.loop.mark_spoke()
        self.history.add_assistant(reply)
        self.output_fn(reply)
        return True

    def _decide_goal(self, *, has_new_message: bool) -> tuple[str | None, bool]:
        """Решить речевое действие через policy (S4) или S3-дефолт.

        Args:
            has_new_message: Пришло ли новое сообщение оператора.

        Returns:
            (goal, allow_speak): цель реплики для IntentFrame и разрешение
            говорить. При ``policy=None`` или ``policy.enabled=False`` —
            S3-поведение (goal=None).
        """
        if self.policy is None or not self.policy.enabled:
            return None, True

        # Рефлекс-throttle запрещает дорогой инициативный вызов LLM, но
        # ответ на сообщение сохраняется (S4_SPEC §3).
        if self.loop.last_throttle.llm_gate and not has_new_message:
            return None, False

        context = self.loop.policy_context(
            has_new_message=has_new_message, mode=self.policy.mode
        )
        trace = select_action(context, self.policy.preferences)
        self.loop.record_policy(trace)
        if trace.chosen is Action.SILENT:
            return None, False
        if trace.chosen is Action.IDENTIFY_PARTNER:
            return "identify_partner", True
        if trace.chosen is Action.INITIATIVE:
            return "initiative", True
        return "respond", True

    def _active_task(self) -> str:
        """Тег активной задачи (колонки) по текущему аттрактору."""
        return self.loop.active_task()

    def run(self, max_turns: int = 0) -> int:
        """Запустить диалог.

        Args:
            max_turns: Максимум реплик оператора (0 → до /quit или EOF).

        Returns:
            Число обработанных реплик.
        """
        turns = 0
        while max_turns == 0 or turns < max_turns:
            try:
                user_input = self.input_fn("> ")
            except EOFError:
                break
            if not self._handle_turn(user_input):
                break
            turns += 1
        return turns
