"""Integration tests for the social contour in ChatSession (S5, проход 1).

The partner model observes each utterance and feeds ``PartnerState`` into
policy. ``partner_model=None`` must leave the S4 contour identical.
"""

from __future__ import annotations

from pathlib import Path

from src.config import HostConfig, MemoryConfig, PolicyConfig
from src.core.policy import Action, Preferences
from src.host.loop import build_host_loop
from src.memory.embedder import FakeEmbedder
from src.speech.chat import ChatSession
from src.speech.controller import SpeechController
from src.speech.history import ConversationHistory
from src.speech.llm import FakeLlmClient
from src.tm import JointAgency, PartnerModel, PartnerState, VigilanceGate


def _session(
    tmp_path: Path,
    inputs: list[str],
    *,
    policy: PolicyConfig | None = None,
    partner_model: PartnerModel | None = None,
    vigilance: VigilanceGate | None = None,
    joint_agency: JointAgency | None = None,
) -> tuple[ChatSession, list[str]]:
    """Собрать сессию с fake-вводом/выводом и опциональной ToM."""
    config = HostConfig(
        log_path=str(tmp_path / "run.jsonl"),
        memory=MemoryConfig(db_path=str(tmp_path / "mem.db"), embedding_dim=64),
    )
    loop = build_host_loop(config)
    controller = SpeechController(
        llm=FakeLlmClient(),
        memory=loop.memory.store if loop.memory else None,
        embedder=loop.memory.embedder if loop.memory else None,
    )
    output: list[str] = []
    it = iter(inputs)

    def fake_input(prompt: str) -> str:
        try:
            return next(it)
        except StopIteration as exc:
            raise EOFError from exc

    session = ChatSession(
        loop=loop,
        controller=controller,
        history=ConversationHistory(max_turns=10),
        input_fn=fake_input,
        output_fn=output.append,
        policy=policy,
        partner_model=partner_model,
        vigilance=vigilance,
        joint_agency=joint_agency,
    )
    return session, output


class TestPartnerAccumulation:
    """The partner signature accumulates across turns."""

    def test_signature_accumulates(self, tmp_path: Path) -> None:
        model = PartnerModel(FakeEmbedder(dim=64), match_threshold=0.1)
        session, _ = _session(
            tmp_path,
            ["alpha beta gamma", "alpha beta gamma", "/quit"],
            partner_model=model,
        )
        session.run()
        session.loop.close()
        assert len(model.signatures) == 1
        assert model.signatures[0].weight >= 2.0

    def test_distinct_styles_two_signatures(self, tmp_path: Path) -> None:
        model = PartnerModel(FakeEmbedder(dim=64), match_threshold=0.99)
        session, _ = _session(
            tmp_path,
            ["alpha beta gamma", "delta epsilon zeta", "/quit"],
            partner_model=model,
        )
        session.run()
        session.loop.close()
        assert len(model.signatures) == 2


class TestSocialCompat:
    """partner_model=None → S4 contour is identical."""

    def test_none_is_s4_behaviour(self, tmp_path: Path) -> None:
        session, output = _session(
            tmp_path, ["привет", "/quit"], policy=PolicyConfig(), partner_model=None
        )
        session.run()
        session.loop.close()
        assert len(output) == 1
        assert session.loop.last_policy_trace is not None
        assert session.loop.last_policy_trace.chosen is Action.RESPOND


class TestIdentifyPartner:
    """High partner uncertainty activates the soft identify intent."""

    def test_identify_on_high_uncertainty(self, tmp_path: Path) -> None:
        """Неопределённость → policy выбирает IDENTIFY_PARTNER."""
        session, _ = _session(
            tmp_path,
            ["/quit"],
            policy=PolicyConfig(
                preferences=Preferences(
                    identify_threshold=0.5,
                    pragmatic_weight=0.0,
                    epistemic_weight=1.0,
                )
            ),
        )
        partner = PartnerState(uncertainty=1.0)
        goal, allow = session._decide_goal(has_new_message=True, partner=partner)
        session.loop.close()
        assert allow is True
        assert goal == "identify_partner"

    def test_no_identify_when_identified(self, tmp_path: Path) -> None:
        """Партнёр опознан → мягкий интент не срабатывает (RESPOND)."""
        session, _ = _session(
            tmp_path,
            ["/quit"],
            policy=PolicyConfig(
                preferences=Preferences(
                    identify_threshold=0.5,
                    pragmatic_weight=0.0,
                    epistemic_weight=1.0,
                )
            ),
        )
        partner = PartnerState(uncertainty=0.0)
        goal, allow = session._decide_goal(has_new_message=True, partner=partner)
        session.loop.close()
        assert allow is True
        assert goal == "respond"


class TestNameDeclaration:
    """``/name`` attaches the declared name to the partner signature (S5)."""

    def test_name_command_attaches(self, tmp_path: Path) -> None:
        model = PartnerModel(FakeEmbedder(dim=64))
        session, output = _session(
            tmp_path,
            ["alpha beta gamma", "/name Сергей", "/quit"],
            partner_model=model,
        )
        session.run()
        session.loop.close()
        assert model.state.name == "Сергей"
        assert any("Сергей" in line for line in output)

    def test_name_reaches_intent_prompt(self, tmp_path: Path) -> None:
        """Имя партнёра попадает в system-промпт (вокатив)."""
        from src.speech.intent import build_intent_frame, render_messages

        frame = build_intent_frame(
            f=0.0, valence=0.0, stress=0.0, task="tone", partner_name="Сергей"
        )
        system = render_messages(frame, "привет")[0]["content"]
        assert "Сергей" in system


class TestVigilanceInChat:
    """Vigilance observes claims without blocking replies (S5, C9)."""

    def test_claim_recorded_not_blocking(self, tmp_path: Path) -> None:
        gate = VigilanceGate(FakeEmbedder(dim=64))
        session, output = _session(
            tmp_path,
            ["я утверждаю что-то новое", "/quit"],
            policy=PolicyConfig(),
            vigilance=gate,
        )
        session.run()
        session.loop.close()
        assert gate.last_claim is not None
        assert len(output) == 1  # ответ не заблокирован
        assert output[0] != ""


class TestJointAgencyInChat:
    """JointAgency tracks shared goals (S5, проход 2)."""

    def test_propose_and_hold(self, tmp_path: Path) -> None:
        agency = JointAgency(default_ttl=10)
        session, _ = _session(
            tmp_path, ["/quit"], policy=PolicyConfig(), joint_agency=agency
        )
        agency.propose("build", tick=0)
        assert agency.update("build", tick=1) is None
        assert agency.update("chat", tick=2) is not None
        session.loop.close()
