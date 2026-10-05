"""Entry point: run the host loop from the command line.

Usage:
    uv run python -m src --ticks 100 --dt 0.01 --log run.jsonl
"""

from __future__ import annotations

import argparse
import logging
import signal
from pathlib import Path

from dotenv import load_dotenv

from src.config import (
    AutonomyConfig,
    HostConfig,
    MemoryConfig,
    PolicyConfig,
    SocialConfig,
    SpeechConfig,
)
from src.host.loop import HostLoop, build_host_loop
from src.integrations import load_integrations
from src.integrations.runtime import connect_probe_transport
from src.speech import (
    ChatSession,
    ConversationHistory,
    SpeechController,
    build_llm_client,
    llm_settings_from_env,
)
from src.tm import JointAgency, PartnerModel, VigilanceGate

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Аргументы (None → sys.argv[1:]).

    Returns:
        Namespace с параметрами запуска.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Запустить host loop neuro2.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=100,
        help="Число тиков (0 = бесконечно, до Ctrl+C). По умолчанию 100.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Шаг интегрирования в секундах (0 = без пауз). По умолчанию 0.01.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("host_telemetry.jsonl"),
        help="Путь к JSONL-файлу телеметрии.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Число победителей k-WTA. По умолчанию 2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Зерно детерминированных провайдеров. По умолчанию 0.",
    )
    parser.add_argument(
        "--embedder",
        choices=("auto", "fake", "api"),
        default="auto",
        help="Эмбеддер памяти: auto (ключ→api, иначе fake), fake, api.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("host_memory.db"),
        help="Путь к SQLite-файлу памяти.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Отключить память (контур S1).",
    )
    parser.add_argument(
        "--precision",
        choices=("ones", "variance"),
        default="variance",
        help="Режим точности γ: variance (S1, 1/var) или ones (baseline).",
    )
    parser.add_argument(
        "--clock-mode",
        choices=("synthetic", "wall"),
        default="synthetic",
        help="Источник времени: synthetic (детерминизм) или wall (реальное).",
    )
    parser.add_argument(
        "--paced",
        action="store_true",
        help="Спать между тиками (реальное время ≈ dt).",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Диалоговый режим: ввод оператора → ответ хоста (S3).",
    )
    parser.add_argument(
        "--llm",
        choices=("auto", "fake", "api"),
        default="auto",
        help="LLM-клиент речи: auto (ключ→api, иначе fake), fake, api.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Модель LLM (переопределяет LLM_MODEL из окружения).",
    )
    parser.add_argument(
        "--register",
        choices=("brief", "terse", "normal", "story"),
        default="brief",
        help="Речевой режим (длина ответа). По умолчанию brief.",
    )
    parser.add_argument(
        "--f-threshold",
        type=float,
        default=1.0,
        help="Порог F для инициативы (речь без сообщения). По умолчанию 1.0.",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=20,
        help="Глубина истории диалога (сообщений). По умолчанию 20.",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Включить reasoning у LLM (по умолчанию выкл; LLM — актюатор, ADR-0007).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Печатать состояние (F/valence/stress/γ/задача/recall/дрейф) в чате.",
    )
    parser.add_argument(
        "--no-policy",
        action="store_true",
        help="Отключить policy (S4): вернуть S3-поведение should_speak в чате.",
    )
    parser.add_argument(
        "--mode",
        choices=("game", "cooperative", "free"),
        default="free",
        help="Режим хоста (макро-контекст policy): game/cooperative/free.",
    )
    parser.add_argument(
        "--no-social",
        action="store_true",
        help="Отключить ToM (S5): контур S4 без модели партнёра.",
    )
    parser.add_argument(
        "--no-autonomy",
        action="store_true",
        help="Отключить автономию (S6): контур S5 без selfcontrol.",
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Выполнить явную консолидацию памяти (S6) и выйти.",
    )
    parser.add_argument(
        "--night-every",
        type=int,
        default=0,
        help="Интервал ночного цикла консолидации, тики (0 = выключен).",
    )
    parser.add_argument(
        "--night-min-episodes",
        type=int,
        default=0,
        help="Минимум эпизодов для срабатывания ночного цикла.",
    )
    parser.add_argument(
        "--integrations",
        type=str,
        default=None,
        help="Путь к TOML-override реестра интеграций (ADR-0011).",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Прогнать sensitivity-harness (матрица ручек → инварианты, S7-A).",
    )
    parser.add_argument(
        "--sensitivity-ticks",
        type=int,
        default=120,
        help="Тиков на прогон в sensitivity-harness. По умолчанию 120.",
    )
    return parser.parse_args(argv)


def _run_sensitivity(args: argparse.Namespace) -> int:
    """Прогнать sensitivity-harness и напечатать матрицу (S7-A).

    Args:
        args: Аргументы CLI (sensitivity_ticks, seed).

    Returns:
        Код выхода: 0, если все инварианты выполнены, иначе 1.
    """
    from src.host.sensitivity import SensitivityRunner

    runner = SensitivityRunner(ticks=args.sensitivity_ticks)
    results = runner.run_all(seed=args.seed)
    ok = True
    for result in results:
        case = result.case
        values = ", ".join(f"{v:.4f}" for v in result.metric_values)
        verdict = "OK" if result.passed else "FAIL"
        ok = ok and result.passed
        print(
            f"[{verdict}] {case.knob} → {case.metric} "
            f"({case.direction}): [{values}]"
        )
    print("sensitivity: all invariants hold" if ok else "sensitivity: FAILURES")
    return 0 if ok else 1


def _run_chat(
    loop: HostLoop, args: argparse.Namespace, social: SocialConfig
) -> int:
    """Запустить диалоговый стенд (S3/S5).

    Args:
        loop: Собранный host loop (с включённой памятью и речью).
        args: Аргументы CLI (llm, model, register, f_threshold).
        social: Параметры социального контура (S5).

    Returns:
        Код выхода (0 — успех).
    """
    settings = llm_settings_from_env()
    model = args.model or str(settings["model"])
    llm = build_llm_client(
        mode=args.llm,
        model=model,
        base_url=str(settings["base_url"]),
        reasoning=args.reasoning,
    )
    controller = SpeechController(
        llm=llm,
        memory=loop.memory.store if loop.memory is not None else None,
        embedder=loop.memory.embedder if loop.memory is not None else None,
        f_threshold=args.f_threshold,
        default_register=args.register,
    )
    history = ConversationHistory(max_turns=args.history_turns)
    policy = None if args.no_policy else loop.policy_config
    partner_model = None
    vigilance = None
    joint_agency = None
    if social.enabled and loop.memory is not None:
        embedder = loop.memory.embedder
        partner_model = PartnerModel(
            embedder=embedder,
            match_threshold=social.match_threshold,
            learning_rate=social.signature_learning_rate,
            trust_gain=social.trust_gain,
            trust_decay=social.trust_decay,
        )
        vigilance = VigilanceGate(
            embedder=embedder, conflict_threshold=social.conflict_threshold
        )
        if args.mode == "cooperative":
            joint_agency = JointAgency()
    session = ChatSession(
        loop=loop,
        controller=controller,
        history=history,
        show_status=args.status,
        policy=policy,
        partner_model=partner_model,
        pause_tau_s=social.pause_tau_s,
        vigilance=vigilance,
        joint_agency=joint_agency,
    )
    logger.info(
        "Chat session started (llm=%s, model=%s). /quit to exit.", args.llm, model
    )
    try:
        turns = session.run(max_turns=0)
    finally:
        loop.close()
    logger.info("Chat session ended: %d turns", turns)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа: собрать loop, гнать тики, корректно завершиться.

    Args:
        argv: Аргументы (None → sys.argv[1:]).

    Returns:
        Код выхода (0 — успех).
    """
    load_dotenv()  # .env → окружение (EMBEDDER_API_KEY, EMBEDDER_MODEL, ...)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    if args.sensitivity:
        return _run_sensitivity(args)
    social = SocialConfig(enabled=not args.no_social)
    autonomy = AutonomyConfig(
        enabled=not args.no_autonomy,
        consolidate_every_ticks=args.night_every,
        consolidate_min_episodes=args.night_min_episodes,
    )

    # Реестр интеграций (ADR-0011): каталог + реальные MCP-клиенты. Без флага
    # поведение S6 идентично (default_affordances + mock-транспорт).
    integrations = None
    probe_fn = None
    probe_affordances = None
    probe_clients: tuple = ()
    if args.integrations:
        integrations = load_integrations(Path(args.integrations))
        probe_fn, probe_clients = connect_probe_transport(integrations)
        if probe_fn is not None:
            # Аффордансы — по фактическим тулам подключённых серверов.
            probe_affordances = probe_fn.affordances()

    loop: HostLoop = build_host_loop(
        HostConfig(
            dt=args.dt,
            max_ticks=args.ticks,
            k=args.k,
            seed=args.seed,
            precision_mode=args.precision,
            clock_mode=args.clock_mode,
            paced=args.paced,
            log_path=str(args.log),
            memory=MemoryConfig(
                enabled=not args.no_memory,
                embedder_mode=args.embedder,
                db_path=str(args.db),
            ),
            speech=SpeechConfig(
                enabled=args.chat,
                llm_mode=args.llm,
                default_register=args.register,
                f_threshold=args.f_threshold,
                history_turns=args.history_turns,
            ),
            policy=PolicyConfig(enabled=not args.no_policy, mode=args.mode),
            social=social,
            autonomy=autonomy,
        ),
        integrations=integrations,
        probe_fn=probe_fn,
        affordances=probe_affordances,
    )

    if args.consolidate:
        pruned = loop.consolidate_memory()
        logger.info("Consolidation done: %d episodes pruned", pruned)
        loop.close()
        return 0

    if args.chat:
        return _run_chat(loop, args, social)

    stopping = {"flag": False}

    def _handle_sigint(signum: int, frame: object) -> None:
        if not stopping["flag"]:
            logger.info("SIGINT received — finishing current tick and shutting down")
            stopping["flag"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    logger.info(
        "Host loop: ticks=%d dt=%.4f bus_dim=%d columns=%d log=%s",
        args.ticks,
        args.dt,
        loop.bus.bus_dim,
        loop.pipeline.ensemble.n_columns,
        args.log,
    )

    try:
        if args.ticks == 0:
            tick = 0
            while not stopping["flag"]:
                loop.step_once(tick)
                tick += 1
            executed = tick
        else:
            executed = loop.run(args.ticks)
    finally:
        loop.close()
        for client in probe_clients:
            client.close()

    logger.info("Done: %d ticks written to %s", executed, args.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
