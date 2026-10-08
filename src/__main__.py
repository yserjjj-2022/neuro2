"""Entry point: run the host loop from the command line.

Usage:
    uv run python -m src --ticks 100 --dt 0.01 --log run.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from src.config import (
    AutonomyConfig,
    HostConfig,
    MemoryConfig,
    PolicyConfig,
    SocialConfig,
    SpeechConfig,
    load_preset,
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

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.host.behavioral_chain import Deviation, ScenarioResult

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
        "--preset",
        type=str,
        default=None,
        help="Именованный пресет конфигурации (S7-B): baseline/stress/dialogue/"
        "autonomy/long-horizon/cooperative.",
    )
    parser.add_argument(
        "--preset-file",
        type=Path,
        default=None,
        help="TOML-override поверх пресета (S7-B).",
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
    parser.add_argument(
        "--behavioral",
        action="store_true",
        help="Прогнать поведенческий автотест по звеньям (VALIDATION §7).",
    )
    parser.add_argument(
        "--behavioral-precondition",
        choices=("born", "primed", "matured"),
        default=None,
        help="Предусловие прогона b-теста (None → все сценарии).",
    )
    parser.add_argument(
        "--behavioral-warmup",
        type=int,
        default=1,
        help="Число прогревочных реплик для primed-предусловия. По умолчанию 1.",
    )
    parser.add_argument(
        "--behavioral-json",
        type=Path,
        default=None,
        help="Путь для JSON-отчёта b-теста (None → без файла).",
    )
    parser.add_argument(
        "--behavioral-baseline",
        type=Path,
        default=None,
        help="JSON-эталон наблюдаемых долей для сравнения (None → без сравнения).",
    )
    parser.add_argument(
        "--behavioral-save-baseline",
        type=Path,
        default=None,
        help="Путь, куда записать эталон наблюдаемых долей текущего прогона.",
    )
    parser.add_argument(
        "--behavioral-band",
        type=float,
        default=0.05,
        help="Полуширина полосы сравнения с эталоном (доля). По умолчанию 0.05.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Запустить диагностическую сессию (дерево проб, S7-C).",
    )
    parser.add_argument(
        "--probes",
        type=Path,
        default=None,
        help="JSON-файл дерева проб (None → встроенное дерево S3–S6).",
    )
    parser.add_argument(
        "--diagnose-start",
        type=str,
        default=None,
        help="id стартовой пробы (None → старт по умолчанию).",
    )
    parser.add_argument(
        "--diagnose-log",
        type=Path,
        default=Path("diagnostic.jsonl"),
        help="JSONL-журнал вердиктов диагностики.",
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


# Человекочитаемые подписи операций и звеньев для микроотчёта (VALIDATION §7.1).
_OPERATION_LABELS: dict[str, str] = {
    "silent": "промолчал",
    "respond": "ответил на сообщение",
    "initiative": "проявил инициативу",
    "identify_partner": "идентифицировал партнёра",
    "explore": "исследовал",
    "escape_hatch": "ушёл в escape hatch (перегрузка, без LLM)",
    "llm_call": "вызвал LLM",
    "throttled": "отсечён throttle",
}

_LINK_LABELS: dict[str, str] = {
    "state": "состояние",
    "decision": "решение",
    "intent": "интент",
    "actuation": "актюация",
    "reply": "реплика",
}


def _plural_raz(n: int) -> str:
    """Согласовать «раз/раза» с числом (человекочитаемость).

    Args:
        n: Число.

    Returns:
        ``"раз"`` или ``"раза"``.
    """
    if n % 10 == 1 and n % 100 != 11:
        return "раз"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return "раза"
    return "раз"


def _render_micro_report(
    result: ScenarioResult,
    deviations: Mapping[str, Deviation] | None = None,
) -> list[str]:
    """Собрать микроотчёт по одному сценарию: операции с числом и ожидаемым.

    Декомпозиция (``operation_facts``) делает числа интерпретируемыми: видно,
    сколько раз хост выполнил каждую операцию, почему и сколько *ожидалось* для
    инварианта. Доля — от числа измеренных тиков (без прогрева; §7.8). Если
    передан эталон (``deviations``), наблюдаемые операции получают ссылку на
    эталонную долю и смещение по полосе (§7.1).

    Args:
        result: Результат прогона.
        deviations: Отклонения наблюдаемых этого сценария (операция → эталон).

    Returns:
        Строки отчёта (заголовок + операции + причина провала).
    """
    from src.host.behavioral_chain import operation_facts

    scenario = result.scenario
    verdict = "OK" if result.passed else "FAIL"
    link = _LINK_LABELS.get(scenario.link, scenario.link)
    measured = scenario.ticks - scenario.measure_from()
    lines = [f"{scenario.id} · {link} · {scenario.precondition} — {verdict}"]

    for fact in operation_facts(result):
        label = _OPERATION_LABELS.get(fact.operation, fact.operation)
        ratio = fact.count / measured if measured else 0.0
        share = f" ({ratio:.0%})" if ratio >= 0.01 else ""
        expected = ""
        if fact.expected is not None:
            mark = "✓" if fact.holds else "✗"
            expected = f" — ожидаемо {fact.expected} {mark}"
        elif deviations is not None and fact.operation in deviations:
            deviation = deviations[fact.operation]
            mark = "✓" if deviation.within_band else "✗"
            expected = (
                f" — эталон {deviation.baseline:.0%} (Δ{deviation.delta:+.0%}) {mark}"
            )
        reason = f" ({fact.reason})" if fact.reason else ""
        lines.append(
            f"    {label}: {fact.count} {_plural_raz(fact.count)}"
            f"{share}{expected}{reason}"
        )

    if not result.passed:
        lines.append(f"    причина провала: {result.reason}")
    return lines


def _run_behavioral(args: argparse.Namespace) -> int:
    """Прогнать поведенческий автотест по звеньям и напечатать микроотчёт (S7).

    Args:
        args: Аргументы CLI (behavioral_precondition, behavioral_warmup,
            behavioral_json, behavioral_baseline, behavioral_save_baseline,
            behavioral_band).

    Returns:
        Код выхода: 0, если провалов инвариантов нет, иначе 1. Отклонение от
        эталона наблюдаемых — калибровочный сигнал (§7.1), код не меняет.
    """
    from src.host.behavioral_chain import (
        BehavioralChainRunner,
        Precondition,
        PreconditionKind,
        baseline_dict,
        compare_to_baseline,
        report_dict,
        summarize,
    )

    precondition: Precondition | None = None
    if args.behavioral_precondition is not None:
        kind = PreconditionKind(args.behavioral_precondition)
        precondition = (
            Precondition.primed(args.behavioral_warmup)
            if kind is PreconditionKind.PRIMED
            else Precondition(kind=kind)
        )

    results = BehavioralChainRunner().run_all(precondition=precondition)

    deviations_by_scenario: dict[str, dict[str, Deviation]] = {}
    out_of_band = 0
    if args.behavioral_baseline is not None:
        baseline = json.loads(args.behavioral_baseline.read_text(encoding="utf-8"))
        for deviation in compare_to_baseline(
            results, baseline, band=args.behavioral_band
        ):
            deviations_by_scenario.setdefault(deviation.scenario_id, {})[
                deviation.operation
            ] = deviation
            if not deviation.within_band:
                out_of_band += 1

    for result in results:
        for line in _render_micro_report(
            result, deviations_by_scenario.get(result.scenario.id)
        ):
            print(line)
        print()

    summary = summarize(results)
    print(
        f"behavioral: {summary['passed']}/{summary['total']} passed "
        f"(failed={summary['failed_ids']})"
    )
    if args.behavioral_baseline is not None:
        total = sum(len(d) for d in deviations_by_scenario.values())
        print(
            f"behavioral: baseline band ±{args.behavioral_band:.0%}: "
            f"{total - out_of_band}/{total} within band"
        )
    if args.behavioral_save_baseline is not None:
        args.behavioral_save_baseline.write_text(
            json.dumps(baseline_dict(results), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"behavioral: baseline written to {args.behavioral_save_baseline}")
    if args.behavioral_json is not None:
        args.behavioral_json.write_text(
            json.dumps(report_dict(results), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"behavioral: report written to {args.behavioral_json}")
    return 0 if summary["failed"] == 0 else 1


def _run_diagnose(args: argparse.Namespace) -> int:
    """Запустить диагностическую сессию и напечатать вердикты (S7-C).

    Args:
        args: Аргументы CLI (probes, diagnose_start, diagnose_log).

    Returns:
        Код выхода (0 — успех).
    """
    from src.host.diagnostic import DiagnosticSession
    from src.host.probes import default_probes, default_start, load_probes

    probes = load_probes(args.probes) if args.probes else default_probes()
    if args.diagnose_start is not None:
        start = args.diagnose_start
    elif args.probes is None:
        start = default_start()
    else:
        start = next(iter(probes))
    session = DiagnosticSession(probes=probes, journal_path=args.diagnose_log)
    results = session.run(start=start)
    for result in results:
        print(f"[{result.probe_id}] {result.verdict.value} → next={result.next_probe}")
    print(f"diagnose: {len(results)} probes, journal={args.diagnose_log}")
    return 0


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


def _build_config(
    args: argparse.Namespace, social: SocialConfig, autonomy: AutonomyConfig
) -> HostConfig:
    """Собрать HostConfig из CLI-флагов или именованного пресета (S7-B).

    При ``--preset`` база берётся из пресета, а поверх применяются только
    операционные CLI-переопределения (dt/ticks/seed/log/db) и выключатели
    (--no-policy/--no-social/--no-autonomy/--chat). Пресет фиксирует
    детерминизм (synthetic, fake LLM/embedder) — он не переопределяется.

    Args:
        args: Аргументы CLI.
        social: Социальный конфиг из CLI-флагов.
        autonomy: Конфиг автономии из CLI-флагов.

    Returns:
        Готовый HostConfig.
    """
    if args.preset is not None:
        config = load_preset(args.preset, override=args.preset_file)
        return replace(
            config,
            dt=args.dt,
            max_ticks=args.ticks,
            seed=args.seed,
            log_path=str(args.log),
            memory=replace(config.memory, db_path=str(args.db)),
            speech=replace(config.speech, enabled=config.speech.enabled or args.chat),
            policy=replace(
                config.policy, enabled=config.policy.enabled and not args.no_policy
            ),
            social=replace(
                config.social, enabled=config.social.enabled and not args.no_social
            ),
            autonomy=replace(
                config.autonomy,
                enabled=config.autonomy.enabled and not args.no_autonomy,
                consolidate_every_ticks=args.night_every,
                consolidate_min_episodes=args.night_min_episodes,
            ),
        )
    return HostConfig(
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
    )


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
    if args.behavioral:
        return _run_behavioral(args)
    if args.diagnose:
        return _run_diagnose(args)
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
        _build_config(args, social, autonomy),
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
