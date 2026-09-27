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

from src.config import HostConfig, MemoryConfig, SpeechConfig
from src.host.loop import HostLoop, build_host_loop
from src.speech import (
    ChatSession,
    ConversationHistory,
    SpeechController,
    build_llm_client,
    llm_settings_from_env,
)

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
    return parser.parse_args(argv)


def _run_chat(loop: HostLoop, args: argparse.Namespace) -> int:
    """Запустить диалоговый стенд (S3).

    Args:
        loop: Собранный host loop (с включённой памятью и речью).
        args: Аргументы CLI (llm, model, register, f_threshold).

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
    session = ChatSession(loop=loop, controller=controller, history=history)
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
        )
    )

    if args.chat:
        return _run_chat(loop, args)

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

    logger.info("Done: %d ticks written to %s", executed, args.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
