"""Entry point: run the host loop from the command line.

Usage:
    uv run python -m src --ticks 100 --dt 0.01 --log run.jsonl
"""

from __future__ import annotations

import argparse
import logging
import signal
from pathlib import Path

from src.config import HostConfig, MemoryConfig
from src.host.loop import HostLoop, build_host_loop

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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа: собрать loop, гнать тики, корректно завершиться.

    Args:
        argv: Аргументы (None → sys.argv[1:]).

    Returns:
        Код выхода (0 — успех).
    """
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
        )
    )

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
