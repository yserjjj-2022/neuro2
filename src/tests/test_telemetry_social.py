"""Telemetry social fields (S5, проход 2).

``TelemetryLogger.log`` accepts the five social fields as keyword-only
defaults, so existing callers keep working; ``record_social`` on the loop
feeds them into the next event.
"""

from __future__ import annotations

from pathlib import Path

from src.config import HostConfig, MemoryConfig
from src.host.loop import build_host_loop
from src.telemetry import TelemetryLogger, TelemetryWriter


def _logger(tmp_path: Path) -> TelemetryLogger:
    return TelemetryLogger(writer=TelemetryWriter(log_path=tmp_path / "t.jsonl"))


class TestSocialLogFields:
    def test_defaults_keep_backward_compat(self, tmp_path: Path) -> None:
        """log() без social-полей не падает (дефолты)."""
        logger = _logger(tmp_path)
        logger.log(1.0, 0.0, 0.0)
        assert logger.writer  # запись прошла

    def test_social_fields_serialized(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        logger.log(
            1.0,
            0.0,
            0.0,
            partner_trust=0.7,
            partner_uncertainty=0.2,
            partner_name="Сергей",
            pause_s=3.5,
            claim_conflict=0.4,
        )
        text = (tmp_path / "t.jsonl").read_text(encoding="utf-8")
        assert "Сергей" in text
        assert "claim_conflict" in text


class TestRecordSocial:
    def test_record_social_reaches_event(self, tmp_path: Path) -> None:
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "m.db")),
        )
        loop = build_host_loop(config)
        loop.record_social(
            trust=0.8, uncertainty=0.1, name="Аня", pause_s=2.0, claim_conflict=0.3
        )
        loop.step_once(0)
        loop.close()
        text = (tmp_path / "run.jsonl").read_text(encoding="utf-8")
        assert "Аня" in text
        assert '"partner_trust": 0.8' in text
