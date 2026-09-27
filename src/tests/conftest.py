"""Shared pytest configuration.

Tests must be hermetic: environment variables that select the real API embedder
or LLM (``EMBEDDER_*`` / ``LLM_*``) are cleared for every test, so
``mode="auto"`` always falls back to the deterministic fake clients. This keeps
runs offline and reproducible regardless of the developer's shell/.env.
"""

from __future__ import annotations

import pytest

from src.memory.embedder import ENV_API_KEY, ENV_BASE_URL, ENV_DIM, ENV_MODEL
from src.speech.llm import (
    ENV_API_KEY as LLM_ENV_API_KEY,
)
from src.speech.llm import (
    ENV_BASE_URL as LLM_ENV_BASE_URL,
)
from src.speech.llm import (
    ENV_MODEL as LLM_ENV_MODEL,
)

_API_ENV = (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_DIM,
    ENV_MODEL,
    LLM_ENV_API_KEY,
    LLM_ENV_BASE_URL,
    LLM_ENV_MODEL,
)


@pytest.fixture(autouse=True)
def _hermetic_api_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Убрать EMBEDDER_*/LLM_* из окружения на время теста (детерминизм)."""
    for name in _API_ENV:
        monkeypatch.delenv(name, raising=False)