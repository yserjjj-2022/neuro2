"""Shared pytest configuration.

Tests must be hermetic: environment variables that select the real API embedder
(``EMBEDDER_*``) are cleared for every test, so ``embedder_mode="auto"`` always
falls back to the deterministic ``FakeEmbedder``. This keeps runs offline and
reproducible regardless of the developer's shell/.env.
"""

from __future__ import annotations

import pytest

from src.memory.embedder import ENV_API_KEY, ENV_BASE_URL, ENV_DIM, ENV_MODEL

_EMBEDDER_ENV = (ENV_API_KEY, ENV_BASE_URL, ENV_DIM, ENV_MODEL)


@pytest.fixture(autouse=True)
def _hermetic_embedder_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Убрать EMBEDDER_* из окружения на время теста (детерминизм)."""
    for name in _EMBEDDER_ENV:
        monkeypatch.delenv(name, raising=False)