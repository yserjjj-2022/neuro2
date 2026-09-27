"""Unit tests for LLM client — fake, factory, env settings (no network)."""

from __future__ import annotations

import pytest

from src.speech.llm import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    ApiLlmClient,
    FakeLlmClient,
    LlmClient,
    build_llm_client,
    llm_settings_from_env,
)


class TestFakeLlmClient:
    def test_deterministic(self) -> None:
        client = FakeLlmClient()
        msgs = [{"role": "system", "content": "Ответь одной-двумя короткими фразами."}]
        assert client.reply(msgs) == client.reply(msgs)

    def test_nonempty(self) -> None:
        assert FakeLlmClient().reply([{"role": "system", "content": "x"}]) != ""

    def test_register_affects_reply(self) -> None:
        client = FakeLlmClient()
        terse = client.reply([{"role": "system", "content": "междометием"}])
        brief = client.reply([{"role": "system", "content": "одной-двумя"}])
        assert terse != brief
        assert len(terse) < len(brief)

    def test_is_llm_client(self) -> None:
        assert isinstance(FakeLlmClient(), LlmClient)


class TestBuildLlmClient:
    def test_fake_mode(self) -> None:
        assert isinstance(build_llm_client(mode="fake"), FakeLlmClient)

    def test_auto_without_key_is_fake(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        assert isinstance(build_llm_client(mode="auto"), FakeLlmClient)

    def test_auto_with_key_is_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_API_KEY, "sk-test")
        client = build_llm_client(mode="auto")
        assert isinstance(client, ApiLlmClient)
        assert client.model == DEFAULT_MODEL

    def test_api_without_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        with pytest.raises(ValueError):
            build_llm_client(mode="api")

    def test_api_with_key_and_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        client = build_llm_client(
            mode="api",
            api_key="sk-explicit",
            model="m",
            base_url="https://x/v1",
            temperature=0.3,
        )
        assert isinstance(client, ApiLlmClient)
        assert client.model == "m"
        assert client.base_url == "https://x/v1"
        assert client.temperature == 0.3

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            build_llm_client(mode="nonsense")

    def test_reasoning_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        client = build_llm_client(mode="api", api_key="sk-x")
        assert isinstance(client, ApiLlmClient)
        assert client.reasoning is False

    def test_reasoning_flag_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        client = build_llm_client(mode="api", api_key="sk-x", reasoning=True)
        assert isinstance(client, ApiLlmClient)
        assert client.reasoning is True

    def test_universal_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Без LLM_API_KEY используется универсальный EMBEDDER_API_KEY."""
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        monkeypatch.setenv("EMBEDDER_API_KEY", "sk-universal")
        client = build_llm_client(mode="api")
        assert isinstance(client, ApiLlmClient)


class TestLlmSettingsFromEnv:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_BASE_URL, raising=False)
        monkeypatch.delenv(ENV_MODEL, raising=False)
        settings = llm_settings_from_env()
        assert settings["base_url"] == DEFAULT_BASE_URL
        assert settings["model"] == DEFAULT_MODEL

    def test_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_BASE_URL, "https://custom/v1")
        monkeypatch.setenv(ENV_MODEL, "other/model")
        settings = llm_settings_from_env()
        assert settings["base_url"] == "https://custom/v1"
        assert settings["model"] == "other/model"


class TestApiLlmClientContract:
    def test_no_key_reply_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.speech.llm import LlmError

        monkeypatch.delenv(ENV_API_KEY, raising=False)
        client = ApiLlmClient()
        with pytest.raises(LlmError):
            client.reply([{"role": "user", "content": "hi"}])

    def test_invalid_temperature_raises(self) -> None:
        with pytest.raises(ValueError):
            ApiLlmClient(temperature=3.0)