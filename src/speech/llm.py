"""LLM client — OpenAI-compatible chat (RouterAI by default).

Functional Core / Imperative Shell (ADR-0004):
    - ``LlmClient`` — Protocol for DI (like ``Embedder`` in memory).
    - ``FakeLlmClient`` — deterministic short reply template (tests/replay).
    - ``ApiLlmClient`` — shell: OpenAI-compatible chat, lazy client.
    - ``build_llm_client`` — factory honouring ``llm_mode`` (auto/fake/api).
    - ``llm_settings_from_env`` — read ``LLM_*`` settings.

Configuration is read from the environment (``.env`` is loaded by the CLI):
    LLM_API_KEY   — API key (optional override)
    LLM_BASE_URL  — OpenAI-compatible base URL (default: RouterAI)
    LLM_MODEL     — chat model id (default: deepseek/deepseek-v4.1-flash)

RouterAI uses a single account key for all models, so when ``LLM_API_KEY`` is
absent the client falls back to ``EMBEDDER_API_KEY`` (the universal key). An
explicit ``LLM_API_KEY`` overrides it.

Unlike the embedder, a real chat reply is non-deterministic (temperature), so
tests assert structure, not text. ``llm_mode="api"`` without a key is
fail-fast (``ValueError``); ``auto`` falls back to fake.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://routerai.ru/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"

ENV_API_KEY = "LLM_API_KEY"
ENV_BASE_URL = "LLM_BASE_URL"
ENV_MODEL = "LLM_MODEL"

# RouterAI: один ключ на все модели. Fallback на универсальный ключ.
FALLBACK_API_KEY_ENV = "EMBEDDER_API_KEY"


def _resolve_api_key(explicit: str | None = None) -> str | None:
    """Ключ API: явный → LLM_API_KEY → универсальный EMBEDDER_API_KEY."""
    return (
        explicit or os.environ.get(ENV_API_KEY) or os.environ.get(FALLBACK_API_KEY_ENV)
    )


class LlmError(Exception):
    """Оборачивает сбои LLM (сеть, API, пустой ответ)."""


@runtime_checkable
class LlmClient(Protocol):
    """Контракт LLM-клиента: messages → текст ответа."""

    def reply(self, messages: list[dict[str, Any]], max_tokens: int = 256) -> str:
        """Сгенерировать ответ по истории сообщений.

        Args:
            messages: Chat messages [{role, content}, ...].
            max_tokens: Лимит токенов ответа.

        Returns:
            Текст ответа (непустой).

        Raises:
            LlmError: При сбое LLM.
        """
        ...


@dataclass(frozen=True)
class FakeLlmClient:
    """Детерминированный LLM-клиент для тестов и offline-стенда.

    Возвращает стабильный короткий шаблон, зависящий от register-подсказки в
    system-сообщении: так тесты видят, что режим дошёл до клиента. Реальный
    текст не имитируется.

    Attributes:
        prefix: Префикс шаблонного ответа.
    """

    prefix: str = "ok"

    def reply(self, messages: list[dict[str, Any]], max_tokens: int = 256) -> str:
        """Шаблонный ответ (детерминированный).

        Args:
            messages: Chat messages.
            max_tokens: Лимит токенов (учитывается: terse → короче).

        Returns:
            Непустая строка.
        """
        system = messages[0]["content"] if messages else ""
        if "междометием" in system:
            return self.prefix
        if "одной-двумя" in system:
            return f"{self.prefix}, понял тебя."
        return f"{self.prefix}, вот мой ответ."


@dataclass
class ApiLlmClient:
    """OpenAI-совместимый chat-клиент (Shell, lazy client).

    По умолчанию — RouterAI (https://routerai.ru). Клиент создаётся при
    первом вызове ``reply`` (не в ``__init__``).

    Reasoning отключён по умолчанию: LLM здесь — речевой актюатор (зона
    Брока), а не центр рассуждения (ADR-0003, ADR-0007). Reasoning-модели
    (напр. deepseek-v4.1-flash) иначе тратят токены на «размышление» и
    возвращают пустой ``content``, а также сильнее переинтерпретируют
    Intent-Frame.

    Attributes:
        model: Идентификатор модели чата.
        base_url: OpenAI-совместимый base URL.
        api_key: Ключ API; None → LLM_API_KEY или универсальный EMBEDDER_API_KEY.
        temperature: Температура генерации.
        reasoning: Включить reasoning-режим (по умолчанию False).
    """

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    temperature: float = 0.7
    reasoning: bool = False
    _client: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")

    def _ensure_client(self) -> object:
        """Лениво создать клиент (первый вызов reply).

        Returns:
            Клиент OpenAI.

        Raises:
            LlmError: Если ключ не найден или пакет openai недоступен.
        """
        if self._client is not None:
            return self._client

        key = _resolve_api_key(self.api_key)
        if not key:
            raise LlmError(
                f"no API key: pass api_key or set {ENV_API_KEY} "
                f"(or {FALLBACK_API_KEY_ENV})"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - зависимость объявлена
            raise LlmError(f"openai package unavailable: {exc}") from exc

        self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    def reply(self, messages: list[dict[str, Any]], max_tokens: int = 256) -> str:
        """Сгенерировать ответ через OpenAI-совместимый API.

        Args:
            messages: Chat messages.
            max_tokens: Лимит токенов ответа.

        Returns:
            Текст ответа (непустой).

        Raises:
            LlmError: При сбое сети/API или пустом ответе.
        """
        client = self._ensure_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
                extra_body={"reasoning": {"enabled": self.reasoning}},
            )
        except Exception as exc:  # любые сбои → LlmError
            raise LlmError(f"chat request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LlmError("empty LLM response")
        return content.strip()


def build_llm_client(
    mode: str = "auto",
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    temperature: float = 0.7,
    reasoning: bool = False,
) -> LlmClient:
    """Фабрика LLM-клиента по режиму.

    Args:
        mode: "auto" (ключ→api, иначе fake), "fake", "api".
        model: Модель чата.
        base_url: OpenAI-совместимый base URL.
        api_key: Ключ API; None → LLM_API_KEY или EMBEDDER_API_KEY.
        temperature: Температура генерации.
        reasoning: Включить reasoning-режим (по умолчанию выключен).

    Returns:
        LlmClient согласно режиму.

    Raises:
        ValueError: Если mode неизвестен, или mode="api" без ключа.
    """
    key = _resolve_api_key(api_key)
    if mode == "fake":
        return FakeLlmClient()
    if mode == "api":
        if not key:
            raise ValueError(
                f"llm_mode='api' requires a key "
                f"(argument, {ENV_API_KEY} or {FALLBACK_API_KEY_ENV})"
            )
        return ApiLlmClient(
            model=model,
            base_url=base_url,
            api_key=key,
            temperature=temperature,
            reasoning=reasoning,
        )
    if mode == "auto":
        if key:
            logger.info(
                "llm: auto → api (model=%s, base_url=%s, reasoning=%s)",
                model,
                base_url,
                reasoning,
            )
            return ApiLlmClient(
                model=model,
                base_url=base_url,
                api_key=key,
                temperature=temperature,
                reasoning=reasoning,
            )
        logger.info("llm: auto → fake (no API key)")
        return FakeLlmClient()
    raise ValueError(f"unknown llm_mode {mode!r} (auto|fake|api)")


def llm_settings_from_env() -> dict[str, str]:
    """Собрать настройки LLM из окружения (для wiring/CLI).

    Returns:
        Словарь с ``base_url`` и ``model`` (из env, с дефолтами).
    """
    return {
        "base_url": os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL),
        "model": os.environ.get(ENV_MODEL, DEFAULT_MODEL),
    }
