"""Speech module — Intent-Frame dialogue actuation (S3, track Б1).

Re-exports:
    IntentFrame / build_intent_frame / render_messages / describe_affect —
        pure core (frame + chat messages)
    REGISTER_MAX_TOKENS / register_max_tokens — reply length by register
    ConversationHistory — in-memory dialogue buffer
    LlmClient / FakeLlmClient / ApiLlmClient / build_llm_client / LlmError
        / llm_settings_from_env — LLM clients (env: LLM_*)
    SpeechDecision / should_speak / SpeechController — event-triggered speech
    ChatSession — CLI dialogue stand
"""

from .chat import ChatSession
from .controller import SpeechController, SpeechDecision, should_speak
from .history import ConversationHistory
from .intent import (
    REGISTER_MAX_TOKENS,
    IntentFrame,
    build_intent_frame,
    describe_affect,
    register_max_tokens,
    render_messages,
)
from .llm import (
    DEFAULT_MODEL,
    ApiLlmClient,
    FakeLlmClient,
    LlmClient,
    LlmError,
    build_llm_client,
    llm_settings_from_env,
)
from .status import format_status

__all__ = [
    "DEFAULT_MODEL",
    "REGISTER_MAX_TOKENS",
    "ApiLlmClient",
    "ChatSession",
    "ConversationHistory",
    "FakeLlmClient",
    "IntentFrame",
    "LlmClient",
    "LlmError",
    "SpeechController",
    "SpeechDecision",
    "build_intent_frame",
    "build_llm_client",
    "describe_affect",
    "format_status",
    "llm_settings_from_env",
    "register_max_tokens",
    "render_messages",
    "should_speak",
]