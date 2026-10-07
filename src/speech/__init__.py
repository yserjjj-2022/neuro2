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
    GOAL_INSTRUCTIONS,
    IntentFrame,
    build_intent_frame,
    goal_for_action,
    goal_instruction,
    register_max_tokens,
    render_messages,
    report_reset_intent,
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
    "GOAL_INSTRUCTIONS",
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
    "goal_for_action",
    "goal_instruction",
    "llm_settings_from_env",
    "register_max_tokens",
    "render_messages",
    "report_reset_intent",
    "should_speak",
]
