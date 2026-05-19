"""Anthropic Claude adapter — defaults to Sonnet 4.6."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

from shared.settings import settings

DEFAULT_MODEL = "claude-sonnet-4-6"


def build(model: str | None = None, **kwargs) -> BaseChatModel:
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env or switch "
            "LLM_PROVIDER=openai."
        )
    return ChatAnthropic(
        model=model or DEFAULT_MODEL,
        api_key=settings.anthropic_api_key,
        **kwargs,
    )
