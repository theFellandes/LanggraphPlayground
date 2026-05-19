"""OpenAI GPT adapter — defaults to GPT-4.1."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from shared.settings import settings

DEFAULT_MODEL = "gpt-4.1"


def build(model: str | None = None, **kwargs) -> BaseChatModel:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or switch "
            "LLM_PROVIDER=anthropic."
        )
    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        api_key=settings.openai_api_key,
        **kwargs,
    )
