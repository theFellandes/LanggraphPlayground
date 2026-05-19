"""Switchable LLM provider layer.

The whole project flips between Anthropic and OpenAI via one env var
(`LLM_PROVIDER`). Lessons never instantiate `ChatAnthropic` or
`ChatOpenAI` directly — they call `get_llm()` and stay portable.
"""

from shared.llm.base import get_llm

__all__ = ["get_llm"]
