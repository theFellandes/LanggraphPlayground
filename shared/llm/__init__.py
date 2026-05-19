"""Switchable LLM provider layer with multi-provider fallback.

The whole project flips between Anthropic and OpenAI via one env var
(`LLM_PROVIDER`). Lessons never instantiate `ChatAnthropic` or
`ChatOpenAI` directly — they call `get_llm()` and stay portable.

By default `get_llm()` returns a Runnable that **tries the primary
provider first and falls back to the other configured providers** on
any error (Aletheia-style). Opt out with `with_fallback=False`.
"""

from shared.llm.base import available_providers, get_llm

__all__ = ["get_llm", "available_providers"]
