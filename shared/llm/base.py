"""Factory that maps a provider name to an adapter.

Adding a new provider is two steps:
  1. Create `shared/llm/<name>_adapter.py` exposing a `build(model, **kw)` callable
  2. Register it in `_ADAPTERS` below
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from shared.llm import anthropic_adapter, openai_adapter
from shared.settings import settings

AdapterBuild = Callable[..., BaseChatModel]

_ADAPTERS: dict[str, AdapterBuild] = {
    "anthropic": anthropic_adapter.build,
    "openai": openai_adapter.build,
}


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """Return a configured chat model.

    >>> llm = get_llm()                              # uses settings.llm_provider
    >>> llm = get_llm("openai")                      # explicit provider
    >>> llm = get_llm("anthropic", model="claude-haiku-4-5", temperature=0.7)
    """
    provider = (provider or settings.llm_provider).lower()
    if provider not in _ADAPTERS:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Available: {sorted(_ADAPTERS)}. "
            f"Add an adapter in shared/llm/ and register it in base.py."
        )
    model = model or settings.llm_model
    return _ADAPTERS[provider](model=model, **kwargs)
