"""Factory that maps a provider name to an adapter — with fallback.

Behaviour (Aletheia-style):

  1. Resolve a primary provider (explicit arg → settings.llm_provider).
  2. If the primary has no API key configured, auto-promote to the
     first available provider (so a misconfigured .env doesn't crash
     every lesson at import time).
  3. By default, return a Runnable that **tries the primary first and
     falls back to the OTHER configured providers** on any error.
  4. Opt out of the fallback chain with `with_fallback=False` if you
     want the raw single-provider model (useful for tests, or when
     you care about deterministic provider identity).

The fallback plumbing is LangChain's `Runnable.with_fallbacks(...)` —
it handles quota errors, network blips, and authentication failures
transparently. For finer-grained quota detection / blacklisting /
per-provider rate limits, see Aletheia's `src/utils/llm_provider.py`.

Adding a new provider:

  1. Create `shared/llm/<name>_adapter.py` with a `build(model, **kw)` callable.
  2. Add `<name>_api_key` to `shared/settings.py`.
  3. Register the adapter in `_ADAPTERS` below.

Nothing else in the codebase needs to change.
"""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from shared.llm import anthropic_adapter, google_adapter, local_adapter, openai_adapter
from shared.settings import settings

logger = logging.getLogger(__name__)

# Ordered registry. Insertion order = default fallback preference.
_ADAPTERS = {
    "anthropic": anthropic_adapter,
    "openai":    openai_adapter,
    "google":    google_adapter,
    "local":     local_adapter,   # bring-your-own VLM (OpenAI-compatible server)
}


def _has_key(provider: str) -> bool:
    """True iff a non-empty API key is configured for `provider`."""
    return bool(getattr(settings, f"{provider}_api_key", None))


def available_providers() -> list[str]:
    """Return provider names that have a configured API key.

    Used to build the fallback chain. A provider without a key is
    skipped — including it would raise on construction.
    """
    return [p for p in _ADAPTERS if _has_key(p)]


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    *,
    with_fallback: bool = True,
    fallback_chain: list[str] | None = None,
    **kwargs,
) -> Runnable:
    """Return a configured chat model, optionally with multi-provider fallback.

    Args:
        provider: Explicit primary provider name. Defaults to settings.llm_provider.
        model: Optional model id override.
        with_fallback: When True (default), the returned Runnable falls
            back to the OTHER configured providers on any error. When
            False, the bare primary-provider model is returned.
        fallback_chain: Explicit list of provider names to use as
            fallbacks (after the primary). Defaults to "every other
            provider that has a key, in registration order."
        **kwargs: Forwarded to each provider's adapter (e.g. `temperature`).

    Returns:
        A LangChain Runnable. Use `.invoke / .stream / .batch / .ainvoke`
        as normal — same surface as a single `BaseChatModel`.

    Raises:
        RuntimeError: If no provider has an API key configured.
        ValueError:   If an unknown provider name is supplied.

    Example:
        >>> llm = get_llm()                              # primary + fallback
        >>> llm = get_llm("openai", model="gpt-4o")      # explicit primary
        >>> llm = get_llm(with_fallback=False)           # bare, no fallback
    """
    requested = (provider or settings.llm_provider).lower()
    if requested not in _ADAPTERS:
        raise ValueError(
            f"Unknown provider {requested!r}. Available: {sorted(_ADAPTERS)}"
        )

    available = available_providers()
    if not available:
        raise RuntimeError(
            "No LLM provider has an API key configured. Add ANTHROPIC_API_KEY or "
            "OPENAI_API_KEY to your .env."
        )

    # Auto-promote: if the requested primary has no key, fall through to
    # whichever provider does (warn loudly).
    if requested not in available:
        promoted = available[0]
        logger.warning(
            "Requested LLM provider %r has no API key. Auto-promoting to %r. "
            "Set %s_API_KEY to use %r as primary.",
            requested, promoted, requested.upper(), requested,
        )
        primary = promoted
    else:
        primary = requested

    primary_llm: BaseChatModel = _ADAPTERS[primary].build(model=model, **kwargs)

    if not with_fallback:
        return primary_llm

    # Resolve the fallback chain. Default: every other available provider.
    others = (
        fallback_chain
        if fallback_chain is not None
        else [p for p in available if p != primary]
    )
    fallbacks: list[BaseChatModel] = []
    for name in others:
        if name == primary:
            continue
        if name not in _ADAPTERS:
            logger.warning("Skipping unknown fallback provider %r", name)
            continue
        if not _has_key(name):
            logger.warning("Skipping fallback %r — no API key configured", name)
            continue
        # NOTE: fallbacks ignore `model` — let each provider use its own default.
        fallbacks.append(_ADAPTERS[name].build(**kwargs))

    if not fallbacks:
        return primary_llm

    return primary_llm.with_fallbacks(fallbacks)
