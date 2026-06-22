"""Google Gemini adapter — defaults to Gemini 2.0 Flash.

Added per `shared/llm/README.md` → "Adding a new provider". Mirrors the
anthropic/openai adapters: one provider file, a `build()` callable, an
explicit API-key guard. The VLM PDF→Markdown extraction path
(`docs/research/vlm-pdf-extraction/`) enters Gemini **only** through
`get_llm("google", ...)`, never a raw SDK call.

Unlike the anthropic/openai adapters, the `langchain_google_genai` import
is **lazy** (inside `build()`). Those two providers are core dependencies
that are always installed, so `base.py` can import their adapters eagerly.
Gemini is an opt-in extra (`uv add langchain-google-genai`), so importing
this module must NOT require the package — otherwise `import shared.llm`
would break for every anthropic/openai user until they install it. The
package is only needed when someone actually builds a Google model.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from shared.settings import settings

DEFAULT_MODEL = "gemini-2.0-flash"


def build(model: str | None = None, **kwargs) -> BaseChatModel:
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to .env or switch "
            "LLM_PROVIDER to anthropic/openai."
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "langchain-google-genai is not installed. Run:\n"
            "    uv add langchain-google-genai"
        ) from exc
    return ChatGoogleGenerativeAI(
        model=model or DEFAULT_MODEL,
        google_api_key=settings.google_api_key,
        **kwargs,
    )
