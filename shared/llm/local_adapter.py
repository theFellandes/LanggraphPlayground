"""Bring-your-own-VLM adapter — any OpenAI-compatible endpoint.

This is how you run the extraction pipeline on **your own** multimodal model
instead of Gemini/Claude. vLLM, Ollama, LM Studio, and HF TGI all expose the
OpenAI chat-completions API, so we reuse ``ChatOpenAI`` pointed at your server's
``base_url`` — no new SDK, no new client. Once wired you call it exactly like any
other provider::

    from shared.llm import get_llm
    vlm = get_llm("local", with_fallback=False, temperature=0)

Configure via .env (see ``.env.example``)::

    LOCAL_VLM_BASE_URL=http://localhost:8000/v1      # your server
    LOCAL_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct       # whatever it serves
    LOCAL_API_KEY=EMPTY                               # any token the server accepts

``langchain-openai`` is a core dependency, so this import is eager like the
openai/anthropic adapters.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from shared.settings import settings

# Example open-weights VLM; override with LOCAL_VLM_MODEL.
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def build(model: str | None = None, **kwargs) -> BaseChatModel:
    if not settings.local_vlm_base_url:
        raise RuntimeError(
            "LOCAL_VLM_BASE_URL is not set. Point it at your OpenAI-compatible VLM "
            "server (vLLM / Ollama / LM Studio / TGI), e.g. http://localhost:8000/v1, "
            "and set LOCAL_API_KEY to any token the server accepts."
        )
    return ChatOpenAI(
        model=model or settings.local_vlm_model or DEFAULT_MODEL,
        base_url=settings.local_vlm_base_url,
        api_key=settings.local_api_key or "EMPTY",
        **kwargs,
    )
