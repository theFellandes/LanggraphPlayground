"""Typed environment access for the whole project.

Lessons should do:

    from shared.settings import settings
    print(settings.llm_provider)

instead of reading `os.environ` directly. This keeps the source of
truth in one place and gives you autocomplete + validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

Provider = Literal["anthropic", "openai", "google", "local"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: Provider = Field(default="anthropic")
    llm_model: str | None = Field(default=None)

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None

    # Bring-your-own VLM: any OpenAI-compatible server (vLLM / Ollama / LM Studio / TGI).
    # Set local_api_key to any token your server accepts (often a dummy like "EMPTY").
    local_api_key: str | None = None
    local_vlm_base_url: str | None = None
    local_vlm_model: str | None = None

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "langgraph-playground"

    tavily_api_key: str | None = None

    postgres_url: str = "postgresql://postgres:postgres@localhost:5432/langgraph"

    @property
    def data_dir(self) -> Path:
        d = ROOT / "data"
        d.mkdir(exist_ok=True)
        return d


settings = Settings()

if settings.langsmith_tracing and settings.langsmith_api_key:
    import os

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
