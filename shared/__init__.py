"""Shared helpers imported by every lesson and capstone.

Re-exports the two things lessons reach for most often:
  - `settings`  → typed access to env vars
  - `get_llm()` → switchable chat model factory
"""

from shared.settings import settings
from shared.llm import get_llm

__all__ = ["settings", "get_llm"]
