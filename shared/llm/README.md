# `shared/llm/` — switchable LLM provider layer + fallback chain

This module is the single place the rest of the project asks for a
chat model. Lessons import `get_llm` and stay provider-agnostic.

```python
from shared.llm import get_llm

llm = get_llm()                              # picks provider from .env + fallback
llm = get_llm("openai", temperature=0)       # explicit override
llm = get_llm(model="claude-haiku-4-5")      # different model, same provider
llm = get_llm(with_fallback=False)           # raw single-provider model
```

## How the switch works

`LLM_PROVIDER` in `.env` is read once by `shared/settings.py`. The
`get_llm()` factory in `base.py` looks the name up in `_ADAPTERS` and
delegates to that adapter's `build()` function.

```
.env (LLM_PROVIDER=openai)
        │
        ▼
shared/settings.py  ───►  settings.llm_provider
        │
        ▼
shared/llm/base.py  ───►  get_llm()  ───►  openai_adapter.build()
                                       └►  anthropic_adapter.build()
```

## Fallback chain (Aletheia-inspired)

By default, `get_llm()` returns a **Runnable with fallbacks** — it
tries the primary, and if that errors (quota, network, auth, …) it
moves to the next configured provider.

```python
llm = get_llm()
# behaviour:
#   1. try primary  (anthropic, if LLM_PROVIDER=anthropic)
#   2. on error, try the next provider that has a key   (openai)
#   3. on error, raise the last exception
```

Three knobs:

| What | How |
|---|---|
| Disable fallback | `get_llm(with_fallback=False)` |
| Custom order | `get_llm(provider="anthropic", fallback_chain=["openai"])` |
| Primary key missing | `get_llm()` auto-promotes to the first provider that has a key (logs a warning) |

The plumbing is LangChain's `Runnable.with_fallbacks(...)` — it handles
the try/except across the chain transparently. The returned object is
still a Runnable; `.invoke / .stream / .batch / .ainvoke` work as
normal.

For finer-grained behaviour (quota-error blacklisting, per-provider
rate limiting via tenacity, etc.) see Aletheia's
`src/utils/llm_provider.py`. The simpler `with_fallbacks` approach
here is enough for teaching + most production cases.

## Adding a new provider (e.g. Google Gemini)

1. `uv add langchain-google-genai`
2. Create `shared/llm/google_adapter.py`:

   ```python
   from langchain_google_genai import ChatGoogleGenerativeAI
   from shared.settings import settings

   DEFAULT_MODEL = "gemini-2.0-flash"

   def build(model=None, **kwargs):
       return ChatGoogleGenerativeAI(
           model=model or DEFAULT_MODEL,
           google_api_key=settings.google_api_key,
           **kwargs,
       )
   ```

3. Add `google_api_key: str | None = None` to `shared/settings.py`.
4. Register it in `shared/llm/base.py`:

   ```python
   from shared.llm import google_adapter
   _ADAPTERS = {
       "anthropic": anthropic_adapter,
       "openai":    openai_adapter,
       "google":    google_adapter,  # ← new
   }
   ```

That's the entire change. The new provider is automatically included
in the fallback chain when its key is set.

## Why this pattern

This is the same adapter pattern used in larger multi-provider
systems (e.g. Mitrailleuse, Aletheia). It keeps **one
provider-specific file per provider** and a tiny factory that knows
nothing about any individual SDK.
