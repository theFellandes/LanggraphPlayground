# `shared/llm/` — Switchable LLM provider layer

This module is the single place the rest of the project asks for a chat
model. Lessons import `get_llm` and stay provider-agnostic.

```python
from shared.llm import get_llm

llm = get_llm()                              # picks provider from .env
llm = get_llm("openai", temperature=0)       # explicit override
llm = get_llm(model="claude-haiku-4-5")      # different model, same provider
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

## Adding a new provider (e.g. Google Gemini)

1. `pip install langchain-google-genai`
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
       "anthropic": anthropic_adapter.build,
       "openai":    openai_adapter.build,
       "google":    google_adapter.build,  # ← new
   }
   ```

That's the entire change. No lesson code needs to be touched.

## Why this pattern

This is the same adapter pattern used in larger multi-provider systems
(e.g. Mitrailleuse). It keeps **one provider-specific file per provider**
and a tiny factory that knows nothing about any individual SDK.
