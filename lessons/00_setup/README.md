# Lesson 00 · Setup

## What you'll learn

- How to install dependencies with `uv`
- How `.env` + `shared/settings.py` give every lesson typed access to env vars
- How `get_llm()` returns a configured chat model without binding you to one provider
- How to invoke a model and read the response

## Why it matters

Every later lesson assumes these two lines work:

```python
from shared import get_llm
llm = get_llm()
```

Get them working now and everything downstream is plug-and-play.

## Key concepts

- **Provider abstraction** — `shared/llm/` is a tiny adapter pattern. You call `get_llm()`, it reads `LLM_PROVIDER` from your `.env`, and returns the matching `ChatModel` (Anthropic or OpenAI). Adding Gemini is one new file — see [shared/llm/README.md](../../shared/llm/README.md).
- **Settings via `pydantic-settings`** — `Settings(BaseSettings)` validates env vars at startup. Misspell a variable name and you find out immediately.
- **`llm.invoke(...)`** — every Runnable in LangChain has `.invoke / .stream / .batch / .ainvoke`. `invoke` is the simplest one: in goes a prompt, out comes an `AIMessage`.

## Walk through `example.py`

1. Import the helpers (`get_llm`, `console`, `section`).
2. Build a chat model with one call — no provider-specific imports here.
3. Print the class name and model id so you can see the switch working.
4. Call `.invoke()` with a plain string and print the response.

## Run it

First-time setup:

```bash
uv sync                       # installs everything
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY (or OPENAI_API_KEY if you flipped LLM_PROVIDER)
```

Then run:

```bash
uv run python -m lessons.00_setup.example
```

## Debug it

Put a `breakpoint()` right before `llm.invoke(...)` and run:

```bash
PYTHONBREAKPOINT=ipdb.set_trace uv run python -m lessons.00_setup.example
```

At the prompt, try `pp llm.__dict__` to inspect the configured model.
You'll learn the proper workflow in [lesson 03](../03_debugging/README.md).

## Try it yourself

- Flip `LLM_PROVIDER` between `anthropic` and `openai` in `.env` and re-run. The output changes but `example.py` doesn't.
- Pass `temperature=0` and `temperature=1.5` to `get_llm(...)` and see how the response changes.

## Next →

[Lesson 01 · Chat models](../01_chat_models/README.md)
