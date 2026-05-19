# Lesson 03 · Debugging with `ipdb`

## What you'll learn

- The `breakpoint()` / `PYTHONBREAKPOINT=ipdb.set_trace` workflow
- The ipdb commands you'll use 90% of the time
- How to debug **inside async code** (`async def` + `await`)
- `Runnable.with_listeners()` — non-interactive logging of every chain step
- A peek at **LangSmith** as the "ipdb for production"

## Why it matters

You will spend more time debugging chains and graphs than writing
them. Print statements work for one bug; for the second bug you want
to *step* through the code with full state visibility. `ipdb` is the
right default tool — colored tracebacks, tab completion, and a tiny
command surface you can memorise.

## Key concepts

- **`breakpoint()`** (PEP 553) — the modern way to drop a debugger statement. Honors the `PYTHONBREAKPOINT` env var, so you can swap `pdb` for `ipdb` (or disable entirely with `PYTHONBREAKPOINT=0`) without editing code.
- **`PYTHONBREAKPOINT=ipdb.set_trace`** — wire `breakpoint()` to ipdb for one run.
- **ipdb commands** — `n` next, `s` step into, `c` continue, `p` print, `pp` pretty-print, `w` where, `l` list, `u/d` up/down the stack, `q` quit. `!expr` runs arbitrary Python.
- **`Runnable.with_listeners(on_start, on_end, on_error)`** — attach side-effects to every step of a chain. Use this for structured logging when you don't want to pause execution.
- **LangSmith** — the production debugger. Set `LANGSMITH_TRACING=true` and every LCEL/LangGraph run becomes a clickable trace in the UI.

## Walk through `example.py`

| Part | What it shows |
|---|---|
| 1 | A planted `breakpoint()` — practice the ipdb commands. |
| 2 | `with_listeners()` — non-interactive `on_start` / `on_end` callbacks log every step. |
| 3 | `breakpoint()` inside an `async` function — `await` resolves at the ipdb prompt too. |
| 4 | A note on enabling LangSmith for prod-style observability. |

## Run it

The whole point is to run with ipdb wired in:

```bash
PYTHONBREAKPOINT=ipdb.set_trace uv run python -m lessons.03_debugging.example
```

At each `breakpoint()` prompt, try:

```text
ipdb> pp chain
ipdb> pp payload
ipdb> p type(chain).__mro__
ipdb> n           # step over
ipdb> c           # continue to the next breakpoint
```

To run **without** stopping (e.g. when revisiting later):

```bash
PYTHONBREAKPOINT=0 uv run python -m lessons.03_debugging.example
```

## Debug it (meta)

Inside the async breakpoint in Part 3, try invoking the chain right
from the prompt:

```text
ipdb> await chain.ainvoke({"language": "French"})
```

This works because ipdb has full async support — you can interactively
probe coroutines mid-flight.

## Try it yourself

- Add a `breakpoint()` inside the `on_end` listener in Part 2 and inspect a `Run` object live.
- Put `breakpoint()` inside a `RunnableLambda` and trace through a multi-step chain.

## Next →

[Lesson 04 · Structured output](../04_structured_output/README.md)
