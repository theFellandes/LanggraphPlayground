# Lesson 02 · LCEL chains

## What you'll learn

- The pipe operator `|` and what it actually produces
- `RunnableParallel` for fan-out, `RunnableLambda` for arbitrary functions, `RunnablePassthrough.assign` for adding fields mid-chain
- All four execution modes: `.invoke`, `.batch`, `.stream`, `.ainvoke`
- How `StrOutputParser` cleans up an `AIMessage` into a plain `str`

## Why it matters

LCEL is the lowest-friction way to compose LLM logic in LangChain. A
`prompt | model | parser` chain gives you streaming, batching, and
async for free — you didn't write any of that yourself. Once you're
comfortable here, you'll naturally reach for LangGraph the moment a
chain needs loops, branching, or persistent state.

## Key concepts

- **Runnable** — any object with `.invoke`, `.stream`, `.batch`, `.ainvoke`. Everything in LCEL is a Runnable.
- **Pipe operator** — `a | b` returns a `RunnableSequence(a, b)`. Output of `a` becomes input of `b`.
- **`RunnableParallel`** — runs multiple sub-chains on the same input and returns a dict of their outputs.
- **`RunnableLambda`** — wraps a plain Python function so it joins the chain.
- **`RunnablePassthrough.assign`** — keeps the original input and adds new keys to it.

## Walk through `example.py`

Six small parts, each demonstrating one capability:

| Part | Demonstrates |
|---|---|
| 1 | `prompt \| model \| StrOutputParser` — the canonical chain |
| 2 | `RunnableParallel` — fan-out to three branches |
| 3 | `RunnablePassthrough.assign` + `RunnableLambda` — enrich inputs mid-chain |
| 4 | `.stream()` — print chunks as they arrive |
| 5 | `.batch()` — concurrent calls for many inputs |
| 6 | `.ainvoke()` — the async twin |

## Run it

```bash
uv run python -m lessons.02_lcel_chains.example
```

## Debug it

In part 2, set a `breakpoint()` after `fan_out.invoke(...)` and inspect
the `result` dict. Notice all three branches ran without any explicit
threading code on your part.

## Try it yourself

- Combine parts 2 and 4: stream a `RunnableParallel` and see chunks from different branches interleave.
- Write a `RunnableLambda` that uppercases the model's output and add it to the chain.

## Next →

[Lesson 03 · Debugging](../03_debugging/README.md)
