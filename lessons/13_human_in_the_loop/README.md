# Lesson 13 · Human-in-the-loop (HITL)

## What you'll learn

- The `interrupt()` primitive — pause a graph and yield control to the caller
- `Command(resume=value)` — how the caller wakes the graph up and feeds it a decision
- Why a **checkpointer is required** for HITL (the paused state has to survive)
- The right pattern for "approve / reject / edit" gates around risky side effects
- When to reach for graph-level `interrupt()` vs. the prebuilt `human_in_the_loop_middleware` from lesson 11

## Why it matters

The moment your agent does anything irreversible — sends an email,
charges a card, escalates a ticket — you want a human in the loop.
`interrupt()` is the clean way to bake that pause directly into the
graph topology so callers must explicitly resume it.

## Key concepts

- **`interrupt(payload)`** — called inside a node. Pauses execution and returns `payload` to the caller as part of the run result. Re-invoking the graph with `Command(resume=value)` causes `interrupt()` to *return* `value` and the node continues from there.
- **Checkpointer is mandatory** — the graph has to persist its position so it can resume. `MemorySaver` is fine for demos; production wants `SqliteSaver` or `PostgresSaver`.
- **Resume value shape** — totally up to you. We send `{"edited": "..."}` to mean "approve but with this rewrite."

## Walk through `example.py`

1. **`draft_node`** writes the email.
2. **`approval_node`** calls `interrupt({...})`. The graph pauses *here*. The caller sees the interrupt payload in the run result.
3. The "human" (your second `graph.invoke(...)`) sends `Command(resume={"edited": "..."})`. That value becomes the return of `interrupt(...)`, the node continues, and the graph proceeds to `send_node`.

## Run it

```bash
uv run python -m lessons.13_human_in_the_loop.example
```

## Debug it

Put `breakpoint()` right after `interrupt(...)` returns and inspect
what came back from `Command(resume=...)`. This is exactly where bugs
between caller and graph manifest.

## `interrupt()` vs `human_in_the_loop_middleware`

| | `interrupt()` (graph node) | `human_in_the_loop_middleware` (lesson 11) |
|---|---|---|
| Where you put it | Inside a custom node you wrote | On a `create_agent` agent |
| Scope of pause | Anywhere you call it | Around tool calls |
| Best when | You already have a hand-built `StateGraph` | You're using `create_agent` and want a no-code approval gate |
| Coupling | Tight — the pause is part of the topology | Loose — middleware plugs in/out |

## Try it yourself

- Wire two separate "human" steps in the same graph — pause for a draft review, then again for a final send-off.
- Change `Command(resume={"edited": ...})` to `Command(resume="reject")` and watch the `send_node` skip.

## Next →

[Lesson 14 · Streaming](../14_streaming/README.md)
