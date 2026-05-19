# Lesson 09 · Conditional edges

## What you'll learn

- `add_conditional_edges(source, router_fn, mapping)` — branching based on state
- How to write a **router function**: `(state) → str` whose return value picks the next node
- The discipline of **bounding cycles** with a counter so a misbehaving model can't loop forever
- The "self-improving" pattern: draft → score → (accept | revise → draft …)

## Why it matters

Conditional edges are the difference between a chain and a real agent.
The first time you write `add_conditional_edges`, your graph stops
being a script and starts being a controller — it makes decisions at
runtime. The bounded-cycle pattern is how you keep that controller
from running away.

## Key concepts

- **Conditional edge** — `add_conditional_edges(source, router, mapping)`. The router returns a string key; the mapping translates that key to a destination node (or `END`).
- **Router function** — pure: `(state) → Literal["a", "b", ...]`. Keep it simple — no side effects, no LLM calls (those belong in nodes).
- **Cycle guardrail** — at minimum a counter (`state["revisions"] >= MAX_REVISIONS`) so the routing logic always has an exit. Production graphs often combine this with a `recursion_limit` on `.invoke()`.

## Walk through `example.py`

The graph has two nodes:

- `draft_node` — writes (or rewrites) a draft.
- `score_node` — asks the model to rate it 1–10.

Edges:

```
START → draft → score → ┬─ revise → draft (loops back)
                        └─ done   → END
```

The router checks `score >= 8 OR revisions >= MAX_REVISIONS`. Either
condition exits the loop, so the graph is guaranteed to terminate.

## Run it

```bash
uv run python -m lessons.09_conditional_edges.example
```

## Debug it

Put `breakpoint()` inside the `route` function. Each call shows you
the state at that decision point — by far the most useful spot to
debug branching logic.

## Try it yourself

- Add a third route `"reject"` that goes to a new `apologise` node when the score is below 4.
- Replace `revisions` with `messages: Annotated[list, add_messages]` (using LangGraph's reducer) to keep a history of all drafts.

## Next →

[Lesson 10 · `create_agent`](../10_create_agent/README.md)
