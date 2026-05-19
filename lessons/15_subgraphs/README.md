# Lesson 15 · Subgraphs

## What you'll learn

- How to **compose** a graph by calling another compiled graph from inside a node
- Why subgraphs can have their own state shape, totally decoupled from the parent's
- The **reducer pattern** for accumulating values: `Annotated[list[str], add]`
- How to wrap a subgraph as a parent node when the parent calls it multiple times with different inputs

## Why it matters

Once your graphs get bigger than ~5 nodes, you want to break them up.
Subgraphs let you encapsulate a sub-workflow (with its own state and
its own internal flow) and reuse it. They're the natural unit for
sharing logic across capstones.

## Key concepts

- **Subgraph = a compiled `StateGraph`** — it has the standard `.invoke / .stream / .ainvoke` interface, so it's just a callable from the parent's point of view.
- **Different state shapes are fine** — the subgraph defines `SectionState`; the parent defines `PostState`. The parent node wrapping the subgraph is responsible for translating between them.
- **Reducers via `Annotated[type, fn]`** — `Annotated[list[str], add]` tells LangGraph "when nodes return `sections`, **append** instead of replacing." This is essential for fan-in patterns.

## Walk through `example.py`

1. **Subgraph** (`SectionState`) — two nodes: `draft` then `polish`. Returns a polished string.
2. **Parent graph** (`PostState`) — two nodes: `intro_section` and `middle_section`. Each is a thin wrapper around the subgraph invoked with a different `topic`. Each wrapper returns `{"sections": [polished]}`, and the `add` reducer appends them into the parent's `sections` list.

Notice the parent's `sections` list grows to two items by the end, even
though each node returned a one-element list. That's the reducer doing
its job.

## Run it

```bash
uv run python -m lessons.15_subgraphs.example
```

## Debug it

Put `breakpoint()` inside `_node` (the closure in `write_section`) and
inspect `out` — that's the subgraph's terminal state, before you
translate it into the parent's shape.

## Try it yourself

- Add a third section. The reducer handles fan-in automatically.
- Convert the two parent nodes into a **parallel** branch: `START → [intro, middle] → END`. The reducer still accumulates correctly.
- Wrap the subgraph with `subgraph.with_config(...)` to give it a different `recursion_limit` than the parent.

## Next →

[Lesson 16 · Supervisor](../16_supervisor/README.md)
