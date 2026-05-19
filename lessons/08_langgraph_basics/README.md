# Lesson 08 · LangGraph basics

## What you'll learn

- The four building blocks: **state**, **nodes**, **edges**, **compile**
- Why state is a `TypedDict` and why every node returns a **partial** update
- The role of the `START` / `END` sentinels
- How to visualise a graph with `graph.get_graph().draw_ascii()`

## Why it matters

You've now seen LCEL (lesson 02) — that handles linear chains. LangGraph
is the next step up: explicit state + explicit edges. From here on
every "agent" you build is a graph; this lesson is the seed.

## Key concepts

- **State (`TypedDict`)** — the dict that flows through the graph. Every key your nodes might write or read should be declared here.
- **Node** — a function `(state) → partial_state`. **Returns the new keys to merge**, not the whole new state.
- **Edge** — a directed transition. `add_edge("a", "b")` means "always go from a to b after a finishes."
- **`START`** / **`END`** — special node names. Every graph has exactly one entry and at least one exit.
- **`compile()`** — turns the graph definition into a runnable. The compiled graph has the standard `.invoke / .stream / .ainvoke` interface, so it composes with everything you learned in lesson 02.

## Walk through `example.py`

We build the smallest non-trivial graph: write a paragraph, then
critique it. Two nodes, three edges (`START → draft → critique → END`),
and a `TypedDict` with three keys. The `draft_node` only returns
`{"draft": ...}` and the `critique_node` only returns `{"critique": ...}`
— LangGraph merges these partial updates into the running state.

## Run it

```bash
uv run python -m lessons.08_langgraph_basics.example
```

## Debug it

Put `breakpoint()` at the top of `critique_node` and inspect `state` —
notice the `draft` field is already populated by the previous node.

## Try it yourself

- Add a third node `polish` that rewrites the draft using the critique. Edges become `START → draft → critique → polish → END`.
- Switch `State` from `TypedDict` to a Pydantic `BaseModel` and pass `state.draft` instead of `state["draft"]`.

## Next →

[Lesson 09 · Conditional edges](../09_conditional_edges/README.md)
