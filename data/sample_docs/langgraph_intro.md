# LangGraph in a nutshell

LangGraph is a low-level orchestration framework for building stateful,
multi-actor LLM applications. It treats a workflow as a **graph of
nodes connected by edges**, where every node receives the current state
and returns a partial update.

## Why a graph?

Most agent frameworks model an agent as a loop: call the model, decide
whether to call a tool, call it, loop again. That works until you need
two agents to talk, or a user to approve a step, or a long task to
survive a process restart. The graph model handles all of those by
treating each step as a first-class node with explicit edges.

## Core concepts

- **State** — a typed dict describing what flows between nodes. LangGraph
  ships with `MessagesState` for chat-style apps, but you can roll
  your own with `TypedDict` or a Pydantic model.
- **Node** — a function `(state) -> partial_state`. Nodes are pure: they
  return updates instead of mutating state in place.
- **Edge** — a directed transition. Edges can be static (always go from
  A to B) or conditional (the routing function decides at runtime).
- **Checkpointer** — a persistence layer that writes state after every
  step, keyed by a `thread_id`. Swap `MemorySaver` for `SqliteSaver`
  or `PostgresSaver` without touching the graph definition.
- **Interrupt** — a pause point. The graph stops, returns the current
  state, and waits for a human to call `Command(resume=...)`.

## How it differs from LCEL

LCEL is great for one-shot pipelines: prompt → model → parser. The
moment you need branching, cycles, or persistence, you graduate to
LangGraph.
