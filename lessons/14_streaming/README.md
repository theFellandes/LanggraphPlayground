# Lesson 14 · Streaming

## What you'll learn

- The four `stream_mode` values and what each one yields:
  - **`values`** — the full state after each step (bandwidth heavy, simplest)
  - **`updates`** — only the keys that changed in each step (bandwidth efficient)
  - **`messages`** — token-level LLM output (best for chat UIs)
  - **`custom`** — payloads you `writer({...})` from inside a node
- `astream_events(version="v2")` — the *finest-grained* stream, fires for every Runnable lifecycle event
- When to pick each mode

## Why it matters

A chat UI that prints word-by-word feels alive. A long pipeline that
silently hangs for 30 seconds feels broken. Streaming is the
difference. LangGraph supports it natively at four different grains —
pick the one that matches what your UI actually needs.

## Key concepts

- **`graph.stream(input, stream_mode=...)`** — sync iterator over chunks.
- **`graph.astream_events(input, version="v2")`** — async iterator, much finer-grained — every Runnable's `on_start` / `on_end` / `on_chat_model_stream`, etc.
- **`get_stream_writer()`** — call inside a node to emit `custom` events. Lets nodes broadcast progress without polluting state.
- **Picking a mode** — for a chat UI use `messages`; for a server pushing JSON deltas use `updates`; for progress bars use `custom`; for full debugging use `astream_events`.

## Walk through `example.py`

1. A two-node graph (`intro` → `body`) where each node calls `writer({"progress": ...})`.
2. `show_modes(graph)` runs the same graph four times — once per mode — so you can compare the chunk shapes side-by-side.
3. `show_events(graph)` uses `astream_events("v2")` and filters for `on_chat_model_stream` to print token-level output.

## Run it

```bash
uv run python -m lessons.14_streaming.example
```

## Debug it

Put `breakpoint()` inside the `for chunk in graph.stream(...)` loop
and inspect `chunk` — each mode produces a distinctly shaped object.

## Picking a mode (quick reference)

| Mode | Chunk shape | Best for |
|---|---|---|
| `values` | full state dict after each step | demos, dashboards |
| `updates` | `{node_name: partial_state_update}` | server-sent events, bandwidth-sensitive UIs |
| `messages` | `(AIMessageChunk, metadata)` | chat UIs (token-by-token) |
| `custom` | whatever you `writer(...)` | progress bars, telemetry |
| `astream_events("v2")` | per-Runnable lifecycle | deep debugging, custom UIs |

## Try it yourself

- Combine modes by passing `stream_mode=["updates", "messages"]`. Chunks are tagged with their mode in the tuple.
- Replace the `body_node` with one that streams its model call via `llm.astream(...)`; then watch how `messages` mode reflects that.

## Next →

[Lesson 15 · Subgraphs](../15_subgraphs/README.md)
