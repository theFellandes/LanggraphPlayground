# Lesson 12 · Persistence

## What you'll learn

- The **checkpointer** abstraction: `MemorySaver` (in-RAM, dev), `SqliteSaver` (on-disk, local), `PostgresSaver` (production)
- The role of `thread_id` in scoping state to a conversation
- How `graph.get_state(config)` reads the current checkpoint
- How `graph.get_state_history(config)` exposes **every** past checkpoint
- **Time-travel**: rewind to a past checkpoint and branch a new conversation from there

## Why it matters

Persistence is the thing that turns a graph into a *system*. Without
it, every `.invoke()` starts from scratch. With it, you have:

- Multi-turn chat (lesson 13 builds on this)
- Crash recovery — restart the process, keep going
- Time-travel debugging — re-run from any past step

Same graph code, different checkpointer in one line.

## Key concepts

- **Checkpointer** — a pluggable backend that writes state after every node. Pass it to `.compile(checkpointer=...)`.
- **`thread_id`** — the scope key. All checkpoints for a single conversation share one `thread_id`. Different `thread_id`s give you isolated conversations on the same graph.
- **`config={"configurable": {"thread_id": "..."}}`** — pass this dict to `.invoke / .stream` and the graph attaches new checkpoints under that thread.
- **`get_state(config)`** — current snapshot for that thread (state values + metadata).
- **`get_state_history(config)`** — iterator over every checkpoint, newest first. Each entry has a `.config` you can replay from.

## Walk through `example.py`

1. **`memory_demo()`** — two threads (Alice and Bob) share one graph but never see each other's state. Demonstrates that "memory" is just "the right `thread_id`".
2. **`sqlite_demo()`** — same graph compiled with `SqliteSaver`. The file at `data/lesson12.sqlite` persists between runs.
3. **`time_travel_demo()`** — three turns, then rewind to the checkpoint after turn 2 and ask a *different* turn 3. The new branch coexists with the original; the thread is a tree, not a line.

## Run it

```bash
uv run python -m lessons.12_persistence.example
```

Run it twice. On the second run, Part 2's thread keeps the state from
the first run — that's the SqliteSaver doing its job.

## Debug it

Put `breakpoint()` after `history = list(graph.get_state_history(cfg))`
in Part 3 and explore:

```text
ipdb> for h in history: print(h.config['configurable']['checkpoint_id'], h.values.get('messages')[-1].content[:40])
```

That's the easiest way to find the exact checkpoint you want to rewind to.

## Try it yourself

- Delete `data/lesson12.sqlite` and re-run — observe a fresh thread.
- Open the SQLite file with any DB browser. Notice every step is one row.

## Next →

[Lesson 13 · Human-in-the-loop](../13_human_in_the_loop/README.md)
