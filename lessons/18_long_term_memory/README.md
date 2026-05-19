# Lesson 18 · Long-term memory (`Store`)

## What you'll learn

- The split between **checkpointer** (per-thread state, lesson 12) and **`Store`** (cross-thread / cross-session memory)
- The `Store` API: `put(namespace, key, value)` and `search(namespace, query=...)`
- The namespace pattern `("memories", user_id)` for per-user memory
- How a node receives the store via the `store` parameter

## Why it matters

A checkpointer remembers a *conversation*. A `Store` remembers a
*user*. Anytime you want "next time Alice logs in, recall her
preferences," the Store is the right tool. It's also how production
agents implement learnings that survive across sessions.

## Key concepts

- **`Store`** — a key/value store organised by `namespace` tuples. `InMemoryStore` is the dev implementation; production swaps in `PostgresStore` or similar.
- **Namespace** — a tuple of strings that scopes a slice of memory. Common pattern: `("memories", user_id)`.
- **`store.put(ns, key, value)`** — write a record under the namespace.
- **`store.search(ns, query=...)`** — list (and optionally semantic-search) records in the namespace.
- **Injecting the store into a node** — define your node as `def node(state, *, store):` and LangGraph will pass the compiled graph's store in at runtime.

## Walk through `example.py`

1. **`remember_node`** — extracts a fact from the user message (if any) and writes it to `("memories", user_id)`.
2. **`reply_node`** — reads every memory we have about the user and builds the system context with it before generating a reply.
3. **`main()`** — three separate threads for the same user, in sequence. The first thread shares a fact ("I love hiking and I'm vegetarian"). Subsequent threads pull that fact from the Store even though they have **no overlapping `thread_id`**.

## Run it

```bash
uv run python -m lessons.18_long_term_memory.example
```

Watch the `[dim]✚ stored: …[/]` lines and the personalised replies in
threads `t2` and `t3` that reference facts learned in `t1`.

## Debug it

Put `breakpoint()` inside `reply_node` and try:

```text
ipdb> pp [item.value for item in store.search(namespace)]
```

That's the user's full memory bank at that moment.

## Try it yourself

- Add a `forget_fact(fact_id)` tool that calls `store.delete(...)`.
- Swap `InMemoryStore` for `PostgresStore` (from `langgraph.store.postgres`) and confirm the memory survives a process restart.
- Add `index={"dims": 1536, "embed": "openai:text-embedding-3-small"}` to enable **semantic search** over memories, and call `store.search(ns, query="vegetarian recipes")`.

## You made it 🎉

That's the curriculum. Next, head to the capstones in [`projects/`](../../projects/):

- **[research_assistant](../../projects/research_assistant/README.md)** — supervisor + tools + LCEL writeup
- **[customer_support_bot](../../projects/customer_support_bot/README.md)** — `create_agent` + middleware + HITL + persistence
- **[rag_qa_api](../../projects/rag_qa_api/README.md)** — FastAPI + LangGraph + Postgres + Docker
