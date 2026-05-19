# Lesson 16 · Supervisor

## What you'll learn

- The **supervisor pattern**: one router agent + N specialist worker agents
- How to build it in one call with `langgraph_supervisor.create_supervisor(...)`
- How `name=` on `create_agent` lets the supervisor route by name
- When supervisor beats swarm (lesson 17) — and vice-versa

## Why it matters

The supervisor pattern is the most widely deployed multi-agent
architecture in production (per the LangGraph team). It gives you
**centralised control** — easy to reason about, easy to debug,
easy to add a worker — at the cost of an extra LLM hop per delegation.

## Key concepts

- **`create_supervisor(agents, model, prompt)`** — returns a graph builder you `.compile()` to get the runnable.
- **Workers are just agents** — each is a `create_agent(...)` with a `name="..."` so the supervisor can address it.
- **Routing prompt** — the supervisor's system prompt tells the LLM what each worker is good at. Be specific; that prompt is the entire routing logic.
- **Handoff back to supervisor** — workers reply, the supervisor reads the result, decides whether more delegation is needed, and otherwise speaks to the user directly.

## Walk through `example.py`

1. Build `math_agent` (tools: `add`, `multiply`).
2. Build `travel_agent` (tools: `get_weather`, `get_local_dish`).
3. Build `supervisor` with both as workers.
4. Ask a question that needs both. Watch the supervisor route to `travel_expert` for the weather and `math_expert` for the arithmetic, then synthesise.

## Run it

```bash
uv run python -m lessons.16_supervisor.example
```

## Debug it

Put `breakpoint()` after `supervisor.invoke(...)` and inspect
`result["messages"]`. Each delegation appears as a message with the
worker's name in the metadata — that's how you trace what the
supervisor decided.

## Supervisor vs Swarm (quick reference)

| | Supervisor | Swarm (lesson 17) |
|---|---|---|
| Topology | Star (boss in the middle) | Mesh (peers) |
| Routing | One LLM decides every hop | Each agent decides when to hand off |
| Easy to debug? | Yes — one place to look | No — handoffs are distributed |
| Extra LLM hops | One per delegation | Zero |
| Use when… | You want central control | Agents truly own different phases of work |

## Try it yourself

- Add a third worker (e.g. `language_expert` that translates strings) and see how the routing prompt needs to grow.
- Replace `create_supervisor`'s `model` with a cheaper one (e.g. Haiku) while keeping workers on a stronger model — that's a common cost optimisation.

## Next →

[Lesson 17 · Swarm](../17_swarm/README.md)
