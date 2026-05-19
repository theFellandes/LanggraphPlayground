# Lesson 17 · Swarm

## What you'll learn

- The **swarm pattern**: peer agents pass control directly, no central router
- `create_handoff_tool(agent_name=..., description=...)` — handoffs are just specially-shaped tools
- `create_swarm(agents, default_active_agent=...)` — wires them up
- How a swarm differs from a supervisor (lesson 16)

## Why it matters

When two or three agents truly own different *phases* of a workflow,
making them peers is leaner than putting a supervisor in front. No
extra LLM hop per delegation, no router prompt to keep in sync.
Trade-off: the routing logic is distributed — each agent has to know
when to hand off.

## Key concepts

- **Handoff tool** — `create_handoff_tool(agent_name, description)` returns a `@tool` that, when called, transfers conversation control to `agent_name`. From the LLM's perspective, handing off looks identical to any other tool call.
- **`create_swarm(agents, default_active_agent)`** — compiles a graph where each agent is a node and handoff tools become edges between them.
- **Active agent state** — the swarm tracks which agent currently "owns" the conversation; handoff tools mutate that field.

## Walk through `example.py`

1. **`triage`** — front desk. Has one tool: `hand_to_refunder`. Decides whether to answer directly or hand off.
2. **`refunder`** — specialist. Has `process_refund` (the real side effect) plus `hand_to_triage` so it can return control after work is done.
3. **`create_swarm(...)`** wires them up; we ask a refund question and watch the handoff happen.

## Run it

```bash
uv run python -m lessons.17_swarm.example
```

## Debug it

Put `breakpoint()` after `swarm.invoke(...)` and look for tool calls
whose name starts with `transfer_to_` — those are the handoffs.

## Supervisor vs Swarm — when to choose what

- Pick **supervisor** when: you want one place to change routing, you have many specialists, or you need a stable "boss" the user can address.
- Pick **swarm** when: you have a small number of agents that each genuinely own a different phase, you care about per-request cost, or the workflow is more like a relay than a fan-out.

## Try it yourself

- Add a third agent `escalation_manager` that the refunder hands to when the requested amount is above $100.
- Notice how each agent's system prompt has to *describe its own handoff conditions*. The lack of a central router is what swarm trades for simplicity.

## Next →

[Lesson 18 · Long-term memory](../18_long_term_memory/README.md)
