# Capstone · `research_assistant_pro`

The Tier 6 version of `research_assistant`. Same overall shape
(supervisor + workers) but **production-shaped**:

| Layer | What's added |
|---|---|
| Graph | Parallel fan-out to N sub-researchers, reducer merges, bounded critic cycle |
| Concurrency | `asyncio.Semaphore` rate-limits the LLM + Tavily; per-topic locks prevent duplicate work |
| Prompts | Jinja2 templates with `{% extends %}`; locale-aware; version-pinned |
| Reliability | `tenacity` retries on the search tool; circuit breaker around Tavily |
| Observability | Every node emits `prompt_version`, `prompt_sha`, `attempt` for tracing |

```
                          ┌──── researcher #1 ────┐
            ┌── plan ──┬─→├──── researcher #2 ────┤── reduce ──┐
   user ───┤           │  ├──── researcher #3 ────┤             ├─ writer ─→ critic ─┐
            └── plan ──┘  └──── researcher #N ────┘             │                    │
                              (semaphore = 5)                    └── revise loop ────┘
                                                                      (max 3 cycles)
```

## What it teaches (concept → lesson)

| Element | Lesson |
|---|---|
| Supervisor + workers | 16 |
| Parallel fan-out with `Send` + reducer | 30 (pattern 1) |
| `asyncio.Semaphore` rate-limit | 27 (pattern 2) |
| Per-topic lock map | 27 (pattern 3) |
| Jinja-templated agent prompts | 28 |
| Versioned prompt registry | 32 |
| Tenacity retries | 30 (pattern 5) |
| Circuit breaker | 30 (pattern 6) |
| Bounded critic cycle | 30 (pattern 4) |

## Prerequisites

```bash
uv sync                              # picks up jinja2, tenacity
# Optional: Tavily for real web search. Without it, a stub search is used.
echo "TAVILY_API_KEY=tvly-..." >> .env
```

## Run it

```bash
uv run python -m projects.research_assistant_pro.graph "What are the latest advances in fusion energy?"

# Or with explicit topic subdivision:
uv run python -m projects.research_assistant_pro.graph \
  "Compare fusion approaches: tokamak vs stellarator vs inertial"
```

The script:

1. **Planner** decomposes the user's question into 3-5 subtopics.
2. **Fan-out** spawns one researcher per subtopic in parallel.
3. **Researchers** run under a semaphore (max 5 concurrent) — protects against rate limits.
4. **Reducer** merges all per-subtopic claims into a single working set.
5. **Writer** drafts the report using the merged claims.
6. **Critic** approves or sends feedback. Loop capped at 3 revisions.

## Concurrency walkthrough

```python
# Module-level state — never inside graph state.
_topic_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_llm_sem = asyncio.Semaphore(5)
_tavily_breaker = CircuitBreaker(threshold=3, cooldown=30)
```

Three coordination primitives:

1. **`_topic_locks[subtopic]`** — if the same subtopic appears in two queries running back-to-back, the second waits. Prevents duplicate work.
2. **`_llm_sem`** — global cap on how many LLM calls are in flight. The supervisor + 5 researchers + writer + critic could otherwise blast 8 concurrent calls.
3. **`_tavily_breaker`** — if Tavily fails 3 times within 30s, the searcher node short-circuits with a "search unavailable" message instead of timing out.

The capstone's `graph.py` shows where each guard lives in the node body.

## Prompt walkthrough

`prompts/agents/researcher.j2` extends `prompts/_base/persona.j2`. The
shared base contains the company voice, safety rules, and an inheritance
block. Each agent overrides only the role and extras.

The system prompt is **a callable** so it re-renders every turn:

```python
def system_for_researcher(state, runtime):
    return registry.render(
        "agents/researcher",
        version="v2",
        agent_name="Researcher",
        company="Acme",
        tools=state["tools"],
        locale=runtime.context.get("locale", "en-US"),
        vip=runtime.context.get("vip", False),
        subtopic=state.get("subtopic"),
    )
```

Swap `version="v2"` → `"v3"` for a one-line A/B; pair with the
`assign_variant` helper from lesson 32 for sticky-per-user routing.

## Try it yourself

1. **Add a `fact_checker` worker** between writer and critic. Wire it into the fan-out so it reads `state["summaries"]` and runs Tavily on each claim.
2. **Swap the LLM provider mid-graph.** Use cheap Haiku for researchers, premium Sonnet for the writer (cost-routing pattern from lesson 26).
3. **Persist runs.** Add `MemorySaver` so you can `get_state_history(cfg)` and replay a failed run from the step before the failure.
4. **Distributed lock.** Promote `_topic_locks` to Redis (lesson 31) so two API replicas don't both research "fusion" when invoked simultaneously.

## Files

```
research_assistant_pro/
├── README.md           ← you are here
├── graph.py            ← the multi-agent graph
├── prompts/
│   ├── _base/
│   │   ├── persona.j2
│   │   └── safety_rules.j2
│   └── agents/
│       ├── planner.j2
│       ├── researcher.j2
│       ├── writer.j2
│       └── critic.j2
└── concurrency.py      ← semaphores, breakers, lock map
```

## Pairs with

- [Lesson 30](../../lessons/30_advanced_graphs/README.md), [Lesson 28](../../lessons/28_dynamic_prompting/README.md), [Lesson 27](../../lessons/27_locks_and_concurrency/README.md), [Lesson 16](../../lessons/16_supervisor/README.md), [Lesson 32](../../lessons/32_prompt_engineering_lab/README.md).
