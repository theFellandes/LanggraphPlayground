# Lesson 30 · Advanced graph patterns

> Tier 6 starts here. The graphs in lessons 08-17 are linear, branching,
> or single-loop. This lesson covers the *runtime-dynamic* patterns
> production agents need.

## What you'll learn

| Pattern | LangGraph primitive | Real-world use |
|---|---|---|
| **Parallel fan-out** | `Send(node, payload)` | Run 5 research agents on 5 subtopics in parallel |
| **Map-reduce** | Fan-out + reducer in state | Summarise 50 PDFs → merge summaries |
| **Dynamic subgraph spawning** | `Send` whose count depends on state | One subgraph per chunk / per file / per tool result |
| **Cycle with budget** | conditional edge + counter in state | Self-correction loop with a hard cap |
| **Retry with backoff** | wrapper node + `tenacity` | Survives transient API failures |
| **Circuit breaker** | shared counter + skip branch | When upstream is down, fail fast instead of timing out |
| **Streaming join** | merge multiple streams into one event log | UI for parallel agents |

Each gets a small runnable demo in `example.py`. No API key required —
the demos use a stub LLM with simulated latency and failures.

## Pattern 1 · Parallel fan-out with `Send`

`Send(node_name, payload)` is LangGraph's primitive for spawning N
parallel branches from one upstream node. The conditional edge
returns a **list** of `Send` objects; the runtime executes them
concurrently.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import TypedDict, Annotated
from operator import add

class State(TypedDict):
    topic: str
    subtopics: list[str]
    summaries: Annotated[list[str], add]    # ← reducer concatenates
    final: str

def plan(state):
    return {"subtopics": ["history", "tech", "economics", "policy"]}

def fan_out(state):
    # Return a list of Send. Each is independent.
    return [Send("research", {"subtopic": s}) for s in state["subtopics"]]

def research(state):
    # Worker. Receives one subtopic.
    return {"summaries": [f"Summary of {state['subtopic']}"]}

def reduce(state):
    return {"final": "\n\n".join(state["summaries"])}

g = StateGraph(State)
g.add_node("plan", plan)
g.add_node("research", research)
g.add_node("reduce", reduce)
g.add_edge(START, "plan")
g.add_conditional_edges("plan", fan_out, ["research"])
g.add_edge("research", "reduce")
g.add_edge("reduce", END)
```

**The reducer is the trick.** `Annotated[list[str], add]` tells the
runtime: "when multiple branches write to `summaries`, append them all
instead of clobbering." Without the reducer, only one branch's value
survives.

Same shape works for **map-reduce over documents**: fan out one branch
per chunk, accumulate summaries in a reduced list, then a final node
merges. This is how lesson 26's eval framework would scale to running
1000 cases in parallel.

## Pattern 2 · Bounded concurrency (combine with lesson 27)

Fan-out spawns *all* branches at once. If you have 100 subtopics and
your LLM provider tops out at 50 RPS, you'll get 50 429s. Combine fan-out
with `asyncio.Semaphore`:

```python
_sem = asyncio.Semaphore(8)

async def research_async(state):
    async with _sem:
        return {"summaries": [await llm.ainvoke(state["subtopic"])]}
```

LangGraph runs nodes inside an asyncio task — the semaphore caps the
real concurrency even though logically 100 branches "started." This is
the canonical way to mix Pattern 1 (fan-out) with rate-limited APIs.

## Pattern 3 · Dynamic subgraph spawning

The Send list doesn't have to be static. Spawn one branch per file in
a directory, one per tool result, one per matched record — anything
where the count is data-dependent.

```python
def fan_out_per_file(state):
    files = list(Path(state["upload_dir"]).glob("*.pdf"))
    return [Send("parse_file", {"path": str(p)}) for p in files]
```

This is the **shape of a document-processing pipeline**: dispatch one
worker per file, reduce when all are done. The pipeline self-sizes.

## Pattern 4 · Cycle with a budget

LangGraph supports loops, but unbounded loops are a tar pit. The
correct shape:

```python
class CritiqueState(TypedDict):
    messages: list
    draft: str
    critique: str
    revisions: int                # bumped each loop

def should_revise(state) -> str:
    if state["revisions"] >= 3:        # hard cap
        return END
    if "APPROVED" in state["critique"]:
        return END
    return "revise"

g.add_conditional_edges("critic", should_revise, {"revise": "writer", END: END})
```

Three is the magic number — empirically, more revisions ≠ better
output, and you've burned tokens. Lesson 26 Topic 5 has the same cap
for self-correction.

## Pattern 5 · Retry with backoff

For idempotent calls, wrap the node with `tenacity`:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def llm_node(state):
    return {"messages": [await llm.ainvoke(state["messages"])]}
```

Three attempts, exponential backoff (1s, 2s, 4s). For *non-idempotent*
calls (charging a card), you must combine this with an idempotency key
(lesson 27).

## Pattern 6 · Circuit breaker

When an upstream service is hard-down, retries pile up and your whole
graph hangs. The circuit breaker pattern: count recent failures, if
they exceed a threshold, *skip the call entirely* for a cooldown
window.

```python
class CircuitBreaker:
    def __init__(self, threshold=5, cooldown=30):
        self.fails = 0
        self.opened_at = None
        self.threshold = threshold
        self.cooldown = cooldown

    def is_open(self):
        if self.opened_at is None:
            return False
        if time.time() - self.opened_at > self.cooldown:
            self.opened_at = None
            self.fails = 0
            return False
        return True

    def record(self, ok: bool):
        if ok:
            self.fails = 0
        else:
            self.fails += 1
            if self.fails >= self.threshold:
                self.opened_at = time.time()

breaker = CircuitBreaker()

async def search_node(state):
    if breaker.is_open():
        return {"context": "(search unavailable; circuit open)"}
    try:
        docs = await search.ainvoke(state["query"])
        breaker.record(True)
        return {"context": docs}
    except Exception:
        breaker.record(False)
        return {"context": "(search failed)"}
```

Production tip: ship `tenacity` + `pybreaker` instead of rolling your
own — they handle the half-open state, jitter, and observability.

## Pattern 7 · Streaming with parallel branches

When you fan out N branches, the UI wants to see *all* of them stream
at once. LangGraph's `stream_mode="updates"` emits per-node updates;
you can tag them with the branch id:

```python
async for chunk in graph.astream(inp, stream_mode="updates"):
    for node_name, update in chunk.items():
        # node_name e.g. "research:0", "research:1", ...
        emit_to_client(node=node_name, payload=update)
```

The client multiplexes the streams into N panels. This is how
multi-agent dashboards work.

## When to reach for which pattern

| Symptom | Pattern |
|---|---|
| "I want to research 5 things at once" | 1 (fan-out) + 2 (semaphore) |
| "Process every PDF in this folder" | 3 (dynamic fan-out) |
| "Self-correct until the critic approves, but cap at 3" | 4 (bounded cycle) |
| "Anthropic timed out, let me retry" | 5 (retry/backoff) |
| "Tavily is hard-down, don't wait" | 6 (circuit breaker) |
| "Show me what each agent is doing in real time" | 7 (streaming) |

## Run it

```bash
uv run python -m lessons.30_advanced_graphs.example
uv run python -m lessons.30_advanced_graphs.example --fan-out
uv run python -m lessons.30_advanced_graphs.example --map-reduce
uv run python -m lessons.30_advanced_graphs.example --bounded-cycle
uv run python -m lessons.30_advanced_graphs.example --retry
uv run python -m lessons.30_advanced_graphs.example --breaker
```

All demos are offline. The "LLM" is a coroutine that sleeps and
occasionally throws — enough to show retries and circuit-breaker
behaviour.

## Anti-patterns

| Smell | Fix |
|---|---|
| Forgetting the reducer on fan-out state | Only the last branch's value persists. `Annotated[list, add]` fixes it |
| Unbounded retry loops | Always `stop_after_attempt(N)`. 3 is the magic number |
| Circuit breaker with no cooldown | Half-open state is mandatory; otherwise you're locked open forever |
| Fan-out with 100 branches, no semaphore | You just DOS'd your LLM provider; combine with lesson 27 pattern 2 |
| `Send` with mutable shared dict as payload | Each branch gets a reference, race city. Copy the payload |
| Returning `Send` in the *node*, not the conditional edge | Send only works from conditional edges. Read the docs twice |

## Pairs with

- **[Lesson 09 · Conditional edges](../09_conditional_edges/README.md)** — the primitive
- **[Lesson 15 · Subgraphs](../15_subgraphs/README.md)** — embedded graphs; Send works inside subgraphs too
- **[Lesson 27 · Locks](../27_locks_and_concurrency/README.md)** — semaphore + retry combo
- **[Lesson 31 · Distributed locks](../31_distributed_locks/README.md)** — distributed circuit breaker (Redis-backed)

## References

- [LangGraph `Send` API](https://docs.langchain.com/oss/python/langgraph/multiple-agents) — official
- [LangGraph map-reduce tutorial](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/) — the canonical example
- [`tenacity`](https://tenacity.readthedocs.io/) — retry primitives
- [`pybreaker`](https://github.com/danielfm/pybreaker) — production-grade circuit breaker
- [Martin Fowler · Circuit Breaker](https://martinfowler.com/bliki/CircuitBreaker.html) — the design pattern

## Next →

[Lesson 31 · Distributed locks](../31_distributed_locks/README.md) — multi-process coordination.
