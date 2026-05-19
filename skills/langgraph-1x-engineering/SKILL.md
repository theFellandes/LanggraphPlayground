---
name: langgraph-1x-engineering
description: LangGraph 1.x (2026) production patterns — StateGraph with TypedDict/MessagesState, conditional edges with bounded cycles, checkpointers (MemorySaver/SqliteSaver/AsyncPostgresSaver), interrupt()+Command(resume=...) HITL, stream modes (values/updates/messages/custom) and astream_events v2, subgraph composition with reducers, langgraph-supervisor for one-boss-many-workers, langgraph-swarm for peer handoffs, the Store API for long-term cross-thread memory, and langgraph.json for LangGraph Studio. Use when building, debugging, or scaling LangGraph agents in 2026. Reflects the LanggraphPlayground reference project.
---

# LangGraph 1.x Engineering

Production patterns for **LangGraph 1.2+** (May 2026 stack).
Imports below are the ones that actually work — no legacy
0.2-era guesses.

## Canonical imports (memorise these)

```python
# Graph primitives
from langgraph.graph import StateGraph, START, END, MessagesState

# Checkpointers
from langgraph.checkpoint.memory   import MemorySaver
from langgraph.checkpoint.sqlite   import SqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver   # requires the api extra

# HITL & control flow
from langgraph.types import interrupt, Command

# Streaming helper (call inside a node to emit `custom` events)
from langgraph.config import get_stream_writer

# Long-term memory store
from langgraph.store.memory import InMemoryStore

# Prebuilts
from langgraph.prebuilt import ToolNode, tools_condition

# Multi-agent (separate packages)
from langgraph_supervisor import create_supervisor
from langgraph_swarm     import create_swarm, create_handoff_tool

# Agent loop (this is langchain v1, not langgraph, but you use them together)
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware, ModelRequest
```

## The mental model

A LangGraph app is **state + nodes + edges + checkpointer**.

- **State** = a typed dict (`TypedDict`, `MessagesState`, or Pydantic `BaseModel`). The dict that flows between nodes.
- **Node** = `(state) -> partial_state`. Returns only the keys it wrote. Never mutate `state` in place.
- **Edge** = directed transition. Static: `g.add_edge("a", "b")`. Conditional: `g.add_conditional_edges("a", router_fn, {"x": "b", "y": END})`.
- **Compile** = `g.compile(checkpointer=...)`. Returns a Runnable with `.invoke`, `.stream`, `.ainvoke`, `.astream`.

Rule of thumb: **start with LCEL.** Reach for LangGraph the moment
you need loops, branching, persistent state, or HITL.

---

## Pattern · The smallest useful StateGraph

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    draft: str
    critique: str

def draft_node(state: State) -> dict:
    return {"draft": get_llm().invoke(f"Write about {state['topic']}.").content}

def critique_node(state: State) -> dict:
    return {"critique": get_llm().invoke(f"Critique:\n{state['draft']}").content}

g = StateGraph(State)
g.add_node("draft",    draft_node)
g.add_node("critique", critique_node)
g.add_edge(START, "draft")
g.add_edge("draft", "critique")
g.add_edge("critique", END)
graph = g.compile()
```

The node returns a **partial dict** — LangGraph merges it into the
running state.

---

## Pattern · Reducers (`Annotated[..., add]`) for accumulation

Default merge replaces a key. To **accumulate**, declare a reducer:

```python
from typing import Annotated, TypedDict
from operator import add

class State(TypedDict):
    title: str
    sections: Annotated[list[str], add]    # nodes that return {"sections": [x]} append

class State2(MessagesState):                # already has messages: Annotated[list, add_messages]
    extra: int
```

`MessagesState` is the prebuilt for chat — `messages` is already
declared with the `add_messages` reducer which knows how to merge
message lists semantically (de-dup by id, etc.).

---

## Pattern · Conditional edges with bounded cycles

```python
from typing import Literal

MAX_REVISIONS = 3

def route(state) -> Literal["revise", "done"]:
    if state["score"] >= 8 or state["revisions"] >= MAX_REVISIONS:
        return "done"
    return "revise"

g.add_conditional_edges("score", route, {"revise": "draft", "done": END})
```

**Always cap cycles** with either a counter in state or
`graph.invoke(..., config={"recursion_limit": 25})`. Otherwise a
runaway loop is one bad model call away.

---

## Pattern · The agent loop, by hand vs `create_agent`

By hand (lesson 05 / lesson 08 + 09 territory) — useful to
understand, sometimes useful in production when you need full
control:

```python
def call_model(state):
    return {"messages": [llm.bind_tools(TOOLS).invoke(state["messages"])]}

def should_continue(state) -> Literal["tools", "end"]:
    return "tools" if state["messages"][-1].tool_calls else "end"

g = StateGraph(MessagesState)
g.add_node("agent", call_model)
g.add_node("tools", ToolNode(TOOLS))
g.add_edge(START, "agent")
g.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
g.add_edge("tools", "agent")
```

With `create_agent` (default — LangChain v1):

```python
from langchain.agents import create_agent

agent = create_agent(
    model=get_llm(),
    tools=TOOLS,
    system_prompt="You are a careful assistant.",
)
```

The `create_agent` agent **is** a compiled LangGraph graph
internally. Inspect with `agent.get_graph().draw_ascii()`.

---

## Pattern · Persistence (checkpointers)

Same graph definition, three different checkpointers depending on
environment:

```python
graph = g.compile(checkpointer=MemorySaver())                # tests / dev
graph = g.compile(checkpointer=SqliteSaver.from_conn_string("state.db").__enter__())  # local
async with AsyncPostgresSaver.from_conn_string(POSTGRES_URL) as cp:                   # prod
    await cp.setup()
    graph = g.compile(checkpointer=cp)
```

`thread_id` scopes a conversation:

```python
cfg = {"configurable": {"thread_id": "alice"}}
graph.invoke({"messages": [("user", "Hi, I'm Alice")]}, cfg)
graph.invoke({"messages": [("user", "What's my name?")]}, cfg)  # remembers
```

Time-travel: `graph.get_state_history(cfg)` gives every checkpoint
newest-first. Re-invoke with one of those `.config`s to branch.

---

## Pattern · Human-in-the-loop with `interrupt`

```python
from langgraph.types import interrupt, Command

def approval_node(state):
    decision = interrupt({"draft": state["draft"], "ask": "approve / reject / edit?"})
    return {"decision": str(decision)}

# Caller side — first invoke pauses, returns interrupt payload
result = graph.invoke({"draft": "..."}, cfg)
pending = result["__interrupt__"][0].value
# ... show pending to the human ...
final = graph.invoke(Command(resume="approve"), cfg)        # resumes
```

**Checkpointer is required** for `interrupt` — the pause has to be
persisted somewhere.

For agents built with `create_agent`, prefer the prebuilt:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model=get_llm(), tools=[refund, lookup],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"refund": True})],
)
```

The middleware pauses before any `refund` tool call. Use
`interrupt()` directly when you've built a hand-rolled graph;
prefer the middleware when you're on `create_agent`.

---

## Pattern · Streaming — pick the right mode

| `stream_mode=` | Chunk shape | Use for |
|---|---|---|
| `"values"`   | full state after each step          | demos, dashboards |
| `"updates"`  | `{node: partial}`                   | SSE backends, low-bandwidth UIs |
| `"messages"` | `(AIMessageChunk, meta)`            | chat UIs, token-by-token |
| `"custom"`   | whatever you `writer(...)` emit     | progress bars, per-node telemetry |
| `astream_events("v2")` | every Runnable lifecycle | deep debugging, custom UIs |

```python
from langgraph.config import get_stream_writer

def node(state):
    writer = get_stream_writer()
    writer({"progress": "indexing..."})         # appears in stream_mode="custom"
    return {"out": do_work(state)}

for chunk in graph.stream({...}, stream_mode="updates"):
    print(chunk)
```

You can combine: `stream_mode=["updates", "messages"]` — chunks
come back tagged with their mode.

---

## Pattern · Subgraphs (composition)

Compose a graph from another compiled graph:

```python
def write_section(topic: str):
    sub = build_section_graph()    # returns compiled StateGraph
    def _node(state):
        out = sub.invoke({"topic": topic})
        return {"sections": [out["polished"]]}   # parent has Annotated[list, add]
    return _node

g.add_node("intro_section", write_section("history"))
g.add_node("body_section",  write_section("chemistry"))
```

Subgraph state shape is independent of the parent's. The wrapper
node translates between them.

---

## Pattern · Supervisor (one boss, many workers)

```python
from langgraph_supervisor import create_supervisor

math_agent = create_agent(model=get_llm(), tools=[add, multiply],
                          system_prompt="Calculate carefully.", name="math_expert")

travel_agent = create_agent(model=get_llm(), tools=[get_weather],
                            system_prompt="Travel facts.", name="travel_expert")

supervisor = create_supervisor(
    agents=[math_agent, travel_agent],
    model=get_llm(),
    prompt="Delegate math to math_expert, travel to travel_expert."
).compile()
```

The supervisor adds **one extra LLM hop per delegation**. The
benefit is one place to change routing logic. Most-deployed
multi-agent shape in production.

---

## Pattern · Swarm (peer handoffs)

```python
from langgraph_swarm import create_swarm, create_handoff_tool

hand_to_refunder = create_handoff_tool(agent_name="refunder",
                                       description="Transfer to refunder.")
hand_to_triage   = create_handoff_tool(agent_name="triage",
                                       description="Hand back to triage.")

triage   = create_agent(model=get_llm(), tools=[hand_to_refunder], name="triage")
refunder = create_agent(model=get_llm(), tools=[process_refund, hand_to_triage], name="refunder")

swarm = create_swarm(agents=[triage, refunder], default_active_agent="triage").compile()
```

No central router → zero supervisor hops → lower cost. Each agent
has to know its own handoff conditions, so routing logic is
distributed. Use when 2–3 agents own distinct phases.

---

## Pattern · Long-term memory with `Store`

Checkpointer = per-thread. Store = cross-thread, per-user.

```python
from langgraph.store.memory import InMemoryStore

def remember(state, *, store):                 # store injected by name
    ns = ("memories", state["user_id"])
    store.put(ns, str(uuid4()), {"fact": extract_fact(state["msg"])})
    return {}

def reply(state, *, store):
    ns = ("memories", state["user_id"])
    facts = [item.value["fact"] for item in store.search(ns)]
    return {"reply": llm.invoke(f"Known: {facts}\nMsg: {state['msg']}").content}

graph = g.compile(checkpointer=MemorySaver(), store=InMemoryStore())
```

For semantic search across memories, configure the store with an
embedder: `InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})`.

---

## Pattern · LangGraph Studio / Platform integration

Expose your graph via `langgraph.json` at the project root (or a
subproject):

```json
{
  "dependencies": ["."],
  "graphs": {
    "qa": "projects.rag_qa_api.graph:build_graph_definition"
  },
  "env": "../../.env"
}
```

Then `langgraph dev` (from `langgraph-cli`) launches the Studio
inspector — step through runs node-by-node, inspect state at
every checkpoint.

---

## Testing a graph

Use a fake LLM and patch the get_llm import:

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from unittest.mock import patch
import importlib

lesson = importlib.import_module("lessons.08_langgraph_basics.example")

def test_runs_both_nodes_in_order():
    fake = FakeListChatModel(responses=["intro", "- a\n- b"])
    with patch.object(lesson, "get_llm", return_value=fake):
        out = lesson.build_graph().invoke({"topic": "x"})
    assert out["draft"] == "intro"
    assert out["critique"].startswith("- a")
```

Test the **shape** with `graph.get_graph().nodes` — confirms the
topology is what you intended.

---

## Debugging recipe

1. `graph.get_graph().draw_ascii()` — visualise the graph.
2. `PYTHONBREAKPOINT=ipdb.set_trace` then `breakpoint()` inside any node.
3. `for chunk in graph.stream(..., stream_mode="updates"): print(chunk)` — step-by-step trace.
4. `graph.get_state(cfg)` after `invoke` — see the final state.
5. `graph.get_state_history(cfg)` — every checkpoint, newest first.
6. Set `LANGSMITH_TRACING=true` for production-grade traces in the LangSmith UI.

---

## Anti-patterns

- **Module-level LLM construction.** `llm = ChatAnthropic(...)` at top of file breaks tests and breaks `LLM_PROVIDER` switching. Build it inside the node.
- **Mutating state in nodes.** Always return a partial dict.
- **Forgetting reducers.** Two nodes both writing `messages` will fight unless `messages` has `add_messages`.
- **Skipping the checkpointer for HITL.** `interrupt()` silently no-ops without one.
- **Confusing `interrupt()` with raising an exception.** It's a yield point, not an error.
- **Using `MemorySaver` in production.** It evaporates on process exit. Use `Sqlite`/`Postgres`.

---

## Pattern · Guardrails — middleware + decorator + judge node

Production agents need **defence in depth** across four axes:
**input** (PII, prompt injection), **output** (schema, hallucination,
safety), **tool** (allowlist, arg validation), **conversation**
(cost cap, message cap, latency SLA). Three implementation patterns
cover all four:

```python
from functools import wraps
from langchain.agents.middleware import (
    AgentMiddleware, ModelRequest, PIIMiddleware,
    ModelCallLimitMiddleware, HumanInTheLoopMiddleware,
)

# Pattern A · Middleware — declarative, reusable, composable
class CostCap(AgentMiddleware):
    def __init__(self, max_calls=6):
        self.max_calls = max_calls; self._n = 0
    def wrap_model_call(self, request: ModelRequest, handler):
        self._n += 1
        if self._n > self.max_calls:
            raise RuntimeError(f"budget {self.max_calls} exceeded")
        return handler(request)

# Pattern B · Decorator — wrap one tool with a runtime rule
def validate_tool_args(check):
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            check(**kw)
            return fn(*a, **kw)
        return wrapper
    return deco

@tool
@validate_tool_args(lambda amount, **_: None if amount <= 200
                                         else (_ for _ in ()).throw(
                                            ValueError("REFUND_BLOCKED")))
def process_refund(order_id: str, amount: float): ...

# Pattern C · Layer-as-node — schema enforcement on the final reply
class FinalAnswer(BaseModel):
    summary:      str  = Field(min_length=4, max_length=400)
    contains_pii: bool

def judge(raw: str) -> FinalAnswer:
    return get_llm().with_structured_output(FinalAnswer).invoke(
        f"Rewrite into schema. Strip PII.\n\n{raw}")

# Compose them
agent = create_agent(
    model=..., tools=[process_refund, lookup],
    middleware=[
        PIIMiddleware(pii_type="email",       strategy="redact"),
        PIIMiddleware(pii_type="credit_card", strategy="block"),
        CostCap(max_calls=6),
        # HumanInTheLoopMiddleware(interrupt_on={"process_refund": True}),
    ],
)
```

**Recoverable vs unrecoverable.** Raising inside a `@tool` becomes
a `ToolMessage(status="error")` the model can read — perfect for
policy violations the model can route around (refund cap). Use real
exceptions only for *unrecoverable* faults (downstream API down).

**Defence in depth, in order:**

```
cheap input filter   →   wrap_model_call (cost cap / cache / fallback)
                      →  tool decorators (arg validation)
                      →  schema/judge on the final reply
```

**Industry shortcut:**

| Tool | When |
|---|---|
| LangChain middleware | default — covers 80% of cases, no extra dep |
| **NeMo Guardrails** (Colang) | topical / dialogue-shape rails ("don't discuss competitors") |
| **Guardrails AI** (`guardrails-ai`) | validator-registry + retry-on-fail for structured output |
| **Llama Guard / ShieldGemma** | dedicated safety classifier — wrap its call as a middleware |
| **LLM-as-judge** | cheap, surprisingly effective output review |

See `lessons/19_guardrails/` in the LanggraphPlayground reference
project for a worked example composing all three patterns on a
single agent.

---

## Choosing a multi-agent shape

| Situation | Pick |
|---|---|
| 1 agent does the job with tools | `create_agent` alone |
| 1 agent with cross-cutting policy (summarisation, retries, HITL, PII) | `create_agent` + middleware |
| 3+ specialists routed by intent | **Supervisor** |
| 2–3 agents owning distinct phases of one workflow | **Swarm** |
| Heavy custom control flow / multiple non-agent nodes | Hand-built `StateGraph` |
