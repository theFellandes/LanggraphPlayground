# Lesson 22 · LLM Application Architecture

**Purely architectural.** No code in this lesson. Diagrams, decision
trees, anti-patterns, and scaling tiers. This is the meta-lesson that
sits on top of lessons 00-21 and tells you how to *compose* them into
a reliable production application.

> *Most LLM-app outages aren't model failures. They're architectural
> failures.* The model returned something plausible, your system
> believed it, and there was no second layer to catch the mistake.
> This lesson is the mental map for designing the second layer.

## What you'll learn

- The **8 architectural layers** of a production LLM app, and what each owns
- The **four scaling tiers** (prototype → pilot → production → multi-tenant) and what each requires
- A decision matrix per architectural concern (provider, reliability, validation, guardrails, orchestration, state, observability, interface)
- The **anti-patterns** that show up in nearly every failed production system
- How the patterns in lessons 00-21 map onto these layers
- When to break the rules

## Why "architectural" matters

If you've built lessons 00-21, you have ~20 primitives:
`get_llm()`, `with_structured_output()`, `create_agent`,
middleware, checkpointers, supervisors, swarms, sanitizers,
guardrails. The hard problem isn't using any one of them. The hard
problem is **knowing which to compose, in what order, at what scale**.

Architecture answers that. And the architecture for an LLM app in
2026 looks very different from the web app you built in 2015 —
non-deterministic outputs, expensive calls, model drift, prompt
injection. New constraints, new layering.

---

## The 8 architectural layers

The layering is roughly bottom-up by *dependency* — each layer
depends only on the ones below it. Top-down by *request flow* — a
user request enters at the top, walks down, then back up with the
response.

```
   ╔══════════════════════════════════════════════════════════════╗
   ║  8 ·  INTERFACE LAYER                                        ║
   ║       FastAPI / CLI / WebSocket / Slack bot / Studio         ║
   ║       sync · async · streaming · batch · scheduled           ║
   ╠══════════════════════════════════════════════════════════════╣
   ║  7 ·  OBSERVABILITY LAYER                                    ║
   ║       LangSmith · OpenTelemetry · structured logs · evals    ║
   ║       traces · token counts · latency · error rates          ║
   ╠══════════════════════════════════════════════════════════════╣
   ║  6 ·  STATE LAYER                                            ║
   ║       checkpointers (Memory / Sqlite / Postgres) · Store     ║
   ║       thread_id scoping · time-travel · long-term memory     ║
   ╠══════════════════════════════════════════════════════════════╣
   ║  5 ·  ORCHESTRATION LAYER                                    ║
   ║       LCEL chains · StateGraph · create_agent · supervisor   ║
   ║       conditional edges · subgraphs · swarm                  ║
   ╠══════════════════════════════════════════════════════════════╣
   ║  4 ·  GUARDRAIL LAYER                                        ║
   ║       PIIMiddleware · ModelCallLimit · HumanInTheLoop        ║
   ║       custom middleware · @validate_tool_args · judge node   ║
   ╠══════════════════════════════════════════════════════════════╣
   ║  3 ·  VALIDATION LAYER                                       ║
   ║       Pydantic schemas · field_validator · schema_sanitizer  ║
   ║       deterministic re-parsers (dateparser / dateutil)       ║
   ╠══════════════════════════════════════════════════════════════╣
   ║  2 ·  RELIABILITY LAYER                                      ║
   ║       fallback chain · retry · circuit breaker               ║
   ║       cost cap · timeout · idempotency                       ║
   ╠══════════════════════════════════════════════════════════════╣
   ║  1 ·  PROVIDER LAYER                                         ║
   ║       ChatAnthropic · ChatOpenAI · ChatGoogleGenAI · ...     ║
   ║       adapter pattern · settings.llm_provider                ║
   ╚══════════════════════════════════════════════════════════════╝
```

Each layer has **one responsibility**, **one place** to look for bugs,
and **one set of tools** to swap if you need to change behaviour.

---

## Layer-by-layer: what each owns

### 1 · Provider Layer

**Owns:** which provider's SDK is talking to which model.

**Knows nothing about:** retries, validation, business logic.

**In this repo:** [`shared/llm/`](../../shared/llm/) — `get_llm()` factory + per-provider adapter files (`anthropic_adapter.py`, `openai_adapter.py`). Settings-driven via `LLM_PROVIDER`.

**Canonical API shape:**

```python
# shared/llm/base.py
def get_llm(provider: str | None = None, model: str | None = None, **kw) -> Runnable:
    """Returns a configured chat model. Provider chosen by env var or override."""

# shared/llm/openai_adapter.py
def build(model: str | None = None, **kw) -> BaseChatModel:
    return ChatOpenAI(model=model or "gpt-4.1", api_key=settings.openai_api_key, **kw)
```

**Decision matrix:**

| You need | Use |
|---|---|
| One provider, simplest case | Direct `ChatAnthropic(...)` or `ChatOpenAI(...)` |
| Two-or-more providers, swappable | **Adapter pattern** — one file per provider, factory dispatches |
| Per-call provider choice (e.g. tier by cost) | Adapter + a router (cheap model for triage, expensive for hard cases) |

**Anti-pattern:** instantiating `ChatAnthropic(...)` at module top level. Breaks tests, breaks provider switching, breaks fallback. **Build inside functions.**

### 2 · Reliability Layer

**Owns:** what happens when a provider call fails (quota, timeout, 5xx, transient network).

**Knows nothing about:** what the response means, business logic.

**In this repo:** [`shared/llm/base.py`](../../shared/llm/base.py) — `get_llm()` returns a `RunnableWithFallbacks` by default. On any error, the next configured provider takes over.

**Canonical API shape:**

```python
# Built on top of the provider layer; consumers don't see the chain.
primary = anthropic_adapter.build(...)
backup  = openai_adapter.build(...)
llm: Runnable = primary.with_fallbacks([backup])     # ← Layer 2
llm.invoke("...")                                     # tries primary, falls back on error
```

**The reliability primitives:**

| Primitive | What it does | When |
|---|---|---|
| **Fallback chain** | Try provider A, if error try B, ... | Always. Even with one provider — fallback to a smaller model of the same provider |
| **Retry with backoff** | Retry the same call N times with exponential delay | Transient network / 5xx errors. Don't retry quota / auth errors. |
| **Circuit breaker** | After N failures in T window, stop calling and surface a clean error | Public-facing endpoints; protects downstream from cascading failure |
| **Timeout** | Per-call wall-clock cap | Always. Default ~30s for chat, ~120s for long-context |
| **Cost cap** | Hard limit on tokens/calls per run/user/day | Public-facing endpoints; bills you can't unsign |
| **Idempotency** | Same request id → same response (cached) | Production write paths that might retry |

**Anti-pattern:** retrying a quota-exceeded error. Quota errors are *permanent until the window resets*. Retrying just burns more quota. Detect them and skip to the next provider.

### 3 · Validation Layer

**Owns:** "the model said something — was it actually correct?"

**Knows nothing about:** how to retry, who's authorised, where to write the result.

**In this repo:** [`shared/llm/schema_sanitizer.py`](../../shared/llm/schema_sanitizer.py), Pydantic schemas in lessons 04 / 19 / 21, `dateparser`-driven validators in lesson 21.

**Canonical API shape:**

```python
class Person(BaseModel):
    birth_date: str | None = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("birth_date")
    @classmethod
    def _parse(cls, v):
        if v is None: return None
        d = dateparser.parse(v, settings={"STRICT_PARSING": True})
        if d is None or d.date() > date.today():
            raise ValueError(f"invalid birth date: {v!r}")
        return d.date().isoformat()

llm = get_llm().with_structured_output(Person)        # Layer 3 wraps Layer 1
```

**The validation primitives:**

| Primitive | What it catches |
|---|---|
| **Pydantic schema with `with_structured_output`** | Wrong JSON shape, wrong types, missing required fields |
| **Schema sanitizer** | Provider-side strict-mode incompatibilities (e.g. `format: "date"`) |
| **`field_validator` (cheap, deterministic)** | Domain rules — birth date in past, age plausible, ISO-format string parseable |
| **`model_validator` (cross-field rules)** | "termination_date must be after hire_date", "DOB + age == today" |
| **Deterministic re-parsers (`dateparser`, `dateutil`)** | Free-text outputs the schema regex passed but aren't actually valid |
| **LLM-as-judge** (lesson 19's judge node) | Whole-conversation-level checks that don't fit a field — "did this answer hallucinate facts?" |

**The defence-in-depth principle:** cheap deterministic checks run on every call; expensive LLM-as-judge checks run only on outputs that will actually be returned to the user.

**Anti-pattern:** "we trust the model's output because GPT-5 is smart enough." It isn't. The clinical literature ([PMC11634005](https://pmc.ncbi.nlm.nih.gov/articles/PMC11634005/)) measured 15-25% wrong dates from GPT-4. **Always validate.**

### 4 · Guardrail Layer

**Owns:** what the model is *allowed to say* and what it's *allowed to do*.

**Knows nothing about:** whether the output is *correct* (that's validation), only whether it's *permissible*.

**In this repo:** [`lessons/19_guardrails/`](../19_guardrails/README.md) — `PIIMiddleware`, custom `AgentMiddleware` for cost caps, `@validate_tool_args` decorators, the judge node.

**Canonical API shape:**

```python
from langchain.agents.middleware import (
    AgentMiddleware, PIIMiddleware, ModelCallLimitMiddleware,
)

class CostCap(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        # ... raise if budget exceeded ...
        return handler(request)

agent = create_agent(
    model=get_llm(), tools=[...],
    middleware=[                                       # ← Layer 4 wraps Layer 5
        PIIMiddleware(pii_type="email", strategy="redact"),
        CostCap(),
        ModelCallLimitMiddleware(max_calls=6),
    ],
)
```

**Four guardrail axes (read [lesson 19](../19_guardrails/README.md) for depth):**

```
            INPUT          OUTPUT
              │              │
              ▼              ▼
        ┌──────────────────────────┐
        │      AGENT LOOP          │
        │   (model + tools + state)│
        │                          │
        │   ┌──────────────────┐   │
        │   │ TOOL EXECUTION   │   │   ← TOOL guardrails
        │   └──────────────────┘   │
        └──────────────────────────┘
                  │
                  ▼
           CONVERSATION-LEVEL
           (cost caps, message caps, summarisation triggers)
```

The four axes — input / output / tool / conversation — should each have
**at least one cheap guardrail and one expensive guardrail**. Cheap (regex /
schema / counter) runs on every call. Expensive (LLM-as-judge / safety
classifier) runs only on returned outputs.

**Anti-pattern:** a single point of guardrail enforcement ("I'll just add
PII redaction in the system prompt"). Guardrails are **defence in depth.**
The prompt-level one is the *outermost* layer — never the only one.

### 5 · Orchestration Layer

**Owns:** the shape of the workflow. "First retrieve, then generate", "If the agent emits a tool call, run it then loop back", "Send to the researcher worker, then the writer worker, then the critic".

**Knows nothing about:** which model is doing the work (that's Provider Layer), how state persists (State Layer), what the user sees (Interface Layer).

**In this repo:** lessons 02 (LCEL), 08-15 (graphs), 16-17 (multi-agent), the capstones.

**Canonical API shape** (the four common shapes side-by-side):

```python
# Shape 1 — one-shot
get_llm().invoke("...")

# Shape 2 — LCEL chain
chain = prompt | get_llm() | StrOutputParser()
chain.invoke({"topic": "..."})

# Shape 3 — agent with tools
agent = create_agent(model=get_llm(), tools=[...])
agent.invoke({"messages": [...]})

# Shape 4 — custom graph (branching / loops / persistent state)
g = StateGraph(MyState)
g.add_node("retrieve", ...); g.add_node("generate", ...)
g.add_edge(START, "retrieve"); g.add_edge("retrieve", "generate")
graph = g.compile(checkpointer=...)
```

**The orchestration shape spectrum:**

```
Simplest                                                         Most complex
───────────────────────────────────────────────────────────────────────────►

  llm.invoke()      Chain (LCEL)    create_agent    Custom StateGraph    Supervisor / Swarm
  (one shot)        prompt|llm|p    (model+tools)   (branches, loops)    (multi-agent)
```

**Rule of thumb:** start at the leftmost shape that *might* work. Promote
only when the simpler shape provably can't do the job. The cost of a wrong
choice rises steeply going right.

**Anti-patterns:**

- Using `StateGraph` when a chain would do — adds checkpointer overhead, debugging surface
- Using `create_agent` when there are no tools — adds an agent loop you don't need
- Using a supervisor when one agent + tools would do — adds an extra LLM hop per delegation
- Using a swarm when supervisor would be clearer to debug — distributed routing logic

### 6 · State Layer

**Owns:** what persists between requests, and how it's scoped.

**Knows nothing about:** which model produced the state, what the user is *allowed* to see.

**In this repo:** lessons 12 (checkpointers), 18 (long-term Store), and the capstones (`SqliteSaver`, `AsyncPostgresSaver`).

**Canonical API shape:**

```python
# Per-conversation: checkpointer + thread_id
from langgraph.checkpoint.sqlite import SqliteSaver
graph = builder.compile(checkpointer=SqliteSaver.from_conn_string("state.db"))
cfg = {"configurable": {"thread_id": "alice"}}
graph.invoke({"messages": [...]}, cfg)                 # persists per thread

# Cross-conversation: Store (per-user / per-org / semantic)
from langgraph.store.memory import InMemoryStore
graph = builder.compile(checkpointer=..., store=InMemoryStore())

def remember(state, *, store):                          # store injected at runtime
    store.put(("memories", state["user_id"]), key, {"fact": ...})
```

**Two distinct scopes:**

```
┌─────────────────────────────────────────────────────────────────┐
│  CHECKPOINTER  — scoped to one conversation (thread_id)         │
│  ──────────────────────────────────────────────────────────    │
│  MemorySaver        — in-process, lost on restart               │
│  SqliteSaver        — file-on-disk, single-process              │
│  PostgresSaver      — production, multi-process, scalable       │
│  AsyncPostgresSaver — production async (FastAPI, etc.)          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  STORE         — scoped to a user, persists ACROSS threads      │
│  ──────────────────────────────────────────────────────────    │
│  InMemoryStore         — dev / tests                            │
│  PostgresStore         — production semantic memory             │
│  (custom)              — backed by any KV / vector store        │
└─────────────────────────────────────────────────────────────────┘
```

**Anti-pattern:** using `MemorySaver` in production. It evaporates on
process restart. Catches everyone exactly once.

### 7 · Observability Layer

**Owns:** *how do I know what happened?*

**Knows nothing about:** retries, validation, business logic. Just *records.*

**In this repo:** lesson 03 (ipdb + `with_listeners`), the `LANGSMITH_*` env vars in [`shared/settings.py`](../../shared/settings.py).

**Canonical API shape:**

```python
# Dev — drop into ipdb when something looks wrong
PYTHONBREAKPOINT = "ipdb.set_trace"
breakpoint()                                          # in any node / chain step

# Non-interactive — attach listeners to log every step
def on_end(run, config=None):
    logger.info("step.end", name=run.name, latency_ms=run.latency_ms,
                tokens=run.tokens, ok=not run.error)
chain.with_listeners(on_end=on_end).invoke(...)

# Prod — enable LangSmith via env, every Runnable becomes a trace
# LANGSMITH_TRACING=true; LANGSMITH_API_KEY=...; LANGSMITH_PROJECT=my-app
```

**The observability stack:**

```
  Development             Staging                Production
─────────────             ───────                ──────────
  print()                 logging                structured logs
  breakpoint() + ipdb     LangSmith local        LangSmith hosted / Langfuse self-hosted
  Runnable listeners      eval suite (cron)      OpenTelemetry traces
  graph.get_state()       full graph replay      pre-/post-deploy eval gates
                                                 alerts on quality drift
```

**Three things every production trace must capture:** the *prompt* sent, the
*raw model output* (before validation), the *final return* (after all
layers). If you only log the final return, you can't debug. If you only log
the prompt, you can't see what the model said. If you only log the model
output, you can't see what was filtered.

**Anti-pattern:** logging only "success" / "failure" booleans. The bug is
always in the *contents* of one of the three.

### 8 · Interface Layer

**Owns:** how the rest of the world talks to the LLM app — **including locale-aware presentation** (rendering an ISO date as *"23 Mayıs 2026"* for a Turkish user; see [Lesson 23](../23_date_localization/README.md)).

**Knows nothing about:** which model, retries, validation, state.

**In this repo:** `projects/rag_qa_api/` (FastAPI + SSE), `projects/customer_support_bot/` (CLI + slash commands), every lesson's `example.py` (script).

**Canonical API shape (FastAPI + SSE — the production-grade default):**

```python
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

app = FastAPI()

@app.post("/chat")                                    # sync request/response
async def chat(req: ChatRequest):
    result = await graph.ainvoke({"messages": [...]}, req.cfg)
    return {"reply": result["messages"][-1].content}

@app.post("/stream")                                  # token-by-token SSE
async def stream(req: ChatRequest):
    async def events():
        async for chunk, _ in graph.astream(..., stream_mode="messages"):
            yield {"event": "token", "data": chunk.content}
        yield {"event": "done", "data": ""}
    return EventSourceResponse(events())
```

**The interface shape determines the architecture above it:**

```
  Sync HTTP request/response   → simplest. /chat returns one JSON
  SSE / WebSocket streaming    → needs stream_mode=messages / astream_events
  CLI                          → input() loop, slash commands for HITL resume
  Background batch             → no interface; reads from a queue / Cron
  Scheduled job                → CronCreate / Airflow / Temporal
  IDE integration              → MCP server (this repo doesn't ship one yet)
```

**Anti-pattern:** mixing two interface shapes in one process (a FastAPI app
that also runs a background scheduler). Split them. The state, error
modes, and lifecycles are different.

---

## The four scaling tiers

What you *need* at each layer depends on scale. Most teams fail because
they over-build at Tier 1 or under-build at Tier 3.

### Tier 1 · Prototype (one developer, one provider, demo-grade)

```
PROVIDER       single ChatAnthropic instance, hardcoded
RELIABILITY    none (it's a demo)
VALIDATION     Pydantic schemas where you remember to
GUARDRAILS     none
ORCHESTRATION  LCEL chain or one StateGraph
STATE          MemorySaver
OBSERVABILITY  print() + LangSmith local
INTERFACE      Jupyter notebook or single Python script
```

**Verdict for Tier 1:** ship it. Don't add the rest yet. You'll
over-engineer for a use case you don't fully understand.

### Tier 2 · Pilot (small team, paying customers, single tenant)

```
PROVIDER       adapter pattern (shared/llm/), settings-driven
RELIABILITY    fallback chain (with_fallbacks)
VALIDATION     Pydantic schemas + field_validators, dateparser pipeline
GUARDRAILS     PIIMiddleware + cost cap middleware
ORCHESTRATION  create_agent + middleware, or hand-built StateGraph
STATE          SqliteSaver or single Postgres
OBSERVABILITY  LangSmith hosted, basic dashboards
INTERFACE      FastAPI with /chat (sync) and /stream (SSE)
```

**Verdict for Tier 2:** the LanggraphPlayground capstones target this tier.
What you *don't* yet need: multi-tenant isolation, autoscaling, complex
caching.

### Tier 3 · Production (multi-team, SLA-bound, growing traffic)

```
PROVIDER       multi-provider with cost-tier routing (cheap → expensive)
RELIABILITY    fallback + retries + circuit breakers + cost caps
VALIDATION     Defence-in-depth — schema + sanitizer + deterministic re-parser + judge node
GUARDRAILS     Full 4-axis coverage incl. Llama Guard / equivalent
ORCHESTRATION  Supervisor for hard delegation, swarm for distinct phases
STATE          Postgres (Async) + Store with semantic search
OBSERVABILITY  LangSmith + OpenTelemetry traces + scheduled eval suite + alerting
INTERFACE      FastAPI cluster behind load balancer; streaming + batch + scheduled
```

**Verdict for Tier 3:** every layer is *real*. The cost of skipping any one
of them is a measurable incident.

### Tier 4 · Multi-tenant / regulated (enterprise, healthcare, finance)

```
PROVIDER       per-tenant model routing; some tenants pinned to specific providers (data residency)
RELIABILITY    everything in Tier 3 + per-tenant rate limits
VALIDATION     Tier 3 + per-tenant business rules (e.g. medical vs legal sanity checks)
GUARDRAILS     Tier 3 + auditable HITL approval flows + per-tenant policy
ORCHESTRATION  Per-tenant graph variants where business logic diverges
STATE          Per-tenant data isolation (schema-per-tenant or row-level security)
OBSERVABILITY  Per-tenant cost attribution; per-tenant SLA dashboards; audit logs
INTERFACE      Per-tenant routing (subdomain / header / token); per-tenant rate quotas
```

**Verdict for Tier 4:** at this point you're not building an LLM app, you
are building *infrastructure* for LLM apps. Different game.

---

## Decision matrix — per-concern at a glance

When designing a new component, walk down this matrix. Each column tells
you what to use at that tier.

| Concern | Tier 1 · Prototype | Tier 2 · Pilot | Tier 3 · Production | Tier 4 · Multi-tenant |
|---|---|---|---|---|
| **Provider** | Hardcoded | Adapter + env | Adapter + cost router | Per-tenant adapter |
| **Fallback** | — | Built-in `with_fallbacks` | Tenacity retries + fallback + circuit breaker | Per-tenant policy |
| **Validation** | `with_structured_output` | + `field_validator` | + sanitizer + judge | + tenant business rules |
| **Guardrails** | — | PII + cost cap | All 4 axes + safety classifier | + audit + HITL |
| **Orchestration** | LCEL | `create_agent` + MW | Supervisor / swarm | Per-tenant graphs |
| **Checkpointer** | MemorySaver | SqliteSaver | AsyncPostgresSaver | Postgres + isolation |
| **Long-term mem** | — | — | `Store` (Postgres) | Per-tenant `Store` |
| **Observability** | `print()` | LangSmith hosted | + OTel + eval suite | + per-tenant dashboards |
| **Interface** | Notebook | FastAPI sync | + streaming + batch | + per-tenant routing |

The diagonal — "make all decisions at the same tier" — is the easiest to
debug. Mixing tiers (Tier-3 reliability + Tier-1 observability) is the
shape of a 3-AM incident: you can't see what just broke.

---

## Architecture anti-patterns (from real failures)

### The "god prompt"

A single 4,000-token prompt that does everything. Hard to test, hard to
debug, brittle to model changes.

**Fix:** decompose into chains (LCEL), graphs (StateGraph), or agents
(`create_agent`). One prompt per *step*.

### The "trust-and-store"

LLM output → straight into the database. No validation, no audit. Then a
year later someone discovers half the birth dates are wrong.

**Fix:** validation layer (Pydantic + deterministic parser) is mandatory
before *anything* hits the database. Lesson 21's pattern.

### The "silent fallback"

LLM call fails, fallback fires silently, user gets a worse model's
answer without knowing. Cost spike at month-end, no alarm.

**Fix:** observability layer — every fallback emits a structured log entry
with the trigger error and chosen replacement. Alert on fallback-rate
spikes.

### The "agent everywhere"

`create_agent` used for tasks that don't need tools or reasoning. Adds the
agent loop overhead (model + tool call decision + tool call + model again)
for what could be a one-shot LCEL chain.

**Fix:** **start at the simplest orchestration shape.** Promote only when
proven necessary.

### The "synchronous streaming"

Streaming endpoints (`/stream`) that block the worker for 30 seconds.
Eats your worker pool, kills throughput.

**Fix:** async all the way down — `AsyncPostgresSaver`, `aclient`, FastAPI
async handlers, `ASGITransport`. Lesson 14 + the `rag_qa_api` capstone show
this end-to-end.

### The "fine-tune for facts"

"Our model doesn't know our product catalogue → let's fine-tune!" Six
weeks later, the catalogue changes and the model is stale again.

**Fix:** RAG (lesson 06-07). Fine-tune for *style, format, behaviour* —
not facts. Decision tree in [`skills/llm-expert`](../../skills/llm-expert/SKILL.md).

### The "single LLM-as-judge"

Using the same model that produced the output to also judge it. Self-grading
is biased upward.

**Fix:** judge with a *different* model — usually a cheaper one with a
clearer rubric. Or: use a deterministic check (regex, dateparser, schema)
where possible.

### The "no eval, no progress"

Iterating on prompts without a held-out eval set. Each change "looks
better" but you can't tell when you're improving vs. regressing.

**Fix:** 20-50 prompt eval suite from day 1. Pin it before iterating.
Lesson 21's `EVAL_SET` shows the seed.

### The "decomposed-fields hallucination" (a.k.a. wide-schema drift)

Splitting one logical value across N sub-fields and asking the LLM to
do the joining / conversion in-prompt. The canonical example:

```python
# wide schema — model fills 3 ints + a string
birth_day: int, birth_month: int, birth_year: int, father_name: str
# vs the narrow shape — model fills 2 strings
birth_date: str, father_name: str
```

Both versions look reasonable. The wide one fails *worse*: every
extra numeric component is an independent chance for the model to
emit the wrong digit, and the prompt typically asks the model to
*also* do the conversion (e.g. Turkish *"on iki"* → `12`). Real-world
failure mode observed on a voice-to-tool pipeline: user says
*"yirmi beş on iki yetmiş dokuz"* (25 / 12 / 79), the model fills
`{birth_day: 25, birth_month: 10, birth_year: 1979}` — months
silently drifted, the other two fields look correct, audit passes,
the wrong DOB lands in the database.

**Fix:** narrow your schema. Prefer **one coherent string** (`"1979-12-25"` or
even the raw spoken span) plus a deterministic server-side parser.
The LLM does NER (the easy part); the parser does the conversion (the
deterministic part). Worked example in [lesson 21 · "Why one string
field, not decomposed ints"](../21_date_parsing/README.md#why-one-string-field-not-decomposed-day--month--year-ints).

This generalises beyond dates: addresses (don't decompose into
`street_number`/`street_name`/`city`/`zip`), currency amounts
(don't decompose into `whole_part`/`fractional_part`/`currency_code`),
phone numbers (don't decompose into `country_code`/`area_code`/
`subscriber`). Every extra field is another roll of the hallucination
die — and any deterministic transformation the prompt asks for is
work that belongs in code.

---

## How this repo embodies the architecture

Mapping each layer to the concrete files in this codebase:

| Layer | Files |
|---|---|
| **8 · Interface** | `projects/rag_qa_api/app.py` (FastAPI), `projects/customer_support_bot/graph.py` (CLI) |
| **7 · Observability** | `LANGSMITH_*` env vars in `shared/settings.py`, lesson 03's `with_listeners` |
| **6 · State** | `MemorySaver` / `SqliteSaver` / `AsyncPostgresSaver` usage in lessons 12, 18, capstones |
| **5 · Orchestration** | `lessons/08-17/*`, `create_agent` in lesson 10, supervisor (16), swarm (17) |
| **4 · Guardrails** | `lessons/19_guardrails/`, the customer_support_bot's middleware stack |
| **3 · Validation** | `lessons/21_date_parsing/`, `shared/llm/schema_sanitizer.py`, every Pydantic schema |
| **2 · Reliability** | `shared/llm/base.py` (fallback chain, auto-promote), the rag_qa_api's PostgresSaver |
| **1 · Provider** | `shared/llm/{__init__,base,anthropic_adapter,openai_adapter}.py` |

When you read a capstone like `projects/research_assistant/graph.py`,
you're seeing all 8 layers compose. Walking down the file from imports to
the bottom is roughly walking down the layer stack.

---

## When to break the rules

Every layer above is *the default*. There are legitimate reasons to skip
or invert it:

- **Skip Provider Layer** when you're explicitly tied to one provider (regulated environment, single-vendor contract). The adapter overhead doesn't pay off.
- **Skip Reliability Layer** for batch / async workloads where the worker can simply retry the whole job. Don't double up.
- **Skip Validation Layer** for *exploratory* / *analytics* outputs where wrong answers are visible to a human reviewer who can correct them.
- **Skip Guardrails** when the model is air-gapped from end users — internal tooling, dev environments.
- **Invert State and Orchestration** when the orchestration is *driven* by state changes (event-sourced agents, Temporal/Restate workflows). State Layer becomes the entry point.

Knowing why the default exists is the only way to break it safely.

---

## The "boring tech" rule, applied

Choose **boring** technology at every layer unless you have a specific
reason not to. For LLM apps in 2026:

| Layer | Boring choice |
|---|---|
| Provider | Anthropic or OpenAI primary, the other as fallback |
| Reliability | LangChain `with_fallbacks` + Tenacity retries |
| Validation | Pydantic v2 + `dateparser` |
| Guardrails | LangChain middleware + a regex / Pydantic-based judge |
| Orchestration | `create_agent` first, `StateGraph` second |
| State | Postgres via `AsyncPostgresSaver` |
| Observability | LangSmith (or Langfuse for self-hosted) |
| Interface | FastAPI + SSE |
| Package mgmt | `uv` |
| Container | Plain Docker |

**Boring** doesn't mean "old". It means "well-understood failure modes",
"the team has heard of it", "the docs exist", "there's a Stack Overflow
answer". Save the novelty budget for the *product*, not the substrate.

---

## Try it yourself (architectural exercises, no code required)

- **Reverse the layering for one of the capstones.** Draw the file
  `projects/rag_qa_api/app.py` as the 8 layers. Which lines belong to
  which layer? Are any responsibilities crossing a layer boundary?
- **Map an outage to a missing layer.** Take a real LLM-app incident
  you've heard about (or one of the anti-patterns above). Which layer
  was missing? At what tier?
- **Promote one capstone from Tier 2 to Tier 3.** What concretely
  changes in each layer?
- **Design a Tier 4 multi-tenant version of `customer_support_bot`.**
  Where does tenant isolation enter each layer?

## Pairs with

- **[`docs/architecture.html`](../../docs/architecture.html)** — the visual companion (module map, adapter pattern, capstone diagrams) of this same architecture, in pretty form.
- **[Lesson 19 · Guardrails](../19_guardrails/README.md)** — the guardrail layer in depth.
- **[Lesson 21 · Date parsing](../21_date_parsing/README.md)** — the validation layer in depth.
- **[`docs/research/llm-date-solutions-deep-dive.md`](../../docs/research/llm-date-solutions-deep-dive.md)** — research backing the multi-approach defence-in-depth philosophy.
- **[`skills/llm-expert`](../../skills/llm-expert/SKILL.md)** — the deeper "how do LLMs actually work" layer the architecture sits on.

## Next →

That's the curriculum. From here, build something: the capstones in [`projects/`](../../projects/) are good starting points to apply this architecture to a real app.
