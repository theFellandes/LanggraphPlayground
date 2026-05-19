# Lesson 19 · Guardrails

## What you'll learn

- The **four axes** of an LLM guardrail: input · output · tool · conversation
- The **three implementation patterns** the industry uses: middleware, decorator, layer-as-node
- How to **compose them** on a single `create_agent` and watch them trip on bad inputs
- Where the major libraries fit: **LangChain middleware**, **NeMo Guardrails**, **Guardrails AI**, **Llama Guard / ShieldGemma**, **LLM-as-judge**

## Why it matters

The day you put an agent in front of real users, you discover the
model will:

- happily echo back a credit card number you handed it,
- call your `process_refund` tool with a $9,999,999 argument,
- get stuck in a tool loop that costs you 200 model calls,
- and produce free-text answers your downstream system can't parse.

Guardrails turn those failure modes from "incidents" into "ToolMessage
errors the model can react to" or "rejected before the model even
saw the input." None of them are optional in production.

## Key concepts

### The four guardrail axes

| Axis | Stops what | Examples |
|---|---|---|
| **Input** | bad stuff *reaching* the model | PII redaction, prompt-injection detection, profanity filter, off-topic routing |
| **Output** | bad stuff *leaving* the model | hallucination check, schema/format validation, content safety, citation-required |
| **Tool** | bad *side effects* | allowlist, arg validation (refund cap), dry-run mode, approval gate |
| **Conversation** | bad *aggregates* | model-call cap, token cap, latency SLA, context-summary trigger |

### The three implementation patterns

**Pattern A · Middleware** (`langchain.agents.middleware`)
Compose `AgentMiddleware` subclasses around `create_agent`. Six hook
points: `before_agent`, `before_model`, `wrap_model_call`,
`wrap_tool_call`, `after_model`, `after_agent`. Cross-cutting, reusable,
declarative.

```python
agent = create_agent(
    model=..., tools=[...],
    middleware=[PIIMiddleware(pii_type="email", strategy="redact"),
                ModelCallLimitMiddleware(max_calls=6),
                MyCustomGuardrail()],
)
```

**Pattern B · Decorator** (Python `@wraps` around a `@tool`)
Stick to one tool when the rule is tool-specific. The decorated
function raises → the LLM sees a `ToolMessage(status="error")` and
typically apologises and tries a different path.

```python
@tool
@validate_tool_args(_refund_policy)
def process_refund(order_id: str, amount: float): ...
```

**Pattern C · Layer-as-node** (a dedicated node in the graph, or a
post-`invoke` step)
For *whole-conversation* checks that don't fit into any of the
middleware hooks — most often **output schema validation** or an
**LLM-as-judge** review.

```python
result = agent.invoke({...})
final  = judge_output(result["messages"][-1].content)   # Pydantic-typed
```

### When to reach for which

| Need | Use |
|---|---|
| Same policy across many tools/agents | **Middleware** |
| Per-tool argument rule | **Decorator** |
| Strict typed output the caller depends on | **Layer-as-node** with Pydantic |
| Pause for a human | `HumanInTheLoopMiddleware` (lesson 13 also shows graph-level `interrupt()`) |
| Cap how much it can spend | `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` (or custom `wrap_model_call`) |
| Detect prompt injection / jailbreak attempts | Llama Guard / ShieldGemma model call, wrapped as middleware |
| Topical / structural rails ("don't discuss competitors") | NeMo Guardrails (Colang flows) |
| Schema-rich validators with retry-on-fail | Guardrails AI |

## Industry overview (one paragraph each)

- **LangChain middleware (v1)** — the *path of least resistance* if you're already on `create_agent`. Six well-named hooks, batteries-included prebuilts (`PIIMiddleware`, `HumanInTheLoopMiddleware`, `ModelCallLimitMiddleware`, `ToolRetryMiddleware`, `SummarizationMiddleware`, `ContextEditingMiddleware`, `ModelFallbackMiddleware`). Pure Python, no DSL.
- **NeMo Guardrails (NVIDIA)** — defines *Colang* flows ("user asks about X → don't answer, redirect to Y"). Better for *topical* and *dialogue-shape* rails than for code-level argument checks. Heavy: ships its own dialog manager.
- **Guardrails AI** — the `guardrails-ai` Python library. RAIL / Pydantic schemas + a registry of pluggable *validators* (`ToxicLanguage`, `PolitelyDeclines`, `RegexMatch`, etc.) with built-in retry-on-fail. Plays nicely with structured-output workflows.
- **Llama Guard / ShieldGemma / similar safety models** — dedicated classifier models that score input/output for safety categories (hate, self-harm, sexual content, etc.). You call them as an extra LLM step, usually inside a middleware. Best in concert with the patterns above, not as a replacement.
- **LLM-as-judge / self-critique** — run a (usually cheaper) model that scores the primary model's output against a rubric, then route based on the score. Cheap to add, surprisingly effective for "is this answer hallucinated?" and "does this answer cite its sources?" Combine with Pattern C.

## Walk through `example.py`

A banking-support agent (lookup orders, issue refunds) protected by
all three patterns at once:

| Layer | Pattern | Implementation |
|---|---|---|
| Redact emails before model sees prompt | Middleware | Prebuilt `PIIMiddleware(pii_type="email", strategy="redact")` |
| Block credit-card numbers entirely    | Middleware | Prebuilt `PIIMiddleware(pii_type="credit_card", strategy="block")` |
| Cap to 6 model calls per run          | Middleware | Custom `CostCap(AgentMiddleware)` overrides `wrap_model_call` |
| Refunds > $200 must be escalated      | Decorator  | `@validate_tool_args(_refund_policy)` on `process_refund` |
| Final reply must match a Pydantic shape | Layer-as-node | `judge_output(raw)` calls a structured-output LLM |

Three scenarios run end-to-end:

1. **Clean request** — all guardrails pass; the judge returns a typed `FinalAnswer`.
2. **Request containing an email** — `PIIMiddleware` redacts before the model sees the prompt; the model's reply is then sanitised by the judge (`contains_pii=False`).
3. **Refund over the $200 cap** — the `@validate_tool_args` decorator raises; the model receives a `ToolMessage(status="error", content="REFUND_BLOCKED ...")` and reroutes to "I'll escalate this."

## Run it

```bash
uv run python -m lessons.19_guardrails.example
```

You should see, for scenario 3, a model call followed by the
decorator's `ValueError` surfacing as a tool-error message — the
agent then handles it gracefully instead of crashing.

## Debug it

Put `breakpoint()` inside `CostCap.wrap_model_call` and step
through one scenario. Inspect `request` — that's exactly what the
agent is about to send to the LLM. The same hook is where you'd
implement **request caching**, **provider fallback**, or
**prompt-injection scanning** in production.

For Pattern B (decorator), set a breakpoint inside `_refund_policy`
to see what the model tried to pass before validation rejected it
— you'll learn a lot about how the model phrases its tool calls.

## Try it yourself

- **Add a 4th middleware** that detects prompt-injection patterns
  (start cheap: a regex for `"ignore previous instructions"`). Reject
  in `before_model` by raising or by mutating `request`.
- **Replace `CostCap` with the prebuilt** `ModelCallLimitMiddleware`
  to see how the official one signals limit-hits cleanly.
- **Swap the judge for a real Llama Guard call** — call a hosted
  safety classifier and only return the answer if `safe == True`.
- **Wrap with `HumanInTheLoopMiddleware`** so refunds always pause
  for `/approve`-style resumption (see lesson 13).

## A note on layering

Guardrails are **defence-in-depth**, not single-point. The cheapest
ones (regex/length checks in `before_model`) run on every call; the
expensive ones (LLM-as-judge, Llama Guard) run only on outputs that
will actually be returned to the user or persisted. Compose them
in this order:

```
[ cheap input filter ]                ← middleware before_model
        │
        ▼
[ model call (+ wrap_model_call)  ]   ← cost cap, caching, fallback
        │
        ▼
[ tool guardrail per @tool        ]   ← decorator
        │
        ▼
[ schema / judge on final answer  ]   ← layer-as-node
```

If a layer rejects, **fail closed** and return the user a generic,
helpful message — never echo back the raw exception or the rejected
input.

## Next →

You've finished the curriculum. Head to the capstones:

- [research_assistant](../../projects/research_assistant/README.md)
- [customer_support_bot](../../projects/customer_support_bot/README.md)
- [rag_qa_api](../../projects/rag_qa_api/README.md)

All three are good places to wire up guardrails for real.
