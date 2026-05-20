# Lesson 25 · Tool design patterns for reliable LLM agents

Tool-writing wisdom is scattered across lessons 05 (basics), 10
(`create_agent`), 19 (guardrails + `@validate_tool_args`), 23 (date
tools), and 24 (the spoken-number wrapper). This lesson consolidates
it into a single reference: **when to make something a tool, how to
write its description, what shape to return, how errors propagate,
and how the tool itself carries enough context that the agent doesn't
have to re-fetch the whole conversation.**

## What you'll learn

- When something *should* be a tool — and when it shouldn't
- The single most underrated lever in agent reliability: the tool's
  **description** and **`Annotated[type, "…"]`** parameter docs
- The **rich-return-value** pattern: structured `{ok, value, metadata,
  warnings}` dicts so the agent reads everything it needs from one
  call — no need to round-trip back through the whole conversation
- **Recoverable vs unrecoverable** errors — when to return an error dict
  vs raise — and why this turns the agent loop into self-correction
- The **confidence + alternatives** pattern that drives the agent's
  next-action choice (accept / confirm / repeat / escalate)
- **Router**, **wrapper**, and **enricher** composition patterns
- A surprising real-world pitfall: `Literal[...]` parameter types
  cause Pydantic to reject unknown values **before** your function can
  produce a helpful error
- A real fragility: generic `@tool` wrappers lose the inner tool's
  schema — and the right way to enrich a tool while preserving its
  signature

## Why it matters

Tools are the seam where business logic meets LLM non-determinism. A
well-designed tool **reduces** the LLM's hallucination surface (the
LLM only has to identify a span and pick an arg; the tool does the
deterministic part). A poorly-designed tool **amplifies** hallucination
(the LLM has to fill many fields with no feedback). Lessons 21 and 24
showed the failure mode; this lesson is the prevention.

## Key concepts

### 1 · When something should be a tool

| Make it a tool when… | Don't make it a tool when… |
|---|---|
| The work is **deterministic** (arithmetic, parsing, lookup) | The work is generative (writing prose) |
| The work has **side effects** (DB write, email, payment) — gate with HITL | Pure transformation of agent state |
| The work needs **external access** (API, DB, file system) | The LLM can do it in one model call cheaply |
| The work has a **precise contract** that fails predictably | The work is vague or under-specified |
| You need an **audit trail** of "what the agent did" | One-shot text generation |

**Rule of thumb:** if you would write a unit test for it in normal code, it should probably be a tool. The unit test boundary is the tool boundary.

### 2 · The tool description — the underrated lever

The `@tool` decorator turns:

- the docstring → the tool's natural-language description (what the LLM sees)
- the type hints + `Annotated[T, "…"]` per-param → the tool's JSON schema (what the LLM is constrained by)

Both are **prompt** to the LLM. Treat the docstring like a system message:

```python
@tool
def process_refund(
    order_id:  Annotated[str,   "Order id exactly as the customer said it. Do NOT modify or guess."],
    amount:    Annotated[float, "Refund amount in USD. Must be > 0 and <= the order total."],
    reason:    Annotated[str,   "Customer's stated reason. Copy verbatim; do not paraphrase."] = "",
) -> dict:
    """Issue a refund. Refunds above $200 require human approval.

    Returns:
      {ok: True,  refund_id: str}                          # success
      {ok: False, error: "needs_human_approval", ...}      # over policy cap
      {ok: False, error: "amount_invalid",       ...}      # caller's mistake

    Decision policy:
      - On "needs_human_approval" → pause and escalate (see lesson 19)
      - On "amount_invalid"        → fix and retry; this means YOU passed a bad value
    """
```

Three things worth their own line:

- **"Do NOT modify or guess"** — the magic words for verbatim copying. Without them, models *helpfully* normalise input (a recipe for hallucinated digits, as lesson 21 documented).
- **A "Returns" section that enumerates the shapes** — the LLM reads it and learns to expect the union; downstream prompts can say *"if the return contains `needs_human_approval`, do X"*.
- **A "Decision policy" section** — tells the LLM what to do on each return shape so the agent's system prompt doesn't have to repeat it.

### 3 · Return rich, structured dicts — not bare values

The single most impactful design choice. Compare:

```python
# Bare — the agent has nothing to work with
def parse_birth_date(text: str) -> str:
    return "1987-04-05"

# Rich — the agent has everything it needs to decide next-action
def parse_birth_date(text: str) -> dict:
    return {
        "ok":           True,
        "iso":          "1987-04-05",
        "confidence":   0.95,
        "engine":       "dateparser",
        "alternatives": [{"iso": "1987-05-04", "score": 0.65}],
        "warnings":     [],
    }
```

With the rich return, the agent can act on **`confidence`** without
asking another LLM call, can present **`alternatives`** to the user
verbatim, and can attribute *which* engine produced the value (useful
for debugging and logging in lesson 22's observability layer).

> **The deeper principle.** When a downstream step needs to decide
> something based on the tool's result, the tool should return enough
> metadata that the decision can be made *from the tool's output
> alone* — without re-sending the whole prior conversation back
> through an LLM. **Stuff the metadata in the tool's return; don't
> re-derive it.** Token cost stays bounded; the agent stays in its
> existing loop.

### 4 · Recoverable vs unrecoverable errors

Two distinct error contracts; pick deliberately:

| Pattern | When | Effect on the agent loop |
|---|---|---|
| **Return `{ok: False, error: ...}`** | The agent could fix and retry (wrong arg, policy block the user can resolve) | LLM reads the error in the next ToolMessage and corrects |
| **`raise SomeError`** | Unrecoverable (downstream API down, programmer bug, security violation) | The agent loop halts; surfaces to the calling system |

Live demo output:

```
process_refund(ABC-1, $50.0)  → ok                     refund_id: ref_…
process_refund(ABC-2, $250.0) → needs_human_approval   suggestion: "/approve"
process_refund(ABC-3, $-5.0)  → amount_invalid         detail: "must be positive"
```

The first is success. The second is **recoverable** (the agent
escalates to HITL middleware — lesson 19). The third is **also
recoverable** (the LLM passed a bad value; it'll fix and retry on its
own). None of them throw.

If `process_refund` discovered fraud on the account, **then** it would
`raise RefundDeniedError(...)` — an unrecoverable error the agent
should NOT retry around.

### 5 · The confidence + alternatives pattern

The pattern from lesson 24, generalised: any tool whose result has
*uncertainty* should expose that uncertainty as a number, not hide it.

```python
result = parse_birth_date("April 5, 1987")
# {"ok": True, "iso": "1987-04-05", "confidence": 0.95, ...}

# Agent's decision policy (encoded in the agent's system prompt):
#   confidence >= 0.95  →  use iso directly
#   0.80 <= conf < 0.95 →  ask user: "Did you mean April 5, 1987?"
#   confidence < 0.80   →  ask user to repeat
```

The numbers come from somewhere principled:

- For NLP parsers: `rapidfuzz` partial_ratio on the closest vocab match (lesson 24)
- For retrievers: the embedding similarity score
- For LLM-as-judge: the judge model's stated confidence
- For external APIs: their own confidence field, if present; otherwise `1.0` for success, `0.0` for error

Three tiers (accept / confirm / repeat) is the right granularity for
most agents. Five tiers is over-engineering; binary accept/reject is
under-engineering.

### 6 · The "don't send the whole conversation back" principle

A natural impulse on a low-confidence result is: *"send the whole
conversation to a more powerful LLM and let it disambiguate."* Three
reasons not to:

1. **The agent already has the conversation.** In an agentic flow with
   `create_agent` or a `StateGraph` using `MessagesState`, the
   conversation IS the state. The next agent step sees everything
   without re-supplying it.
2. **Token cost scales super-linearly** with conversation length × ambiguous tokens. A 20-turn chat with one ambiguous date = ~30 KB of resent tokens, every retry.
3. **Misleading context.** Prior turns can bias the LLM. If the
   user was just told "your balance is $1,987", the LLM might "see"
   1987 in an ambiguous birth-year slot.

**What to send instead.** The minimum payload: the original input,
the parser's best guess, the confidence, and at most 2-3 alternatives.
The agent's existing prompt + existing state handle the rest.

### 7 · Router / dispatch tools — one surface, N implementations

```python
@tool
def parse_number(text: str, locale: str = "en") -> dict:
    """Parse a spoken-form number. Routes by locale; returns supported list on error."""
    if locale == "tr":
        return {"ok": True, "value": _our_turkish_parser(text), "engine": "turkish_rule_parser"}
    if locale in {"en", "fr", "es", "de"}:
        return {"ok": True, "value": text2num(text, locale), "engine": "text2num"}
    return {"ok": False, "error": f"unsupported locale {locale!r}",
            "supported_locales": ["en", "fr", "es", "de", "tr"]}
```

Key choice: **`locale: str` not `Literal[...]`.** A Literal would have
Pydantic reject `"jp"` at the schema layer, and the LLM gets a terse
Pydantic ValidationError. Keeping the type as `str` lets the function
return a structured error with `supported_locales` — the LLM reads
the list and retries with a valid value. **Trade-off: lose schema-time
validation; gain explainability.** For router-shape tools, the latter
wins almost always.

### 8 · Wrappers / enrichers — preserve the inner tool's schema

Caching, logging, retries, fallbacks — these are cross-cutting
concerns that *should* compose at the tool layer. **But** a generic
`@tool` wrapper has a real pitfall:

```python
# DOESN'T WORK reliably — wrapper's **kwargs signature loses the inner
# tool's param names, so the schema the LLM sees is empty
def with_cache(inner_tool):
    @tool(inner_tool.name + "_cached")
    def wrapped(**kwargs):
        ...
    return wrapped
```

Two right shapes:

1. **Domain-specific explicit wrapper** (simpler, recommended):

   ```python
   @tool
   def cached_lookup_order(order_id: str) -> dict:
       """Cached lookup_order — same args, adds `from_cache: bool` to return."""
       if order_id in _cache: return {**_cache[order_id], "from_cache": True}
       result = lookup_order.invoke({"order_id": order_id})
       if result.get("ok"): _cache[order_id] = result
       return {**result, "from_cache": False}
   ```

2. **Build a StructuredTool by hand** (if you really need a generic decorator):

   ```python
   from langchain_core.tools import StructuredTool
   def with_cache_generic(inner: StructuredTool) -> StructuredTool:
       def _wrapped(**kw):
           ...
       return StructuredTool.from_function(
           func=_wrapped,
           name=inner.name + "_cached",
           description=inner.description,
           args_schema=inner.args_schema,   # ← reuse the original schema
       )
   ```

For middleware-style concerns that *don't* need to live in the tool
surface (auth, rate limiting, audit logging), use **lesson 11's
middleware system** instead — `AgentMiddleware.wrap_tool_call` is
exactly the right hook.

## Walk through `example.py`

Five worked patterns; all run without an API key (they call the tools
directly via `.invoke({...})`):

| Demo | Pattern | What it shows |
|---|---|---|
| 1 `--read-only` | Read-only data tool | `lookup_order` — structured dict with `ok` flag + `suggestion` on failure |
| 2 `--computation` | Confidence + metadata | `parse_birth_date` returns iso + confidence + alternatives + warnings; the demo shows accept/confirm/reject tiering |
| 3 `--side-effect` | Recoverable error contract | `process_refund` returns `{ok: False, error: …}` for policy blocks; raises only for unrecoverable cases |
| 4 `--router` | Dispatch tool | `parse_number` routes by locale; returns `supported_locales` on unknown |
| 5 `--enricher` | Cached wrapper | `cached_lookup_order` adds `from_cache` to every return; explicit signature preserves the schema |

## Run it

```bash
uv run python -m lessons.25_tool_design.example          # all five
uv run python -m lessons.25_tool_design.example --read-only
uv run python -m lessons.25_tool_design.example --computation
uv run python -m lessons.25_tool_design.example --side-effect
uv run python -m lessons.25_tool_design.example --router
uv run python -m lessons.25_tool_design.example --enricher
```

## Debug it

Put `breakpoint()` inside any tool function and call it from the
demo. The most useful thing to inspect: the result dict's `ok`,
`error`, `confidence`, and any metadata fields — that's exactly what
the LLM sees and reasons about in the next step of the agent loop.

## Tool design anti-patterns

Beyond the ones already named in lessons 21 / 22:

| Smell | Fix |
|---|---|
| Tool returns a bare value (`str` / `int` / `bool`) | Return a dict with `ok` flag + value + metadata; future-proof against adding fields |
| Tool raises on conditions the agent can recover from | Return an `{ok: False, error: …}` dict; the LLM reads it and self-corrects |
| Tool swallows errors silently and returns "" or None | Always either return the error or raise it; never both, never neither |
| Tool's docstring says "what it does" but not "when to call it" / "what the args mean" | Add `Annotated[T, "…"]` per param; add a "Decision policy" section to the docstring |
| Tool takes 8+ args | Decompose into smaller tools or take a single structured input |
| Tool uses `Literal[...]` on a param the LLM might guess wrong | Switch to `str`; let the tool return `supported_*` lists |
| Generic higher-order `@tool` wrapper (loses inner schema) | Domain-specific explicit wrapper, or `StructuredTool.from_function(args_schema=inner.args_schema)` |
| Side-effect tool with no HITL gate | Wire `HumanInTheLoopMiddleware(interrupt_on={...})` from lesson 19 |
| Tool that reads global state and mutates it | Pass dependencies as args; the LLM can't see globals so behaviour becomes non-debuggable |
| Two tools that secretly depend on call order | Make the dependency explicit — one tool's output becomes the next's input |

## Tool catalogue — which kind for which job

| Job | Tool kind | Example |
|---|---|---|
| Fetch a record by id | **Read-only data** | `lookup_order(order_id)` |
| Convert / normalise data | **Computation + metadata** | `parse_birth_date(text)`, `parse_spoken_number(text, locale)` |
| Mutate external state (DB, payment, email) | **Side-effect with HITL gate** | `process_refund(order_id, amount)` |
| Pick between N implementations | **Router** | `parse_number(text, locale)` |
| Add caching / logging / metadata to an existing tool | **Domain-specific wrapper** | `cached_lookup_order(order_id)` |
| Cross-cutting policy across many tools | **Middleware**, not a tool | `AgentMiddleware.wrap_tool_call` (lesson 11) |

## Pairs with

- **[Lesson 05 · Tools (basics)](../05_tools/README.md)** — the `@tool` decorator and the tool-call round-trip
- **[Lesson 10 · `create_agent`](../10_create_agent/README.md)** — the agent loop that calls tools
- **[Lesson 11 · Agent middleware](../11_agent_middleware/README.md)** — where cross-cutting tool concerns live
- **[Lesson 19 · Guardrails](../19_guardrails/README.md)** — `@validate_tool_args` and the HITL-middleware pattern for side-effect tools
- **[Lesson 23 · Date computation & localized output](../23_date_localization/README.md)** — the date tools (`today_iso()`, etc.) as worked examples of the computation pattern
- **[Lesson 24 · Spoken-number normalization](../24_spoken_numbers/README.md)** — the source of the confidence-tiered tool and the partial-match escalation policy
- **[Lesson 22 · Architecture](../22_architecture/README.md)** — the layered view; this lesson is mostly Validation Layer + Orchestration Layer concerns

## References

### Vendor / official

- [LangChain · Tool calling guide](https://docs.langchain.com/oss/python/langchain/tools) · the v1 tool surface
- [LangChain · `@tool` and `StructuredTool` reference](https://reference.langchain.com/python/langchain-core/tools/)
- [OpenAI · Function calling guide](https://platform.openai.com/docs/guides/function-calling) · provider-side semantics
- [Anthropic Claude · Tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) · `strict: true`, `input_examples`, the tool-result round-trip

### Engineering blogs

- [Anthropic · *Tool use overview*](https://docs.anthropic.com/en/docs/build-with-claude/tool-use-overview) · the "specific tools beat general tools" guidance
- [LangChain blog · *Tool calling patterns*](https://blog.langchain.dev/) · routing and composition patterns at scale

## Next →

This is the curriculum closer for tool wisdom. From here, the
capstones in [`projects/`](../../projects/) put all 25 lessons into
practice — `customer_support_bot` in particular is where the
`parse_birth_date`, `parse_number`, `process_refund` and `lookup_*`
patterns from this lesson actually compose.
