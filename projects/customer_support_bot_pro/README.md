# Capstone · `customer_support_bot_pro`

The Tier 6 version of `customer_support_bot`. Same agent loop (RAG +
refund tool + HITL + Sqlite persistence) but **scaled and hardened**:

| Layer | What's added |
|---|---|
| Prompts | Jinja templates, per-tenant persona, locale switching |
| Concurrency | Per-customer lock map prevents racing messages |
| Idempotency | `request_refund` carries an idempotency key — safe to replay |
| Tools | Bounded LLM concurrency; structured `recoverable_error` returns |
| Persistence | Sqlite (single process) OR Postgres (multi-replica) — toggled by env |

```
            user (different customer_ids)
              │
              ▼
        per-customer lock                ← lesson 27 pattern 3
              │
              ▼
       create_agent (LLM + tools)
        │   │
        │   ├── lookup_handbook   (RAG via lesson 06's Chroma)
        │   ├── request_refund    (idempotent + interrupt for >$100)
        │   └── escalate_ticket   (recoverable_error returns)
        │
        ▼
       summarisation_middleware (if context > N tokens)
              │
              ▼
          checkpointer
```

## What it teaches (concept → lesson)

| Element | Lesson |
|---|---|
| `create_agent` + middleware | 10, 11 |
| RAG | 06 |
| HITL interrupt + resume | 13 |
| Persistence (Sqlite / Postgres) | 12 |
| Per-customer lock map | 27 |
| Jinja templates + persona inheritance | 28 |
| Idempotency-key tool design | 31 |
| Recoverable-error tool returns | 25 |

## Prerequisites

```bash
uv sync                     # installs jinja2, langgraph extras
# Optional Postgres mode:
#   docker compose -f ../../lessons/29_vector_databases/docker-compose.yml up -d pgvector
```

## Run it

```bash
# Sqlite mode (default — works zero-setup):
uv run python -m projects.customer_support_bot_pro.graph

# Postgres mode:
SUPPORT_BOT_BACKEND=postgres uv run python -m projects.customer_support_bot_pro.graph
```

The chat shows the customer id in the prompt — switch customers with
`/login alice@acme.com` to see per-customer thread isolation.

Useful commands:

```text
> How many PTO days do I get?
> /login alice@acme.com
> /login bob@globex.com
> I want a refund of $250 on order #ABC-2
⚠  HITL: {...}
> /approve
> /quit
```

## Concurrency walkthrough

```python
# graph.py module-level
_customer_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
LLM_SEM = asyncio.Semaphore(8)
```

When the CLI processes a user message:

```python
async def handle(customer_id: str, message: str):
    async with _customer_locks[customer_id]:
        async with LLM_SEM:
            return await agent.ainvoke(...)
```

Two messages from the same `customer_id` are processed serially.
Different customers run in parallel. This is the right shape for any
chat server — without it, two messages a second apart can corrupt the
conversation state.

## Idempotency walkthrough

`request_refund` accepts an `idempotency_key`:

```python
@tool
async def request_refund(order_id: str, amount: float, reason: str, idempotency_key: str) -> dict:
    """Issue a refund. Replays with the same idempotency_key return the cached result."""
```

The agent generates the key from `f"{customer_id}:{order_id}:{round(amount, 2)}"`.
On a retry, the same key returns the same receipt — no double-charge.
This is the lesson 31 pattern applied at the tool layer.

## Prompt walkthrough

`prompts/agents/support.j2` inherits from `_base/persona.j2`. The
system prompt is rendered **per turn** with the current customer's
data:

```python
def system_for_turn(state, runtime):
    return env.get_template("agents/support.j2").render(
        agent_name="Acme Support",
        company="Acme",
        customer=runtime.context["customer"],
        tier=runtime.context["customer"].get("tier", "free"),
        locale=runtime.context["customer"].get("locale", "en-US"),
    )
```

This is what dynamic prompting actually buys you: an enterprise
customer sees "...you have priority support" in the system prompt; a
free customer doesn't. Same code, different rendered output, no
spaghetti `if/else` chain in Python.

## Try it yourself

1. **Distributed lock.** Replace `_customer_locks` with a Redis SETNX lock (lesson 31) so two API replicas can both serve `alice@acme.com` without racing.
2. **Per-tenant Chroma collections.** Index the handbook into `chroma["customer-{id}"]` so different organisations have isolated FAQ corpora.
3. **Switch to pgvector.** Move from Chroma to pgvector — gives you tenant-filtering for free (lesson 29).
4. **Add a `pii_redaction` middleware.** Strip credit-card numbers before the LLM ever sees them; the redacted text is what the model gets, the original stays in your DB.

## Pairs with

- [Lesson 11](../../lessons/11_agent_middleware/README.md), [Lesson 13](../../lessons/13_human_in_the_loop/README.md), [Lesson 27](../../lessons/27_locks_and_concurrency/README.md), [Lesson 28](../../lessons/28_dynamic_prompting/README.md), [Lesson 31](../../lessons/31_distributed_locks/README.md).
