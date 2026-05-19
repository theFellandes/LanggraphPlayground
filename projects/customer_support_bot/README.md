# Capstone · `customer_support_bot`

A stateful customer-support bot for a fictional company. Combines:

- **`create_agent`** (lesson 10) — the agent loop
- **`summarization_middleware`** (lesson 11) — keeps long chats from blowing context
- **`SqliteSaver`** (lesson 12) — every thread persists to `data/support_bot.sqlite`
- **`interrupt()`** (lesson 13) — refunds over $100 pause for human approval
- **RAG retrieval** (lesson 06) — the `lookup_handbook` tool queries a Chroma index built from `data/sample_docs/company_handbook.md`

## How it works

The agent has two tools:

1. **`lookup_handbook(question)`** — semantic search over the company handbook for policy questions ("how much PTO?", "what's the refund policy?").
2. **`request_refund(order_id, amount, reason)`** — if `amount <= 100`, processes immediately; if `amount > 100`, calls `interrupt(...)` and waits for the operator to approve or reject.

## Run it

```bash
uv run python -m projects.customer_support_bot.graph
```

Then chat at the `>` prompt. Try:

```text
> How many PTO days do I get?
> I want a refund of $42 on order #ABC-1
> I want a refund of $250 on order #ABC-2
⚠  HITL needed: {...}
  Type /approve or /reject.
> /approve
```

Other slash commands:

- `/quit` — exit
- `/approve` — resume a pending interrupt with approval
- `/reject` — resume a pending interrupt with rejection

The conversation is persisted under `thread_id="cli-session"` in
`data/support_bot.sqlite`. Restart the script and it resumes where you
left off.

## Concepts exercised

| Lesson | What this capstone uses |
|---|---|
| 04 · structured output | (none — kept the demo simple) |
| 05 · tools | `@tool` for both handbook lookup and refunds |
| 06 · RAG basics | Chroma index over the handbook |
| 10 · `create_agent` | the agent loop |
| 11 · middleware | `summarization_middleware` |
| 12 · persistence | `SqliteSaver` |
| 13 · HITL | `interrupt()` + `Command(resume=...)` |

## Try it yourself

- Add `pii_redaction_middleware` so credit-card numbers never reach the LLM.
- Replace `SqliteSaver` with `PostgresSaver` and run two CLI sessions against the same DB to simulate two agents working in parallel.
- Wrap the bot in a FastAPI endpoint (cribbing from `projects/rag_qa_api/`) and stream tokens to a chat UI.
