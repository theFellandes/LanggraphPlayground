# LangGraph Playground

A **zero-to-hero curriculum** for building agentic apps with
**LangChain 1.x** and **LangGraph 1.x**. Nineteen small lessons walk
you from your first `llm.invoke("hi")` call to a multi-agent supervisor
running over a Postgres checkpointer, and three capstone projects pull
it all together into apps you'd actually ship.

By the time you finish, you'll have built:

- A **multi-agent research assistant** (supervisor + researcher + writer + critic)
- A **customer-support bot** with human-in-the-loop escalation and persistent memory
- A **RAG Q&A API** in FastAPI, running in Docker with Postgres-backed state

> 📖 **Prefer to browse visually first?** Open [`docs/curriculum.html`](docs/curriculum.html) (double-click — single-file, no server) for the same curriculum in a navigable HTML study guide. The companion [`docs/architecture.html`](docs/architecture.html) has the module map and capstone diagrams.

---

## Why this exists

LangChain and LangGraph cover overlapping territory. The rule of thumb
worth memorizing on day one:

> **Start with LCEL.** When you hit a wall — looping, branching on agent
> decisions, or carrying state between steps — switch to LangGraph.

LangChain is optimized for *building* agents quickly. LangGraph is
optimized for *running* them in production: durability, memory,
human-in-the-loop, and control. The curriculum below mirrors that
arc: you learn the simple thing first, then learn the more powerful
thing the moment the simple thing creaks.

---

## The stack

| Library | Version | Why it's here |
|---|---|---|
| `langchain` | 1.3.x | The new `create_agent` API + middleware |
| `langgraph` | 1.2.x | `StateGraph`, checkpointers, `interrupt` |
| `langgraph-supervisor` | 0.1.x | One-line supervisor agent |
| `langgraph-swarm` | 0.1.x | Peer-to-peer handoffs |
| `langchain-chroma` + `fastembed` | latest | Local vector store, zero extra API key |
| `pydantic` 2 + `pydantic-settings` | 2.x | Structured output, typed env |
| `ipdb` | 0.13.x | Interactive debugging (lesson 03) |
| `fastapi` + `uvicorn` | latest | Capstone API |

Python 3.11 or 3.12.

---

## Curriculum map

Check the boxes as you go.

### Tier 1 · LangChain fundamentals

- [ ] **[00 · Setup](lessons/00_setup/README.md)** — env vars, model factory, your first `llm.invoke`
- [ ] **[01 · Chat models](lessons/01_chat_models/README.md)** — messages, prompt templates, roles
- [ ] **[02 · LCEL chains](lessons/02_lcel_chains/README.md)** — the pipe operator, parallel + streaming + batch
- [ ] **[03 · Debugging](lessons/03_debugging/README.md)** — **ipdb** for Runnables and async, listeners, LangSmith preview
- [ ] **[04 · Structured output](lessons/04_structured_output/README.md)** — Pydantic schemas + `with_structured_output`
- [ ] **[05 · Tools](lessons/05_tools/README.md)** — `@tool`, `bind_tools`, the tool-call round-trip

### Tier 2 · RAG

- [ ] **[06 · RAG basics](lessons/06_rag_basics/README.md)** — load → split → embed → store → retrieve → generate
- [ ] **[07 · RAG advanced](lessons/07_rag_advanced/README.md)** — multi-query, parent-document, contextual compression
- [ ] **[20 · Chunking & parsing strategies](lessons/20_chunking_and_parsing/README.md)** — side-quest: four chunkers + three parser tiers (numbered 20 because added later)

### Tier 3 · LangGraph core

- [ ] **[08 · LangGraph basics](lessons/08_langgraph_basics/README.md)** — your first `StateGraph`
- [ ] **[09 · Conditional edges](lessons/09_conditional_edges/README.md)** — branching and bounded cycles
- [ ] **[10 · `create_agent`](lessons/10_create_agent/README.md)** — the LangChain v1 prebuilt agent
- [ ] **[11 · Agent middleware](lessons/11_agent_middleware/README.md)** — composable hooks around `create_agent`
- [ ] **[12 · Persistence](lessons/12_persistence/README.md)** — checkpointers, threads, time-travel
- [ ] **[13 · Human-in-the-loop](lessons/13_human_in_the_loop/README.md)** — `interrupt()` + `Command(resume=…)`
- [ ] **[14 · Streaming](lessons/14_streaming/README.md)** — `values` / `updates` / `messages` / `custom`
- [ ] **[15 · Subgraphs](lessons/15_subgraphs/README.md)** — graphs that embed graphs

### Tier 4 · Multi-agent + long-term memory

- [ ] **[16 · Supervisor](lessons/16_supervisor/README.md)** — one boss, many workers
- [ ] **[17 · Swarm](lessons/17_swarm/README.md)** — peer-to-peer handoffs
- [ ] **[18 · Long-term memory](lessons/18_long_term_memory/README.md)** — the `Store` API across threads

### Tier 5 · Production hardening

- [ ] **[19 · Guardrails](lessons/19_guardrails/README.md)** — input · output · tool · conversation; middleware + decorator + judge node; tour of NeMo, Guardrails AI, Llama Guard
- [ ] **[21 · Date parsing with LLMs](lessons/21_date_parsing/README.md)** — INPUT side. Four-layer pipeline (prompt → structured output → `dateparser` → field validator). Backed by [`date-parsing-with-llms.md`](docs/research/date-parsing-with-llms.md) + [`llm-date-solutions-deep-dive.md`](docs/research/llm-date-solutions-deep-dive.md)
- [ ] **[22 · LLM application architecture](lessons/22_architecture/README.md)** — purely architectural meta-lesson: the 8 layers, four scaling tiers, decision matrix per concern, anti-patterns from real production systems
- [ ] **[23 · Date computation & localized output](lessons/23_date_localization/README.md)** — OUTPUT side. The `today_iso()` tool pattern for date arithmetic + `babel` for locale-aware formatting ("23 Mayıs 2026", "23. Mai 2026", "23 مايو 2026", "1405/03/02" Jalali, "1447-12-06" Hijri, "2026年5月23日"). Runs without an API key for the localization demo
- [ ] **[24 · Spoken-number → digit normalization](lessons/24_spoken_numbers/README.md)** — PyPI survey (`text2num` for 7 EU langs, none for Turkish, why tokenizers aren't the right tool), a rule-based Turkish parser, fuzzy partial-matching with 3-tier escalation (accept / confirm / reject), the `parse_spoken_number` `@tool` that wraps `text2num` multilingual + our parser, and the honest answer to "does wrapping it as a tool cause hallucination?"
- [ ] **[25 · Tool design patterns](lessons/25_tool_design/README.md)** — consolidates tool wisdom from lessons 05/10/19/23/24. Five canonical shapes (read-only / computation+metadata / side-effect+HITL / router / wrapper), the rich-return-dict principle, recoverable vs unrecoverable errors, the *"return enough metadata that the agent doesn't need the whole conversation back"* rule, two real pitfalls (`Literal` framework rejects, generic wrappers lose schema), 10 anti-patterns + a tool catalogue. Five runnable demos, no API key

### Tier 6 · Capstones

- [ ] **[research_assistant](projects/research_assistant/README.md)** — supervisor + tools + LCEL writeup
- [ ] **[customer_support_bot](projects/customer_support_bot/README.md)** — `create_agent` + middleware + HITL + persistence
- [ ] **[rag_qa_api](projects/rag_qa_api/README.md)** — FastAPI + LangGraph + Postgres + Docker

---

## Folder map

```
LanggraphPlayground/
├── lessons/    # Numbered tutorials — one concept per folder
├── projects/   # Bigger capstones combining many concepts
├── shared/     # Helpers every lesson imports (LLM factory, settings, printers)
├── data/       # Sample docs for RAG lessons
├── docs/       # Standalone HTML study guide + architecture diagrams
├── skills/     # Five Claude Code skills derived from this repo's patterns
├── tests/      # Pytest example showing how to unit-test a StateGraph
└── pyproject.toml
```

Per-folder summary:

- **`lessons/`** — each subfolder is one lesson. Always contains `README.md` + `example.py`, optionally `exercise.py`.
- **`projects/`** — bigger, multi-file capstones. They combine concepts from several lessons.
- **`shared/`** — `settings.py` (typed env), `pretty.py` (rich printers), and `llm/` (provider adapters).
- **`data/`** — small sample documents used by the RAG lessons. Chroma indexes are written next to them and gitignored.
- **`docs/`** — single-file HTML reports (`curriculum.html`, `architecture.html`). Self-contained — double-click to open. See [`docs/README.md`](docs/README.md) for how to add more.
- **`skills/`** — five Claude Code skills (`python-design-patterns-applied`, `python-clean-code`, `fastapi-pytest-functional`, `langgraph-1x-engineering`, `langchain-1x-engineering`). Install with `cp -r skills ~/.claude/skills/langgraph-playground` — see [`skills/README.md`](skills/README.md).
- **`tests/`** — sample pytest test for lesson 08 showing the testing pattern.

---

## Setup in three commands

```bash
uv sync                                        # installs everything
cp .env.example .env                           # then fill in ANTHROPIC_API_KEY
uv run python -m lessons.00_setup.example      # smoke test
```

> Don't have `uv`? Install it from <https://docs.astral.sh/uv/>. It's the
> 2026 Python package manager — fast, lockfile-based, drop-in for pip.

The capstones that need extras:

```bash
uv sync --extra api    # for projects/rag_qa_api
uv sync --extra dev    # for pytest
```

---

## Switching LLM providers

The whole curriculum runs on either Anthropic or OpenAI. Flip one env var:

```bash
# in .env
LLM_PROVIDER=anthropic    # default
# LLM_PROVIDER=openai
```

No code changes. Every lesson calls `get_llm()` from `shared/llm/`,
which reads `LLM_PROVIDER` and returns the right `ChatModel`. Adding a
new provider (Gemini, Ollama, …) is one new adapter file — see
[shared/llm/README.md](shared/llm/README.md).

---

## How to debug

Whenever a lesson does something unexpected:

```bash
PYTHONBREAKPOINT=ipdb.set_trace uv run python -m lessons.NN_topic.example
```

Any `breakpoint()` call drops you into [ipdb](https://github.com/gotcha/ipdb)
with colored tracebacks and tab completion. Lesson 03 teaches the full
workflow — including how to debug async code and how to peek inside
LCEL chains and LangGraph nodes.

---

## How each lesson is laid out

Every `lessons/NN_topic/` folder follows the same shape so you always
know where to look:

```
lessons/NN_topic/
├── README.md      ← What you'll learn, why, how to run, what to debug
├── example.py     ← Runnable script, ~30–100 lines, heavily commented
└── exercise.py    ← (optional) "your turn" with hints
```

Every lesson README ends with:

- **Run it** — the exact command
- **Debug it** — a recommended `breakpoint()` location
- **Try it yourself** — a small extension
- **Next →** — link to the next lesson

---

## Glossary

| Term | One-line meaning |
|---|---|
| **Runnable** | LangChain's base unit — anything with `.invoke / .stream / .batch / .ainvoke`. |
| **LCEL** | LangChain Expression Language — composing Runnables with the `\|` pipe. |
| **StateGraph** | LangGraph's stateful workflow primitive. |
| **MessagesState** | A built-in state type that holds a `messages` list. |
| **Node** | A function `(state) → partial_state` that runs at a graph step. |
| **Edge** | A directed transition between nodes (static or conditional). |
| **ToolNode** | A prebuilt node that executes any `tool_calls` from the last AI message. |
| **Checkpointer** | The persistence layer — saves graph state after every step. |
| **Thread** | A conversation id (`thread_id`) used to scope checkpoints. |
| **Reducer** | A function that decides how new state values merge into existing ones. |
| **Interrupt** | A pause point that yields control to a human, then resumes. |
| **Middleware** | Composable hooks around `create_agent` — `before_model`, `wrap_tool_call`, etc. |
| **Supervisor** | A boss agent that routes work to specialized worker agents. |
| **Swarm** | A flat group of agents that hand off to each other directly. |
| **Store** | LangGraph's long-term memory API — survives across threads. |

---

## References

- LangChain docs — <https://docs.langchain.com/oss/python/>
- LangGraph docs — <https://docs.langchain.com/oss/python/langgraph/>
- LangGraph repo — <https://github.com/langchain-ai/langgraph>
- `langgraph-supervisor` — <https://reference.langchain.com/python/langgraph-supervisor>
- `langgraph-swarm` — <https://reference.langchain.com/python/langgraph-swarm>
- "Building LangGraph: an agent runtime from first principles" — <https://blog.langchain.com/building-langgraph/>

---

## License

MIT. Use it, fork it, teach with it.
