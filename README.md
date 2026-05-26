# LangGraph Playground

A **zero-to-hero AI-engineer curriculum** built around **LangChain
1.x** and **LangGraph 1.x**. Thirty-three lessons walk you from your
first `llm.invoke("hi")` call to a multi-replica, distributed-locked,
Jinja-templated, hybrid-search RAG service. Six capstone projects
(three core + three Tier 6 advanced) pull it all together into apps
you'd actually ship.

By the time you finish, you'll have built:

- A **multi-agent research assistant** (supervisor + researcher + writer + critic), and a **`_pro`** version with parallel fan-out, Jinja prompts, semaphore rate-limits, circuit breakers
- A **customer-support bot** with HITL escalation and persistent memory, and a **`_pro`** version with per-customer locks, idempotent tools, tenant-aware prompts
- A **RAG Q&A API** in FastAPI on Postgres + Docker, and a **`_pro`** version with pgvector OR Qdrant, Redis-coordinated index rebuilds, query rewriting and grading

Plus a **sibling track** ([`ml_foundations/`](ml_foundations/README.md))
that takes you under the hood: train your own tokenizer, embeddings,
intent classifier, and contrastively-finetuned encoder. Same repo,
opt-in dep group, clean conceptual separation.

> 📖 **Prefer to browse visually first?** Open [`docs/curriculum.html`](docs/curriculum.html) (double-click — single-file, no server) for the curriculum in a navigable HTML study guide. The companion [`docs/architecture.html`](docs/architecture.html) has the module map and capstone diagrams.

---

## Why this exists

LangChain and LangGraph cover overlapping territory. The rule of thumb
worth memorizing on day one:

> **Start with LCEL.** When you hit a wall — looping, branching on agent
> decisions, or carrying state between steps — switch to LangGraph.

LangChain is optimized for *building* agents quickly. LangGraph is
optimized for *running* them in production: durability, memory,
human-in-the-loop, and control. Tiers 1-5 mirror that arc: simple
thing first, then the more powerful thing the moment the simple thing
creaks.

**Tier 6** is where most curricula stop and most production teams
start hurting. Concurrency races between replicas. Prompts that grew
into 200-line `if/else` chains. Vector indexes that two pods rebuild
at the same time. "Why is this prompt change making things worse and
we have no way to tell?" The lessons in Tier 6 (27-32) and the
**`_pro`** capstones cover the *boring* parts that turn a
prototype-grade agent into a system you can wake up to. Locks,
fencing tokens, Jinja-templated prompt registries, eval-driven
promotion, distributed coordination, pgvector vs Qdrant tradeoffs.

**`ml_foundations/`** answers the next layer of questions — what's
inside the model you're calling. Train your own tokenizer; train
Word2Vec; fine-tune DistilBERT for classification; contrastively
fine-tune an encoder for retrieval. Same repo, opt-in dep group,
separate conceptual track so the LangGraph promise stays clean.

---

## The stack

Core:

| Library | Version | Why it's here |
|---|---|---|
| `langchain` | 1.3.x | The new `create_agent` API + middleware |
| `langgraph` | 1.2.x | `StateGraph`, checkpointers, `interrupt` |
| `langgraph-supervisor` | 0.1.x | One-line supervisor agent |
| `langgraph-swarm` | 0.1.x | Peer-to-peer handoffs |
| `langchain-chroma` + `fastembed` | latest | Local vector store, zero extra API key |
| `pydantic` 2 + `pydantic-settings` | 2.x | Structured output, typed env |
| `ipdb` | 0.13.x | Interactive debugging (lesson 03) |
| `jinja2` | 3.1.x | Dynamic prompting (lesson 28) + capstone templates |
| `tenacity` | 9.x | Retries with backoff (lesson 30) |
| `redis` | 5.x | Distributed locks (lesson 31), pro RAG capstone |

`api` extra (FastAPI capstones):

| Library | Why it's here |
|---|---|
| `fastapi` + `uvicorn` | Capstone API surface |
| `langgraph-checkpoint-postgres` | Postgres-backed checkpointer |
| `langchain-postgres` | pgvector retriever (lesson 29, rag_qa_api_pro) |
| `langchain-qdrant` + `qdrant-client` | Qdrant retriever (lesson 29, rag_qa_api_pro) |

`ml` extra ([`ml_foundations/`](ml_foundations/README.md)):

| Library | Why it's here |
|---|---|
| `tokenizers` + `sentencepiece` + `tiktoken` | Lesson 01 — tokenizer training and comparison |
| `gensim` | Lesson 02 — Word2Vec / FastText |
| `torch` + `transformers` + `datasets` | Lesson 03 — DistilBERT fine-tune |
| `sentence-transformers` | Lesson 04 — contrastive encoder fine-tune |

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
- [ ] **[26 · Miscellaneous](lessons/26_misc/README.md)** — the Tier 5 closer. Token counting (tiktoken), cost estimation (per-provider × per-workload matrix), caching (LangChain `InMemoryCache` for 1000× speedup + Anthropic native prompt caching), the eval framework you wish you'd built sooner, self-correction loop, MCP (`langchain-mcp-adapters`), reranking (Cohere + cross-encoders). Five runnable demos. Plus an honest "what we deliberately don't cover" section with pointers (local LLMs, multimodal, GraphRAG, etc.)
- [ ] **[27 · Locks & concurrency](lessons/27_locks_and_concurrency/README.md)** — `asyncio.Lock`, semaphore-bounded fan-out, per-key lock maps, idempotency keys, the four patterns every production agent needs
- [ ] **[28 · Dynamic prompting with Jinja2](lessons/28_dynamic_prompting/README.md)** — `template_format="jinja2"`, file-based templates, inheritance + `{% extends %}`, prompt as callable, per-turn rendering, sandboxing
- [ ] **[29 · Vector databases (deep dive)](lessons/29_vector_databases/README.md)** — pgvector vs Qdrant side-by-side, Docker Compose for both, hybrid search (dense + BM25), tenant filtering, migration playbook, the 2026 vendor comparison

### Tier 6 · Advanced (production hardening — deep dive)

- [ ] **[30 · Advanced graph patterns](lessons/30_advanced_graphs/README.md)** — parallel fan-out with `Send`, map-reduce in LangGraph, dynamic subgraph spawning, bounded cycles, retry with tenacity, circuit breakers, streaming joins
- [ ] **[31 · Distributed locks](lessons/31_distributed_locks/README.md)** — Redis SETNX + Lua release, fencing tokens (the Kleppmann fix), Postgres advisory locks, Redlock debate, the lease-expiry problem
- [ ] **[32 · Prompt engineering lab](lessons/32_prompt_engineering_lab/README.md)** — versioned registry, sticky-per-user A/B routing, eval-driven promotion in CI, hot reload with locks, sandboxing user-supplied templates
- [ ] **[33 · Vector database internals](lessons/33_vector_database_internals/README.md)** — the under-the-hood companion to lesson 29. ANN algorithms (Flat, LSH, IVF, IVF-PQ, OPQ, HNSW, ScaNN, DiskANN, SPANN), per-vendor algorithmic map (FAISS, Annoy, Qdrant, Weaviate, Milvus, Pinecone, pgvector, LanceDB, Vespa, ScaNN), and a FAISS benchmark that prints latency × recall × memory across four index types on the same data

### Tier 7 · Capstones

Two flavours per capstone — the **simple** version (Tier 1-5 concepts) is the
right place to start; the **`_pro`** version layers in Tier 6 production
patterns (locks, Jinja prompts, complex graphs, distributed coordination).

- [ ] **[research_assistant](projects/research_assistant/README.md)** — supervisor + tools + LCEL writeup
- [ ] **[research_assistant_pro](projects/research_assistant_pro/README.md)** — parallel fan-out researchers, Jinja prompts, semaphore + per-topic lock map, Tavily circuit breaker, bounded critic cycle
- [ ] **[customer_support_bot](projects/customer_support_bot/README.md)** — `create_agent` + middleware + HITL + persistence
- [ ] **[customer_support_bot_pro](projects/customer_support_bot_pro/README.md)** — per-customer lock map, idempotency-keyed refund tool, Jinja persona by tier/locale, Sqlite/Postgres backend toggle
- [ ] **[rag_qa_api](projects/rag_qa_api/README.md)** — FastAPI + LangGraph + Postgres + Docker
- [ ] **[rag_qa_api_pro](projects/rag_qa_api_pro/README.md)** — rewrite → retrieve → grade → generate → cite, pgvector/Qdrant toggle, Redis-coordinated index rebuild, API-key auth, tenant filtering

### Sibling track — [`ml_foundations/`](ml_foundations/README.md)

Not numbered alongside `lessons/` because it teaches a different muscle
(PyTorch + Hugging Face, not LangChain/LangGraph). Same repo, separate
opt-in dep group (`uv sync --extra ml`):

- **[00 · Overview](ml_foundations/00_overview/README.md)** — the mental map: tokeniser → embed → encoder → decoder
- **[01 · Tokenizers from scratch](ml_foundations/01_tokenizers/README.md)** — train BPE + SentencePiece, then compare against GPT-4o on the same text
- **[02 · Word embeddings](ml_foundations/02_word_embeddings/README.md)** — Word2Vec Skip-gram, why these still matter
- **[03 · Transformer architecture](ml_foundations/03_transformer_architecture/README.md)** — implement self-attention by hand; encoder vs decoder vs encoder-decoder; RoPE, flash-attention, GQA, MoE
- **[04 · Text classification](ml_foundations/04_text_classification/README.md)** — fine-tune DistilBERT; when fine-tuned encoders beat LLM prompts
- **[05 · Fine-tuning encoders](ml_foundations/05_finetuning_encoders/README.md)** — contrastive fine-tune `all-MiniLM` for retrieval; this is where `bge-small-en-v1.5` comes from

A future `gnn/` track will sit next to this one for the same reason.

---

## Folder map

```
LanggraphPlayground/
├── lessons/          # Numbered tutorials (00–32) — one concept per folder
├── projects/         # Capstones (simple + _pro variants)
├── ml_foundations/   # Sibling track: tokenizers, embeddings, classifiers, encoder fine-tune
├── shared/           # Helpers every lesson imports (LLM factory, settings, printers)
├── data/             # Sample docs, Chroma indexes, trained models (gitignored)
├── docs/             # Standalone HTML study guide + architecture diagrams
├── skills/           # Claude Code skills derived from this repo's patterns
├── tests/            # Pytest example showing how to unit-test a StateGraph
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

The capstones and the ml track each have their own extras:

```bash
uv sync --extra api    # for projects/rag_qa_api + rag_qa_api_pro (FastAPI, pgvector, qdrant)
uv sync --extra ml     # for ml_foundations/ (torch, transformers, tokenizers, gensim)
uv sync --extra dev    # for pytest
```

The Tier 6 lessons (27, 31) and `_pro` capstones also assume Redis +
Postgres + Qdrant running locally. The lesson 29 `docker-compose.yml`
brings all three up with one command:

```bash
cd lessons/29_vector_databases
docker compose up -d           # pgvector + qdrant + redis
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
| **Send** | LangGraph primitive that fans out into N parallel branches from one node. |
| **Reducer** | An `Annotated[T, fn]` annotation that merges concurrent writes to the same state key. |
| **Semaphore** | `asyncio.Semaphore(n)` — caps concurrent operations at `n`. The cure for "I just DOS'd my LLM provider." |
| **Per-key lock map** | `dict[key, asyncio.Lock]` — serial within a key, parallel across keys. Right shape for chat servers. |
| **Distributed lock** | A lock whose state lives in shared storage (Redis, Postgres). Coordinates across replicas. |
| **Fencing token** | A monotonically-increasing token paired with a lock so the storage layer rejects stale writes. |
| **Idempotency key** | A client-supplied id so retries of a side-effect tool return the same result. |
| **Circuit breaker** | Counts recent failures; when above threshold, fails fast for a cooldown window. |
| **Jinja template** | Versioned, inheritable prompt artefact. Pass `template_format="jinja2"` to `PromptTemplate`. |
| **Prompt registry** | A `name → version → artefact` lookup. The shape LangSmith Hub generalises. |
| **Hybrid search** | Vector + BM25 fusion (typically RRF). Beats either alone on keyword-heavy queries. |
| **pgvector / Qdrant** | The two boring + the two purpose-built defaults for production vector storage. |

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
