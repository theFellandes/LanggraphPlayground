# Lesson 26 · The miscellaneous lesson

Topics that are too important to skip but didn't deserve a whole
dedicated lesson. Five high-severity gaps surfaced in a curriculum
audit, all addressed here in compact form. Plus an honest list at the
end of what's **deliberately not covered** and where to go for each.

## What you'll learn

| # | Topic | Demo runnable? | Why it's here |
|---|---|---|---|
| 1 | **Token counting** with `tiktoken` (and the Anthropic alternative) | ✅ | Without counting tokens, you can't budget; you can't budget, you can't ship |
| 2 | **Cost estimation** — turn token counts into dollars | ✅ | The number every engineering manager asks before approving the demo |
| 3 | **Caching** — LangChain's `InMemoryCache` / `SQLiteCache` and Anthropic's `AnthropicPromptCachingMiddleware` | ✅ (2300× speedup on the demo) | Biggest single cost-and-latency lever for chat workloads |
| 4 | **Evaluation framework** — a working pytest-style eval suite in 40 lines | ✅ | The single highest-leverage missing piece — without evals you can't iterate safely |
| 5 | **Self-correction** (Reflexion-lite retry loop) | ✅ | The manual version of what `Instructor` / `Marvin` automate |
| 6 | **MCP** (Model Context Protocol) — `langchain-mcp-adapters` | README only | Anthropic-led standard for tool-server interop; 2026's hot integration story |
| 7 | **Reranking** for RAG — Cohere Rerank + cross-encoders (BGE) | README only | Major RAG-quality lever; needs external API or a downloaded model |

The first five run end-to-end without an API key. MCP and reranking
need external services (an MCP server / a Cohere key / a downloaded
cross-encoder) so they're code-in-README rather than runnable demos.

## Why a miscellaneous lesson exists

Building real LLM apps surfaces dozens of "I need this *now*" needs:
*"how many tokens is this prompt?"*, *"can I cache that retrieval
call?"*, *"how do I know if my prompt change made things better?"*.
Each is too small for a 20-page lesson but too important to leave to
"figure it out yourself." This lesson is the bag of small,
high-leverage things — each gets a paragraph, a code snippet, and a
pointer to where to go deeper if you need it.

---

## Topic 1 · Token counting (`tiktoken`)

The OpenAI tokenizer ships as `tiktoken`. Same encoder works for GPT-4o
and most modern OpenAI models. **It is fast and offline** — no API
call needed.

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
n_tokens = len(enc.encode("How many tokens is this?"))   # → 6
```

**Anthropic models** use a different tokenizer; for exact counts:

```python
from anthropic import Anthropic
client = Anthropic()
result = client.beta.messages.count_tokens(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "..."}],
)
print(result.input_tokens)
```

**Approximate rules of thumb** (from demo 1):

| Content | Chars/token |
|---|---|
| English prose | ~4 |
| Code | ~3-4 |
| Turkish / non-Latin scripts | ~2-2.5 (multi-byte chars cost more) |
| JSON / heavily-punctuated text | ~3 |

Use this for budget planning *before* the call. Doing it after is too
late — the bill is already incurred.

## Topic 2 · Cost estimation

Once you have token counts, dollars is multiplication:

```python
PRICING = {   # USD per 1M tokens, (input, output) — verify on the vendor page
    "claude-sonnet-4-6":  (3.00, 15.00),
    "claude-haiku-4-5":   (0.80,  4.00),
    "gpt-4.1":            (5.00, 15.00),
    "gpt-4o-mini":        (0.15,  0.60),
}

def cost_usd(model, input_tokens, output_tokens):
    p_in, p_out = PRICING[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000
```

The demo prints a workload × model matrix that makes a single
realisation jump out: **at 1 million users, "agent w/ 3 tool turns"
costs $2,100 on `gpt-4o-mini` vs $62,500 on `gpt-4.1`** — same task,
30× cost spread. This is the routing-tier-by-cost decision from lesson
22's architecture map.

In production, integrate this into the **observability layer** so
every request emits a `cost_usd` field. Stack on top of LangSmith
(which already exposes token counts per run) and your CFO will love
you.

## Topic 3 · Caching

Two distinct layers, both worth knowing:

### A · LangChain global cache (any provider)

Caches the whole `(model, prompt) → response` round-trip. Set once,
applies to every model call in the process.

```python
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

set_llm_cache(InMemoryCache())            # dev — in-process, lost on restart
# OR
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".llm_cache.db"))   # local persistence
```

Demo 3 shows a 2,355× speedup (507 ms → 0.2 ms) on a fake 500 ms model
— in production with real model latency, expect 100-1000×.

**For chat workloads** where prompts are similar-but-not-identical, also
consider **semantic caching** — embed each prompt, look up by cosine
similarity, return the cached response if a near-duplicate is found.
`langchain-community` ships `RedisSemanticCache` / `GPTCache`
integrations for this.

### B · Anthropic native prompt caching

Anthropic charges 10% of normal input price for cached prefix tokens
and 1.25× write cost the first time you cache. For a system prompt or
RAG context reused across many calls, this is huge. LangChain wires it
in automatically:

```python
from langchain.agents import create_agent
from langchain_anthropic.middleware.prompt_caching import AnthropicPromptCachingMiddleware
from shared.llm import get_llm

agent = create_agent(
    model=get_llm(),
    tools=[...],
    middleware=[AnthropicPromptCachingMiddleware()],   # tags system + tools for cache
)
```

The middleware tags the last content block of the system message + all
tool definitions with `cache_control`. On the next request with the
same prefix, Anthropic skips the prefix processing entirely.

**OpenAI prompt caching is automatic** — they cache server-side
without you opting in, as long as your prompt prefix is ≥ 1024 tokens
and identical. You don't directly control it; you just see a
`cached_tokens` field on the usage response.

## Topic 4 · Evaluation framework

The single highest-leverage missing piece in most LLM codebases. An
eval is just **`{input, expected, scorer}`** — run the system, score
each output, aggregate. Demo 4 ships a working version in 40 lines.

```python
EvalCase = tuple[str, str, str]   # (label, input, expected)

def run_eval_suite(sut, cases, scorer):
    failures = []
    for label, inp, expected in cases:
        actual = sut(inp)
        if not scorer(actual, expected):
            failures.append({"label": label, "expected": expected, "got": actual})
    return {"passed": len(cases) - len(failures), "total": len(cases), "failures": failures}
```

Three scorer flavours:

| Scorer | When |
|---|---|
| **`exact_match`** | Deterministic outputs — extraction tasks (`"1987-04-05"`), classifications |
| **`contains_match`** | "Did the answer mention X?" |
| **`schema_match`** | JSON outputs — check the required keys are present |
| **`semantic_match`** (LLM-as-judge) | Free-text outputs — ask a separate (cheaper) LLM to judge |

**Operational guidance** (the part most teams skip):

1. **Build the eval set BEFORE iterating on prompts.** Without it, every change "feels better" but you can't tell when you've regressed.
2. **30-50 cases per task** is the right starting point. Too few = noisy. Too many = slow.
3. **Run in CI on every prompt change.** Track pass-rate over time. A drop ≥ 5% blocks the merge.
4. **Pin the scorer model.** If you use LLM-as-judge, the judge model is itself a moving target; lock it.
5. **Include "should refuse" cases.** "What's the user's password?" → expected: a refusal. Otherwise you only test happy-path behaviour.

**Productionised frameworks** (don't roll your own past the prototype
stage):

- **[LangSmith evals](https://docs.smith.langchain.com/evaluation)** — first-party LangChain; integrates with traces; UI for failure inspection
- **[promptfoo](https://www.promptfoo.dev/)** — declarative YAML, CLI, supports CI well
- **[inspect-ai](https://inspect.ai-safety-institute.org.uk/)** — UK AI Safety Institute's framework; great for safety evals
- **[deepeval](https://docs.confident-ai.com/)** — pytest-native with built-in metrics
- **[RAGAS](https://docs.ragas.io/)** — specifically for RAG (faithfulness, answer relevance, context precision/recall)

## Topic 5 · Self-correction (Reflexion-lite)

When a tool / model returns an invalid result, the manual fix is:
**re-prompt with the previous output AND the validator's error
message**. Demo 5 ships a 20-line implementation.

```python
def self_correcting_call(generator, validator, initial_prompt, max_attempts=3):
    prompt = initial_prompt
    for attempt in range(1, max_attempts + 1):
        output = generator(prompt)
        ok, error = validator(output)
        if ok: return output, attempt
        prompt = (f"{initial_prompt}\n\n"
                  f"Previous attempt: {output!r}\nError: {error}\n"
                  f"Try again, fixing the error.")
    return None, max_attempts
```

This is the **manual** version of what:
- **Instructor** (lesson 24's references) does automatically via Pydantic ValidationError
- **lesson 21's four-layer pipeline** does for dates
- **lesson 11's middleware** can wrap around any tool call

For production: cap at **3 attempts** (more = diminishing returns +
escalating cost), log every retry for monitoring, **graduate to
Instructor** which automates the loop with proper error formatting.

## Topic 6 · MCP (Model Context Protocol)

[Anthropic's MCP](https://modelcontextprotocol.io/) is the 2026
standard for **how tool servers expose tools to LLM agents**. Instead
of writing each integration custom, you connect to any MCP server
(filesystem, GitHub, Slack, your own) and get a uniform tool surface.

**LangChain integration** ships as `langchain-mcp-adapters`:

```python
# pip install langchain-mcp-adapters
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from shared.llm import get_llm

# Connect to one or more MCP servers
mcp_client = MultiServerMCPClient({
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "transport": "stdio",
    },
    "github": {
        "url": "http://localhost:3000/mcp",  # if you run a remote MCP server
        "transport": "streamable_http",
    },
})

# Pull the MCP-exposed tools as LangChain tools
tools = await mcp_client.get_tools()

# Use them with create_agent as if they were native @tool functions
agent = create_agent(model=get_llm(), tools=tools)
```

**Transports supported:** `stdio` (subprocess), `sse` (Server-Sent
Events), `streamable_http`, `websocket`.

**Why it matters now:**

- Cuts integration work — one MCP server can serve OpenAI, Anthropic, and Google clients without rewrites
- Standard registry of public servers ([modelcontextprotocol.io/servers](https://modelcontextprotocol.io/servers)) for common tools (filesystem, Git, GitHub, Slack, Postgres, etc.)
- All major IDE assistants (Claude Code, Cursor, Windsurf, Zed) speak it

**This lesson stops at the integration shape.** Building your own
MCP server is its own substantial topic — start from the [Python MCP
SDK](https://github.com/modelcontextprotocol/python-sdk) for that.

## Topic 7 · Reranking for RAG

Vector search (lesson 06) returns the top-k by embedding similarity.
A **reranker** then re-orders those k candidates using a much more
expensive but more accurate model — typically a *cross-encoder* that
reads `(query, document)` together rather than embedding them separately.

The pattern: **retrieve N candidates (cheap), rerank to top K (expensive)**.

### Option A — Cohere Rerank (managed API)

```python
# pip install langchain-cohere
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
reranker = CohereRerank(model="rerank-v3.5", top_n=5)
final_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever,
)
final_retriever.invoke("How does X work?")   # 20 candidates → 5 reranked
```

**Cost:** ~$1 per 1k searches at Cohere's current price. Often pays
for itself in answer-quality improvement.

### Option B — Local cross-encoder (free, slower)

```python
# pip install sentence-transformers
from sentence_transformers import CrossEncoder
ce = CrossEncoder("BAAI/bge-reranker-v2-m3")
scores = ce.predict([(query, doc) for doc in candidates])
top_k = [c for _, c in sorted(zip(scores, candidates), reverse=True)[:5]]
```

**Free** but loads ~1 GB of model weights. Good for offline/private
workloads.

### When you need a reranker

- Your RAG answers are *plausible-but-wrong* — the right info IS in the corpus, but the top-k retrieval missed it
- You're at the "vector search alone gets us 70% accuracy, need 90%" stage
- Adding more documents to top-k *without* reranking just adds noise

**When you don't:** if your top-3 results are already correct most of
the time, a reranker is overkill. Add it when you actually have a
measured retrieval-quality problem (see Topic 4 above for the eval
suite to measure it).

---

## What we deliberately don't cover

Honest pointers, not pretending these don't exist:

| Topic | Where to go |
|---|---|
| **Local LLMs (Ollama, llama.cpp, vLLM)** | Different ecosystem; `langchain-ollama`, `langchain-llamacpp` for the LangChain bridge. Worth a separate lesson if you need it. |
| **Multimodal (vision LLMs)** | Lesson 21's deep-dive doc touches Docling VLM + GPT-4o Vision. A dedicated lesson on `ChatAnthropic`/`ChatOpenAI` with image inputs would be useful — left as a stretch goal. |
| **GraphRAG / Knowledge graphs + LLMs** | Niche but powerful for relational data. `langchain-neo4j`, `LightRAG`, Microsoft's GraphRAG project. |
| **Vector store comparison** (Qdrant, Weaviate, Pinecone, pgvector, Milvus, LanceDB) | We use Chroma throughout because it works locally with no setup. For production with shared state, Postgres + `pgvector` is the boring default; Qdrant for higher scale. |
| **Embedding model comparison** | Lesson 06 uses FastEmbed (local). Comparison: OpenAI `text-embedding-3-large`, Cohere `embed-v3.0`, Voyage `voyage-3`, Nomic open weights. MTEB leaderboard is the standard benchmark. |
| **LLM routing (cheap → expensive escalation)** | Touched in lesson 22's architecture decision matrix. Tools like RouteLLM exist for automated routing based on query difficulty. |
| **Agent benchmarks** (SWE-bench, GAIA, etc.) | Research-grade; useful for model comparison but rarely actionable for a single-product team. |
| **Fine-tuning** | `llm-expert` skill covers when to fine-tune vs prompt vs RAG. Implementation: use a vendor (OpenAI fine-tuning API, Anthropic fine-tuning for Haiku) or roll your own with `axolotl` / `unsloth`. |
| **Prompt versioning / management** | LangSmith Hub (`hub.pull(...)`); for self-hosted, store prompts in a Git-tracked directory with semantic versioning. |

If any of these become load-bearing for what you're building, that's a
signal to invest in them properly — not to learn them from a
6-paragraph summary in a misc lesson.

## Run it

```bash
uv run python -m lessons.26_misc.example                   # all five demos
uv run python -m lessons.26_misc.example --tokens
uv run python -m lessons.26_misc.example --cost
uv run python -m lessons.26_misc.example --caching
uv run python -m lessons.26_misc.example --eval
uv run python -m lessons.26_misc.example --self-correct
```

Sample output (caching demo):

```
miss:   507.3 ms  →  'The answer is 42.'
hit:      0.2 ms  →  'The answer is 42.'

speedup ≈ 2355×  (cache returns response without re-invoking)
```

## Anti-patterns

| Smell | Fix |
|---|---|
| Counting tokens with `len(text.split())` | `len(tiktoken.encoding_for_model(model).encode(text))` — splits aren't tokens |
| Quoting cost without per-million math | Engineering managers want $X/month at Y users, not $0.0001/call |
| Caching everything indiscriminately (incl. random sampling) | Cache deterministic calls; explicitly skip cache for any call with `temperature > 0` you want random |
| Adding a reranker before measuring retrieval quality | First build an eval set (Topic 4); then decide if a reranker actually helps |
| "Production has evals" actually means "we eyeball the demo before shipping" | A real eval suite is in CI, gates merges, has dashboards |
| Self-correction with no max-attempts cap | Always cap; otherwise a confidently-wrong model can burn unlimited tokens |

## Pairs with

- **[Lesson 22 · Architecture](../22_architecture/README.md)** — Topic 2's cost discussion ties into the Reliability + Observability layers; Topic 4's eval framework lives in the Observability layer
- **[Lesson 11 · Agent middleware](../11_agent_middleware/README.md)** — `ModelCallLimitMiddleware` and `SummarizationMiddleware` are the runtime versions of what Topics 2-3 talk about
- **[Lesson 25 · Tool design](../25_tool_design/README.md)** — Topic 5 (self-correction) is the manual version of the recoverable-error contract from lesson 25
- **[`skills/llm-expert`](../../skills/llm-expert/SKILL.md)** — covers fine-tune vs RAG vs prompt, model selection, the deep technical layer this lesson sits on top of

## References

- [tiktoken on PyPI](https://pypi.org/project/tiktoken/) — OpenAI tokenizer
- [Anthropic token counting API](https://docs.anthropic.com/en/api/messages-count-tokens) — exact counts for Claude models
- [Anthropic prompt caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — vendor-side caching
- [LangChain LLM caching guide](https://python.langchain.com/docs/how_to/llm_caching/) — `InMemoryCache`, `SQLiteCache`, etc.
- [`AnthropicPromptCachingMiddleware` reference](https://reference.langchain.com/python/langchain-anthropic/middleware/prompt_caching/AnthropicPromptCachingMiddleware)
- [LangSmith evaluation docs](https://docs.smith.langchain.com/evaluation) — first-party eval framework
- [promptfoo](https://www.promptfoo.dev/) — YAML/CLI-driven eval framework
- [RAGAS](https://docs.ragas.io/) — RAG-specific metrics
- [Model Context Protocol](https://modelcontextprotocol.io/) — the spec
- [`langchain-mcp-adapters` on PyPI](https://pypi.org/project/langchain-mcp-adapters/)
- [`langchain-mcp-adapters` GitHub](https://github.com/langchain-ai/langchain-mcp-adapters)
- [Cohere Rerank docs](https://docs.cohere.com/docs/rerank-on-langchain)
- [BGE Reranker on Hugging Face](https://huggingface.co/BAAI/bge-reranker-v2-m3) — open-weights cross-encoder
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) — the academic paper behind Topic 5's pattern

## Next →

That's the curriculum, honestly. From here: pick a [capstone](../../projects/) and ship.
