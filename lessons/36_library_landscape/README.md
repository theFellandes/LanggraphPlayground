# Lesson 36 · The AI engineering library landscape

You've been deep on LangChain + LangGraph for 35 lessons. **Real AI
engineers know what surrounds them.** In an interview, "why LangGraph
and not CrewAI?" or "have you tried Instructor?" or "what would you
use DSPy for?" are not trick questions — they test whether you've
*looked at the field* or just learned one stack.

This is the lesson that catches you up. ~15 libraries you should know
*about* (not necessarily master), each with a code recipe and a
one-sentence "use it when X."

## What you'll learn

A working map of the 2026 Python AI engineering stack outside the
LangChain/LangGraph world, structured by what they replace or
complement:

| Category | Libraries |
|---|---|
| **Structured output** | Instructor, Outlines, Marvin, Pydantic AI |
| **Declarative prompting** | DSPy |
| **Alternative frameworks** | LlamaIndex, Haystack |
| **Alternative agent frameworks** | CrewAI, AutoGen, smolagents, OpenAI Agents SDK |
| **Memory backends** | Mem0, Letta (MemGPT), Zep |
| **Web ingestion** | Firecrawl, Jina Reader, Crawl4AI |
| **Routing + cost** | semantic-router, RouteLLM, LiteLLM |
| **Inference servers** | vLLM, TGI, SGLang, Ollama |
| **Hosted inference** | Modal, Replicate, Banana, Together |

By the end you'll be able to choose between LangGraph + a library vs
the library standalone, with reasoning beyond "I know LangGraph."

---

## Structured output

### Instructor — Pydantic + retries, the standard

```python
# pip install instructor
import instructor
from anthropic import Anthropic
from pydantic import BaseModel

class Support(BaseModel):
    category: str
    severity: int            # 1-5
    needs_human: bool

client = instructor.from_anthropic(Anthropic())
result = client.messages.create(
    model="claude-sonnet-4-6",
    response_model=Support,           # ← the magic
    max_tokens=512,
    messages=[{"role": "user", "content": "I want my $250 refunded."}],
)
print(result)   # Support(category='refund', severity=4, needs_human=True)
```

Wraps any LLM client to enforce a Pydantic schema. **On validation
failure, automatically retries with the error message.** Lesson 26's
self-correction loop is what Instructor does for you.

**Use when**: any time you'd otherwise call `.with_structured_output`
on a vendor client directly. Better retries; better error messages;
provider-agnostic.

### Outlines — constrained generation via grammars

```python
# pip install outlines
import outlines
from pydantic import BaseModel

class Reply(BaseModel):
    category: str
    confidence: float

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = outlines.generate.json(model, Reply)
result = generator("Classify: 'My order is broken.'")
```

The trick: Outlines modifies the **sampling step** — at each token,
only allow tokens that keep the partial output a valid prefix for the
target grammar. **Invalid JSON is mathematically impossible.**

**Use when**: you control the model (local / vLLM), and want 100%
schema compliance instead of "retry until valid." Doesn't work with
closed APIs.

### Marvin — declarative AI functions

```python
# pip install marvin
import marvin

@marvin.fn
def summarise(text: str, sentences: int = 2) -> str:
    """Summarise the input in N sentences."""

summarise("LangGraph models stateful workflows as graphs...", sentences=1)
```

Function signature → prompt. Docstring → instructions. Return type →
parsing. Lightest weight of the structured-output libraries.

**Use when**: you want LLM functions to *look like* regular Python
functions. Notebooks. Demos. Scripts.

### Pydantic AI — Pydantic's own framework

```python
# pip install pydantic-ai
from pydantic_ai import Agent

agent = Agent("anthropic:claude-sonnet-4-6", system_prompt="You are a researcher.")
result = agent.run_sync("What's the capital of Belgium?")
print(result.output)
```

Type-safe agents, structured outputs, tool calling — built by the
Pydantic team. **Native typed everything.** Newer than LangChain, much
smaller surface area, intentionally less "magic."

**Use when**: typed agents over LangChain's broader runtime is your
priority. Or you find LangChain's class hierarchy excessive.

---

## Declarative prompting

### DSPy — Stanford's "compile your prompts"

```python
# pip install dspy
import dspy

dspy.settings.configure(lm=dspy.LM("anthropic/claude-sonnet-4-6"))

class CitedAnswer(dspy.Signature):
    """Answer the question with a single inline citation."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citation: str = dspy.OutputField()

qa = dspy.Predict(CitedAnswer)
print(qa(question="How does Tylenol work?"))
```

The killer feature: **automatic prompt optimization**. Give DSPy a
training set + a metric; it iteratively improves the prompts via
techniques like bootstrap few-shot, MIPRO, etc. The output is a
*compiled* prompt that beats your hand-written one.

**Use when**: you have a measurable task (eval set + scorer) and want
to optimise without manually iterating on prompts. Research-grade
results; production-ready in 2024+.

---

## Alternative frameworks

### LlamaIndex — the other RAG framework

```python
# pip install llama-index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

docs = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(docs)
qe = index.as_query_engine()
print(qe.query("What's the refund policy?"))
```

LlamaIndex was **RAG-first**, agents came later. LangChain is the
opposite. The mental models differ:

- **LangChain**: chains and graphs of Runnables; RAG is one pattern
- **LlamaIndex**: indexes and query engines; abstractions around the retrieval lifecycle

**Use when**: pure RAG, lots of document types (LlamaParse is
excellent), or you find LangChain's RAG abstractions verbose.

### Haystack — search-engine-shaped RAG

```python
# pip install haystack-ai
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator

pipe = Pipeline()
pipe.add_component("retriever", InMemoryBM25Retriever(document_store=ds))
pipe.add_component("generator", OpenAIGenerator())
pipe.connect("retriever.documents", "generator.documents")
```

Component-graph based. Strong on **hybrid search** + **evaluation
loops** + **deployment as a service**. Used heavily in the German /
European NLP ecosystem.

**Use when**: search-engine-style pipelines with hybrid retrieval and
evaluation feedback loops.

---

## Alternative agent frameworks

### CrewAI — role-based multi-agent

```python
# pip install crewai
from crewai import Agent, Crew, Task

researcher = Agent(role="Researcher", goal="Find sources", backstory="...")
writer = Agent(role="Writer", goal="Write a report", backstory="...")

t1 = Task(description="Research fusion energy", agent=researcher)
t2 = Task(description="Write a 200-word report", agent=writer, context=[t1])

crew = Crew(agents=[researcher, writer], tasks=[t1, t2])
result = crew.kickoff()
```

The mental model: **role-playing agents** with backstories,
collaborating through tasks. Lighter than LangGraph; less control over
state; faster to prototype.

**Use when**: you want a multi-agent demo running in 20 lines of code.
Slower / harder to debug as complexity grows.

### Microsoft AutoGen — agent conversations

```python
# pip install autogen-agentchat autogen-ext
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat

researcher = AssistantAgent(name="researcher", model_client=...)
critic = AssistantAgent(name="critic", model_client=...)

team = RoundRobinGroupChat([researcher, critic], max_turns=4)
await team.run(task="Find 3 sources for fusion energy.")
```

Microsoft's framework, originally focused on **conversational
multi-agent**. Strong in research; also has CodeAct (agents that
execute Python).

**Use when**: research / experimentation with agent communication
patterns. Production usage is growing but still less common than
LangGraph.

### smolagents — Hugging Face, code-based agents

```python
# pip install smolagents
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=LiteLLMModel("anthropic/claude-sonnet-4-6"),
)
agent.run("How many people live in Istanbul?")
```

**The unusual choice**: agent actions are Python code, not JSON tool
calls. The model writes `result = search('istanbul population')` and
the runtime executes it in a sandbox. Empirically stronger reasoning
on complex tool sequences.

**Use when**: agentic tasks where the model needs to compose tool
calls (transform data between calls, conditional logic). Sandboxing
matters — by default uses a local Python; use Docker/E2B for safety.

### OpenAI Agents SDK — the official framework

```python
# pip install openai-agents
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    return f"Sunny in {city}"

agent = Agent(name="Helper", instructions="Be helpful.", tools=[get_weather])
result = Runner.run_sync(agent, "What's the weather in Berlin?")
```

OpenAI's own agent framework (released 2025). Direct competitor to
LangChain agents. Smaller surface area, OpenAI-first but model-agnostic
via LiteLLM.

**Use when**: you're OpenAI-aligned and want a leaner alternative to
LangChain.

---

## Memory backends

### Mem0 — universal memory layer

```python
# pip install mem0ai
from mem0 import Memory

m = Memory()
m.add("I'm allergic to peanuts", user_id="alice")
m.add("My favourite drink is matcha", user_id="alice")

results = m.search("dietary restrictions", user_id="alice")
# → [{"memory": "I'm allergic to peanuts", "score": 0.91}]
```

Vector-backed long-term memory, with deduplication + relevance scoring
+ memory updating (vs append-only). Drop-in upgrade for lesson 18's
LangGraph Store.

**Use when**: per-user persistent memory across sessions; the LangGraph
Store works but you want richer semantics.

### Letta (formerly MemGPT) — managed memory tiers

```python
# pip install letta
from letta_client import Letta
client = Letta(token="...")
agent = client.agents.create(model="anthropic/claude-sonnet-4-6", ...)
```

OS-inspired memory hierarchy: short-term (context window) ↔ long-term
(searchable archive) ↔ "core memory" (always-loaded persona facts).
The agent autonomously decides what to promote / archive. Heavier
runtime; more autonomous.

**Use when**: long-running agents with months of state. Letta runs as
a service (self-hosted or hosted).

### Zep — graph-based memory

```python
# pip install zep-cloud
from zep_cloud.client import Zep
zep = Zep(api_key="...")
zep.memory.add(session_id="abc", messages=[...])
```

Knowledge-graph backed memory with temporal awareness (it knows when
facts become stale). Strong on agent personas and dialogue history.

**Use when**: chatbot / assistant where conversation history needs
structured semantics.

---

## Web ingestion (the RAG-input problem)

> **PDF-input sibling:** for the PDF-shaped version of this problem (rendering
> pages to a VLM to get clean Markdown), see the deeper
> [VLM-based PDF→Markdown extraction research](../../docs/research/vlm-pdf-extraction/FINDINGS.md).

### Firecrawl — web → markdown at scale

```python
# pip install firecrawl-py
from firecrawl import FirecrawlApp
app = FirecrawlApp(api_key="fc-...")
result = app.scrape_url("https://anthropic.com")
markdown = result["markdown"]
```

Handles JS rendering, anti-bot, link following. The most reliable
"give me clean text from a URL" service.

**Use when**: RAG pipelines that ingest from public web. Cheaper than
rolling your own Playwright cluster.

### Jina Reader — free, no key

```python
import requests
markdown = requests.get("https://r.jina.ai/https://anthropic.com").text
```

A *single GET*. Free tier; quotas. Brutally simple.

**Use when**: prototyping; low-volume. Switch to Firecrawl when you
hit quota.

### Crawl4AI — open-source, self-host

```python
# pip install crawl4ai
from crawl4ai import AsyncWebCrawler
async with AsyncWebCrawler() as crawler:
    result = await crawler.arun(url="https://anthropic.com")
```

OSS alternative — runs in your infra. Handles dynamic pages with
Playwright under the hood.

**Use when**: data sovereignty / cost control matter more than
convenience.

---

## Routing + cost control

### semantic-router — fast intent routing

```python
# pip install semantic-router
from semantic_router import Route, RouteLayer
from semantic_router.encoders import FastEmbedEncoder

routes = [
    Route(name="refund",   utterances=["I want a refund", "refund my order"]),
    Route(name="policy",   utterances=["how many PTO days", "what's the policy on"]),
]
rl = RouteLayer(encoder=FastEmbedEncoder(), routes=routes)
print(rl("Can I get my money back?"))   # → "refund"
```

Local, sub-millisecond intent classification via embeddings + cosine
threshold. No LLM call — the agent can route before it pays for an
LLM token.

**Use when**: you have a small set of intents and want zero-latency
routing in front of your agent.

### RouteLLM — cheap → expensive cascading

```python
# pip install routellm
from routellm.controller import Controller
client = Controller(
    routers=["mf"],     # matrix factorisation router
    strong_model="anthropic/claude-sonnet-4-6",
    weak_model="anthropic/claude-haiku-4-5",
)
result = client.chat.completions.create(
    model="router-mf-0.11643",   # threshold tuned for 50% cost reduction
    messages=[{"role": "user", "content": "What's 2+2?"}],
)
```

Trained classifier decides per-query which tier of model to use.
Empirically 30-70% cost savings at ~98% of strong-model quality.

**Use when**: production cost is real; you have a quality budget.

### LiteLLM — unified provider interface

```python
# pip install litellm
from litellm import completion
resp = completion(
    model="anthropic/claude-sonnet-4-6",
    messages=[{"role": "user", "content": "hi"}],
)
resp = completion(model="openai/gpt-4o", messages=[...])
resp = completion(model="ollama/llama3", messages=[...])
```

100+ providers behind one OpenAI-shaped API. Adds fallbacks, retries,
caching, cost tracking. The "switch provider with a string change"
layer.

**Use when**: multi-provider + provider-agnostic code paths.
LangChain's `get_llm()` factory is the LangChain-native version; LiteLLM
is the standalone equivalent.

---

## Inference servers (self-hosted)

| Server | Model formats | Strengths |
|---|---|---|
| **vLLM** | HF Transformers | Continuous batching, PagedAttention, the production default |
| **TGI** | HF Transformers | HuggingFace's own; tight HF Hub integration |
| **SGLang** | HF Transformers | Faster on tool-heavy / structured workloads; built-in routing |
| **Ollama** | GGUF (llama.cpp) | Local dev / laptops; one-line install |
| **llama.cpp** | GGUF | CPU/Metal inference; lowest infra footprint |
| **MLC LLM** | Universal (web, mobile, edge) | Inference everywhere |

The local-dev default in 2026 is **Ollama**. The production
self-hosted default is **vLLM**.

---

## Hosted serverless inference

| Service | Model offering | When |
|---|---|---|
| **Together** | Open-weights (Llama, Mistral, DeepSeek) | Cheapest hosted OSS models |
| **Replicate** | Anything you can pack into a container | One-off models, custom packaging |
| **Modal** | Bring your own code; great for batch / cron | When you also have non-LLM Python jobs |
| **Banana** | Serverless GPU | Simple |
| **Fireworks** | Open-weights with fine-tune support | Fine-tunes hosted |
| **Anyscale** | OSS models | Enterprise scale |
| **Hugging Face Inference Endpoints** | Any HF model | When already on HF |

**Use the one your ops team prefers.** They're commoditising fast.

---

## How to pick — the decision tree

```
Is it a structured-output problem?
├── closed-API model + simple retry-on-fail        → Instructor
├── local/open model + 100% schema guarantee       → Outlines
├── notebooks / scripts / quick demos              → Marvin
└── typed-everything agent                          → Pydantic AI

Is it a prompt-optimisation problem?
└── you have a measurable metric + eval set        → DSPy

Is it a RAG framework choice?
├── stateful agents central + RAG one piece        → LangGraph
├── RAG + document parsing central                  → LlamaIndex
└── search-engine-shaped, hybrid retrieval         → Haystack

Is it a multi-agent framework choice?
├── stateful, debuggable, persistent                → LangGraph
├── quick role-play demo                            → CrewAI
├── conversational research                         → AutoGen
├── code-as-action agents                           → smolagents
└── OpenAI-aligned, lean                            → OpenAI Agents SDK

Is it a memory problem?
├── per-user fact memory                            → Mem0
├── long-running agents (months of state)           → Letta
└── conversation-history with structure             → Zep

Is it a web ingestion problem?
├── managed, high-volume                            → Firecrawl
├── one-off, free                                   → Jina Reader
└── self-hosted, OSS                                → Crawl4AI

Is it a routing / cost problem?
├── fast intent classification                      → semantic-router
├── cheap→expensive cascading                       → RouteLLM
└── unified provider interface                      → LiteLLM

Is it a hosting problem?
├── local dev                                       → Ollama
├── production self-host                            → vLLM
└── serverless                                       → Together / Modal / Replicate
```

---

## Why your LangGraph skill still matters

Don't read this lesson as "LangGraph is replaceable." Every alternative
above either:

- **Solves a narrower problem** (Instructor = structured output only; semantic-router = intent only)
- **Lacks production primitives** that LangGraph has (CrewAI has no persistent checkpoints; AutoGen's debug story is rougher)
- **Targets a different mental model** (LlamaIndex = "index"-shaped; DSPy = "compile"-shaped)

LangGraph remains the right primary tool when you need: stateful
graphs, durable checkpointing, human-in-the-loop, complex routing,
production observability. The other libraries are **tools you reach
for inside or around** a LangGraph app, not replacements for it.

The interview answer: *"LangGraph for the runtime; Instructor for
structured outputs at tool boundaries; semantic-router for fast intent
classification before invoking the agent; LiteLLM if I need provider
flexibility beyond what `get_llm()` gives me; DSPy if I'm optimising a
prompt against a measured metric."*

## Run it

```bash
uv run python -m lessons.36_library_landscape.example
uv run python -m lessons.36_library_landscape.example --instructor
uv run python -m lessons.36_library_landscape.example --semantic-router
uv run python -m lessons.36_library_landscape.example --marvin
```

The demos in `example.py` are **opt-in installs** — each one explains
the `uv add ...` command if the library isn't present, so you can pick
and choose what to try.

## Anti-patterns

| Smell | Fix |
|---|---|
| "We use LangChain" as the answer to every architecture question | Know your alternatives. Mention the decision matrix |
| Picking the trendiest library on Twitter | Pick from the decision tree. Trend lags |
| Three frameworks in one codebase | Each adds a learning curve. Pick one primary, others as tools |
| Avoiding alternatives because "we know LangChain" | When DSPy compiles a better prompt than you can write, use DSPy |
| Vendor agent SDK without LiteLLM | Locks you in for no benefit |

## Pairs with

- **[Lesson 04 · Structured output](../04_structured_output/README.md)** — Instructor is the production form
- **[Lesson 10 · `create_agent`](../10_create_agent/README.md)** — CrewAI / smolagents / OpenAI Agents SDK are the alternatives
- **[Lesson 18 · Long-term memory](../18_long_term_memory/README.md)** — Mem0 / Letta / Zep extend the Store
- **[Lesson 38 · Reasoning + routing](../38_reasoning_and_routing/README.md)** — RouteLLM + semantic-router applied
- **[Lesson 34 · Observability](../34_observability_tracing/README.md)** — these libraries all need observability too

## References

- [Instructor docs](https://python.useinstructor.com/)
- [Outlines docs](https://dottxt-ai.github.io/outlines/)
- [Marvin docs](https://www.askmarvin.ai/)
- [Pydantic AI docs](https://ai.pydantic.dev/)
- [DSPy docs](https://dspy.ai/) + [GitHub](https://github.com/stanfordnlp/dspy)
- [LlamaIndex docs](https://docs.llamaindex.ai/)
- [Haystack docs](https://docs.haystack.deepset.ai/)
- [CrewAI docs](https://docs.crewai.com/)
- [AutoGen docs](https://microsoft.github.io/autogen/)
- [smolagents docs](https://huggingface.co/docs/smolagents/) + [GitHub](https://github.com/huggingface/smolagents)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [Mem0 docs](https://docs.mem0.ai/) · [Letta docs](https://docs.letta.com/) · [Zep docs](https://help.getzep.com/)
- [Firecrawl docs](https://docs.firecrawl.dev/) · [Jina Reader](https://jina.ai/reader/) · [Crawl4AI](https://crawl4ai.com/)
- [semantic-router](https://github.com/aurelio-labs/semantic-router) · [RouteLLM](https://github.com/lm-sys/RouteLLM) · [LiteLLM](https://docs.litellm.ai/)
- [vLLM](https://docs.vllm.ai/) · [Ollama](https://ollama.com/) · [llama.cpp](https://github.com/ggerganov/llama.cpp)

## Next →

[Lesson 37 · Multimodal AI](../37_multimodal/README.md) — vision + audio LLMs.
