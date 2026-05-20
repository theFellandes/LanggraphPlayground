---
name: langchain-1x-engineering
description: LangChain 1.x (2026) engineering patterns — LCEL Runnable composition with pipe and parallel, the switchable provider adapter (anthropic/openai/...) via get_llm(), messages and ChatPromptTemplate, structured output with Pydantic via with_structured_output, the @tool decorator and bind_tools, the new create_agent API plus middleware (Summarization, PII, HumanInTheLoop, ModelCallLimit), RAG with Chroma + FastEmbed (no extra API key), and the langchain_classic.retrievers location for MultiQuery/ContextualCompression. Use when building, debugging, or refactoring LangChain apps on the 2026 stack.
---

# LangChain 1.x Engineering

Production patterns for **LangChain 1.3+** (May 2026 stack). Knowing
where things moved in v1 — and which APIs are now canonical —
saves you most of the debugging time.

## Where things moved in v1 (cheat sheet)

| You want | Import from |
|---|---|
| `create_agent` (new prebuilt) | `langchain.agents` |
| Middleware classes (`SummarizationMiddleware`, …) | `langchain.agents.middleware` |
| `Runnable`, `RunnableLambda`, `RunnableParallel` | `langchain_core.runnables` |
| `ChatPromptTemplate`, `MessagesPlaceholder` | `langchain_core.prompts` |
| `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage` | `langchain_core.messages` |
| `@tool`, `BaseTool` | `langchain_core.tools` |
| `StrOutputParser` | `langchain_core.output_parsers` |
| `MultiQueryRetriever`, `ContextualCompressionRetriever`, `LLMChainExtractor` | `langchain_classic.retrievers` (NOT `langchain.retrievers`) |
| `RecursiveCharacterTextSplitter` | `langchain_text_splitters` |
| `Chroma` | `langchain_chroma` |
| `FastEmbedEmbeddings` | `langchain_community.embeddings.fastembed` |
| `TavilySearchResults` | `langchain_community.tools.tavily_search` |

**Common pitfall:** `from langchain.retrievers import …` no longer
works. Use `langchain_classic.retrievers` in v1.

### Production-bug to memorise — `date` fields + structured output

Pydantic's `date` / `datetime` / `EmailStr` / `HttpUrl` / `UUID4` fields
emit a JSON-Schema `"format"` keyword that **Mistral's and OpenAI's
strict structured-output modes reject** with HTTP 400 *"Received
unsupported keyword 'format' in schema."* LangChain's
`_rm_titles` strips `title` but not `format`, and the upstream issue
([#29604](https://github.com/langchain-ai/langchain/issues/29604)) is
**closed as "not planned"**.

**Three working fixes**, in order of preference:

```python
# A · str + Pydantic validator — universal, no LangChain coupling
class Person(BaseModel):
    birth_date: str | None = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("birth_date")
    @classmethod
    def _parse(cls, v):
        if v is None: return None
        d = dateparser.parse(v, settings={"STRICT_PARSING": True})
        if d is None: raise ValueError(f"unparseable: {v!r}")
        return d.date().isoformat()

# B · schema sanitizer — keeps native date typing
from shared.llm import with_structured_output_safe   # in this repo
class Person(BaseModel):
    birth_date: date | None                          # native type works again
llm = with_structured_output_safe(Person)            # strips "format" first

# C · function_calling mode — sometimes works on providers that reject json_schema
llm = get_llm().with_structured_output(Person, method="function_calling")
```

Pick **A** by default. Pick **B** when downstream code needs a real
`datetime.date`. Pick **C** when you can't change the schema.

See `docs/research/langchain-date-field-bug.md` for the full
investigation (reproducer, root cause in `_rm_titles`, sanitizer code).

---

## Pattern · Switchable provider via `get_llm()`

Never instantiate `ChatAnthropic` / `ChatOpenAI` directly in
business logic. Wrap them once in an adapter layer:

```python
# shared/llm/base.py
from langchain_core.language_models.chat_models import BaseChatModel
from shared.llm import anthropic_adapter, openai_adapter

_ADAPTERS = {
    "anthropic": anthropic_adapter.build,
    "openai":    openai_adapter.build,
}

def get_llm(provider: str | None = None, model: str | None = None, **kw) -> BaseChatModel:
    provider = (provider or settings.llm_provider).lower()
    return _ADAPTERS[provider](model=model, **kw)
```

```python
# shared/llm/anthropic_adapter.py
from langchain_anthropic import ChatAnthropic
DEFAULT_MODEL = "claude-sonnet-4-6"

def build(model=None, **kw):
    return ChatAnthropic(model=model or DEFAULT_MODEL,
                         api_key=settings.anthropic_api_key, **kw)
```

**Bonus — fallback chain** (Aletheia-style, but using LangChain's
built-in `Runnable.with_fallbacks`):

```python
primary = ChatAnthropic(model="claude-sonnet-4-6")
backup  = ChatOpenAI(model="gpt-4.1")
llm = primary.with_fallbacks([backup])    # tries primary; backup on any error
llm.invoke("...")
```

Wrap that inside `get_llm()` and the whole project gets resilience
to quota errors, network blips, and auth failures for free. See
`shared/llm/base.py` in the LanggraphPlayground reference project
for a clean implementation that auto-promotes to a configured
provider when the primary's key is missing.

Switching providers is one env-var flip — no code changes. Adding
a provider is one new file + one dict entry. See the
`python-design-patterns-applied` skill for the full pattern.

---

## Pattern · LCEL — `prompt | model | parser`

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

chain = (
    ChatPromptTemplate.from_template("Translate {text!r} to {language}.")
    | get_llm()
    | StrOutputParser()
)
chain.invoke({"text": "hi", "language": "Turkish"})
```

Every Runnable supports `.invoke`, `.stream`, `.batch`, `.ainvoke`,
`.astream`, `.abatch`. You wrote one line and got streaming, async,
and concurrent batching for free.

## Pattern · Fan-out with `RunnableParallel`

```python
from langchain_core.runnables import RunnableParallel

llm = get_llm()
fan_out = RunnableParallel(
    poem = ChatPromptTemplate.from_template("2-line poem about {topic}") | llm | StrOutputParser(),
    joke = ChatPromptTemplate.from_template("one-liner joke about {topic}") | llm | StrOutputParser(),
    fact = ChatPromptTemplate.from_template("one fact about {topic}")  | llm | StrOutputParser(),
)
fan_out.invoke({"topic": "octopuses"})   # → {"poem": ..., "joke": ..., "fact": ...}
```

For a RAG-shaped chain, the dict-syntax shortcut is the canonical
form:

```python
from langchain_core.runnables import RunnablePassthrough

rag = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
rag.invoke("How does fermentation work?")
```

---

## Pattern · Structured output

Don't parse free text. Use `with_structured_output`:

```python
from pydantic import BaseModel, Field
from typing import Literal

class Ticket(BaseModel):
    category: Literal["billing", "bug", "feature_request", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    summary: str = Field(description="One sentence.")

llm = get_llm().with_structured_output(Ticket)
ticket: Ticket = llm.invoke("Your service charged me twice last week. Refund please.")
```

The `Field(description=...)` text is part of the prompt the model
sees. Treat it as inline instruction.

---

## Pattern · Tools — `@tool` and `bind_tools`

```python
from langchain_core.tools import tool
from typing import Annotated

@tool
def add(a: Annotated[int, "first number"], b: Annotated[int, "second number"]) -> int:
    """Add two integers."""
    return a + b

llm_with_tools = get_llm().bind_tools([add, multiply])
ai = llm_with_tools.invoke([HumanMessage("What's 17 + 25?")])
# ai.tool_calls -> [{"name": "add", "args": {"a": 17, "b": 25}, "id": "..."}]
```

`tool_calls` is structured. To complete the round-trip by hand, run
each tool, build a `ToolMessage(content, tool_call_id=call["id"])`,
append, and call the model again. **In practice, let `create_agent`
do this for you** (next section).

---

## Pattern · `create_agent` (the v1 prebuilt)

```python
from langchain.agents import create_agent

agent = create_agent(
    model=get_llm(),
    tools=[add, multiply, lookup_city_population],
    system_prompt="Be precise. Use tools whenever they help.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "What's 17+25?"}]})
# result["messages"] holds the full conversation incl. tool calls + responses
```

Input is `{"messages": [...]}`. Output is the same key with the full
history appended.

Replaces `create_react_agent` from earlier versions. Inspect the
underlying graph with `agent.get_graph().draw_ascii()`.

---

## Pattern · Middleware (the v1 killer feature)

Middleware lets you intercept the agent loop at six hook points:
`before_agent`, `before_model`, `wrap_model_call`, `wrap_tool_call`,
`after_model`, `after_agent`. Custom or prebuilt:

```python
from langchain.agents.middleware import (
    AgentMiddleware,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    ModelCallLimitMiddleware,
    ToolRetryMiddleware,
)

class LoggingMW(AgentMiddleware):
    def before_model(self, state, runtime):
        print(f"→ msg count: {len(state['messages'])}")

agent = create_agent(
    model=get_llm(),
    tools=[refund_tool, lookup_tool],
    middleware=[
        LoggingMW(),
        SummarizationMiddleware(model=get_llm(), trigger=("tokens", 1500)),
        PIIMiddleware(pii_type="credit_card", strategy="redact"),
        HumanInTheLoopMiddleware(interrupt_on={"refund_tool": True}),
        ModelCallLimitMiddleware(max_calls=10),
    ],
)
```

Middleware composes in **declaration order** for `before_*` and
reverse for `after_*` (onion).

**Custom middleware = subclass `AgentMiddleware` and override
hooks.** Override `wrap_model_call(request, handler)` for full
control over the model call (caching, fallbacks, request
rewriting).

---

## Pattern · RAG (load → split → embed → store → retrieve → generate)

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

docs    = TextLoader("data/handbook.md", encoding="utf-8").load()
chunks  = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_documents(docs)
embed   = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")    # local CPU, ~33MB
store   = Chroma.from_documents(chunks, embedding=embed,
                                persist_directory="data/chroma_index",
                                collection_name="handbook")
retriever = store.as_retriever(search_kwargs={"k": 4})
```

`FastEmbedEmbeddings` runs locally — no extra API key. Great for
teaching, dev, and offline-capable production. Swap for
`OpenAIEmbeddings` or `CohereEmbeddings` when you need scale.

---

## Pattern · Advanced retrievers (`langchain_classic.retrievers`)

```python
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

base = store.as_retriever(search_kwargs={"k": 4})

# Query expansion — LLM rewrites the query into variants
multi = MultiQueryRetriever.from_llm(retriever=base, llm=get_llm())

# Per-chunk filtering — LLM keeps only relevant sentences
compressed = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(get_llm()),
    base_retriever=base,
)

# Stack them
final = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(get_llm()),
    base_retriever=multi,
)
```

**Critical:** these are in **`langchain_classic.retrievers`** in v1.
The old `langchain.retrievers` path is gone.

---

## Pattern · Listeners — non-interactive debugging

```python
from langchain_core.tracers.schemas import Run

def on_start(run: Run, config=None):
    print(f"→ {run.name}  inputs={run.inputs}")

def on_end(run: Run, config=None):
    print(f"← {run.name}  outputs={run.outputs}")

chain.with_listeners(on_start=on_start, on_end=on_end).invoke(...)
```

Use when you want a log of every Runnable step without stopping
execution. For interactive stepping, drop a `breakpoint()`
(`PYTHONBREAKPOINT=ipdb.set_trace`).

---

## Pattern · LangSmith — prod-grade observability

```bash
# in .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls-...
LANGSMITH_PROJECT=my-app
```

Every Runnable + every LangGraph node becomes a clickable trace in
the LangSmith UI. Inputs, outputs, latency, token counts, errors.
The production analogue of ipdb.

---

## Pattern · Choosing a chain shape

| You want | Use |
|---|---|
| Linear: prompt → model → parser | LCEL (`a | b | c`) |
| Multiple parallel branches from the same input | `RunnableParallel(a=..., b=...)` |
| Enrich the input mid-chain | `RunnablePassthrough.assign(...)` |
| Arbitrary Python step in the chain | `RunnableLambda(fn)` |
| Branching on a condition | `RunnableBranch` or move to LangGraph |
| Loops / cycles / persistent state | **LangGraph** — see `langgraph-1x-engineering` |
| Tools + agent loop | `create_agent` |
| Tools + agent loop + cross-cutting policy | `create_agent` + middleware |

---

## Anti-patterns

| Anti-pattern | Fix |
|---|---|
| `from langchain.retrievers import ...` | Use `langchain_classic.retrievers` |
| Instantiating `ChatAnthropic(...)` at module top level | Build inside a function; use `get_llm()` factory |
| Hand-rolled JSON parsing of model output | `with_structured_output(MySchema)` |
| Hand-rolled tool-call loop for an agent | `create_agent` |
| Per-tool retry code duplicated across tools | `ToolRetryMiddleware` |
| Long chats blowing context | `SummarizationMiddleware` |
| Re-implementing PII redaction in each prompt | `PIIMiddleware(pii_type="email", strategy="redact")` |
| Re-implementing approval gates per tool | `HumanInTheLoopMiddleware(interrupt_on={...})` |

---

## Minimal pyproject deps (May 2026)

```toml
dependencies = [
  "langchain>=1.3",
  "langchain-core>=1.0",
  "langchain-classic>=1.0",      # for retrievers, indexing API
  "langchain-community>=0.4",
  "langchain-anthropic>=0.3",    # or openai
  "langchain-openai>=0.3",
  "langchain-chroma>=0.2",       # RAG
  "fastembed>=0.4",              # local CPU embeddings
  "pydantic>=2.7",
  "pydantic-settings>=2.4",
]
```

If you also use LangGraph (very likely): see `langgraph-1x-engineering`.
