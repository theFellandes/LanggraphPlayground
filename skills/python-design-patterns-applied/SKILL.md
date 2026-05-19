---
name: python-design-patterns-applied
description: Practical Python design patterns with worked examples — Adapter (multi-provider/SDK swaps), Factory, Strategy, Decorator, Observer/pub-sub, Context Manager, Registry, and the rule of three. Use when designing reusable abstractions, building multi-LLM/multi-provider systems, deciding between patterns, or refactoring tangled code into composable parts. Skip premature abstractions — patterns are tools, not goals.
---

# Python Design Patterns — Applied

Practical patterns for real Python codebases. Every section answers
two questions: *what problem does this solve* and *what does the
real-world code look like*.

## Guiding rules (read first)

1. **Patterns are tools, not goals.** If a `dict` literal works, use a dict literal. Reach for a pattern when the simpler thing has actually broken down.
2. **Rule of three.** Wait until you have three near-duplicate cases before abstracting. Two cases are still cheaper to copy-paste than to design around.
3. **Composition over inheritance.** Default to composing small objects. Inheritance is for true is-a relationships and rarely more than one level deep.
4. **Name patterns honestly.** Call the file `openai_adapter.py`, not `concrete_openai_strategy_factory.py`.
5. **Pure functions where possible.** A pattern that boils down to a function is usually better than a pattern that boils down to a class hierarchy.

---

## Pattern · Adapter (the workhorse)

**Problem:** you want one calling convention but multiple backends behind it (different LLM providers, different storage backends, different vendor APIs).

**Shape:** a small factory + one file per backend. Each backend file exposes a `build(...)` function (or class with one method) that returns the unified type.

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
    if provider not in _ADAPTERS:
        raise ValueError(f"Unknown provider {provider!r}. Available: {sorted(_ADAPTERS)}")
    return _ADAPTERS[provider](model=model, **kw)
```

```python
# shared/llm/openai_adapter.py
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-4.1"

def build(model: str | None = None, **kw) -> BaseChatModel:
    return ChatOpenAI(model=model or DEFAULT_MODEL, api_key=settings.openai_api_key, **kw)
```

**Why this shape wins**

- Adding a provider is **one new file + one dict entry**. Zero changes to callers.
- The factory has no provider-specific imports — testing the dispatch logic doesn't pull in OpenAI's SDK.
- Each adapter owns its defaults (model id, retry behaviour, etc.) so they don't leak into the factory.

**When NOT to use:** if you have exactly one backend, don't pre-emptively build an adapter layer. Add it the day you add the second backend.

---

## Pattern · Factory function (vs class)

For "give me a configured X based on these inputs", a **function** beats a `*Factory` class 9 times out of 10:

```python
# good
def make_retriever(kind: str = "vector", **kw) -> BaseRetriever:
    if kind == "vector":  return Chroma(**kw).as_retriever()
    if kind == "bm25":    return BM25Retriever.from_texts(**kw)
    raise ValueError(kind)

# overkill
class RetrieverFactory:
    def create(self, kind, **kw): ...
```

Reach for a class only when the factory itself has state (e.g. a connection pool it reuses across calls).

---

## Pattern · Strategy (interchangeable algorithms)

When the *same call site* needs to swap between behaviours at runtime:

```python
from typing import Callable, Protocol

class Scorer(Protocol):
    def __call__(self, candidate: str, query: str) -> float: ...

def bm25_score(c: str, q: str) -> float: ...
def cosine_score(c: str, q: str) -> float: ...

def rerank(candidates: list[str], query: str, *, scorer: Scorer = cosine_score) -> list[str]:
    return sorted(candidates, key=lambda c: scorer(c, query), reverse=True)
```

`Protocol` (structural typing) lets you accept any callable matching the
shape — no inheritance required. The default keyword argument lets
callers ignore the strategy 95% of the time.

---

## Pattern · Decorator (function + class flavours)

**Function decorators** wrap a call. Use for cross-cutting concerns: caching, timing, retries, validation.

```python
from functools import wraps
import time, logging

def timed(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:    return fn(*a, **kw)
        finally: logging.info(f"{fn.__name__} took {time.perf_counter()-t0:.3f}s")
    return wrapper

@timed
def embed_documents(docs): ...
```

**`@dataclass`-as-decorator** is the most useful built-in decorator pattern — auto-generates `__init__`, `__repr__`, `__eq__`:

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Citation:
    source: str
    text: str
    confidence: float
```

Use `frozen=True` for value objects (hashable, immutable). `slots=True` saves memory when you create lots of them.

---

## Pattern · Context manager

For paired setup/teardown — DB connections, files, temp dirs, mock patching:

```python
from contextlib import contextmanager

@contextmanager
def temporary_env(**vars):
    import os
    original = {k: os.environ.get(k) for k in vars}
    os.environ.update(vars)
    try:
        yield
    finally:
        for k, v in original.items():
            if v is None: os.environ.pop(k, None)
            else:         os.environ[k] = v

with temporary_env(LLM_PROVIDER="openai"):
    llm = get_llm()  # uses the override
```

Use `@contextmanager` for the 80% case. Only build a class with
`__enter__` / `__exit__` if you need to *also* be usable manually.

---

## Pattern · Observer / pub-sub (in-process)

For "many independent reactions to one event" without coupling the
producer to the consumers. The Python-native shape is a list of
callables + a `notify` method — no `Observable` base class needed.

```python
from typing import Callable, TypeVar
T = TypeVar("T")

class Signal[T]:
    def __init__(self) -> None:
        self._handlers: list[Callable[[T], None]] = []

    def connect(self, handler: Callable[[T], None]) -> None:
        self._handlers.append(handler)

    def emit(self, payload: T) -> None:
        for h in self._handlers:
            h(payload)

# usage
job_started = Signal[dict]()
job_started.connect(lambda p: logging.info("started: %s", p["id"]))
job_started.connect(metrics.record_job_start)
job_started.emit({"id": "abc123"})
```

For *cross-process* pub-sub, don't roll your own — use Redis pub/sub,
Kafka, or an MQ.

---

## Pattern · Registry (plugin discovery)

When you want third-party / user code to add behaviour without
editing core code. The cleanest Python form is a decorator that
registers into a dict:

```python
TOOLS: dict[str, callable] = {}

def register_tool(name: str):
    def deco(fn):
        TOOLS[name] = fn
        return fn
    return deco

@register_tool("weather")
def get_weather(city: str) -> str: ...

@register_tool("time")
def current_time() -> str: ...
```

This is the underlying shape behind `@app.get("/path")` (FastAPI) and
`@pytest.fixture`. Don't overthink it.

---

## Pattern · Singleton — usually wrong, usually a module

You almost never need the GoF Singleton in Python. **Modules are
already singletons.** Put your "global" thing at module scope:

```python
# shared/settings.py
settings = Settings()   # constructed once on import
```

Anywhere that does `from shared.settings import settings` shares the
same instance. If you need explicit lifecycle control, use a
factory function:

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_db() -> Database:
    return Database(url=settings.db_url)
```

The "real" Singleton class with `__new__` overrides is almost always
a code smell. Avoid.

---

## Pattern · Repository / data-access layer

Separate "how data is stored" from "what your domain does." The
repository exposes domain-shaped methods; callers don't know whether
storage is Postgres, an HTTP API, or a dict.

```python
from typing import Protocol

class UserRepo(Protocol):
    def get(self, user_id: str) -> User | None: ...
    def save(self, user: User) -> None: ...
    def list_active(self) -> list[User]: ...

class PostgresUserRepo:    # production
    def __init__(self, pool): self.pool = pool
    def get(self, user_id): ...

class InMemoryUserRepo:    # tests
    def __init__(self): self._db: dict[str, User] = {}
    def get(self, user_id): return self._db.get(user_id)
```

Now your service code accepts a `UserRepo` and you swap in
`InMemoryUserRepo` for fast, deterministic tests.

---

## Anti-patterns to recognise

- **God object / mega-class.** A class with 30+ methods is doing too many things. Split by responsibility.
- **Premature inheritance.** `class FooService(BaseService): ...` with one subclass — collapse it.
- **Pseudo-OO Python.** Static-method-only classes are namespaced functions. Use a module.
- **Mutable default arguments.** `def f(items=[])` — items is shared across calls. Use `items: list | None = None`.
- **Catch-all `except Exception`.** Catch what you can handle; let the rest crash with a real traceback.
- **Importing for side effects.** Modules should be safe to import. Construct things in functions, not at import time.

---

## Choosing a pattern (decision shortcuts)

| Situation | Reach for |
|---|---|
| Two-or-more interchangeable backends | **Adapter** + factory function |
| One algorithm with multiple variants | **Strategy** via `Protocol` callable |
| Cross-cutting wrap (timing, cache, retry) | **Decorator** |
| Paired setup/teardown | `@contextmanager` |
| One event, many handlers, in-process | **Signal/Observer** (list of callables) |
| External code adds entries | **Registry** with `@register(name)` |
| Domain wants storage to be swappable | **Repository** via `Protocol` |
| Process-wide "single instance" | **Module-level value** or `@lru_cache(maxsize=1)` |
