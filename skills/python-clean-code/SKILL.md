---
name: python-clean-code
description: Opinionated Python clean-code rules — naming, type hints (PEP 604 unions, Protocol, Annotated), error handling, dataclasses vs Pydantic vs TypedDict, docstring style, function size, import order, and module structure. Use when writing new Python code, reviewing a PR for readability, refactoring for clarity, or setting team style. Pairs with python-design-patterns-applied.
---

# Python Clean Code

Concrete, opinionated rules. Where Black/Ruff disagree with this
doc, Black/Ruff win — those are mechanical and consistent. This
doc covers the parts they can't enforce.

## Quick rules (the 80% you'll use daily)

1. **Type-hint every public function.** Internal helpers may go without, but module APIs always have hints.
2. **Use PEP 604 unions: `str | None`, not `Optional[str]`.** Python 3.10+ only — fine for our 3.12 default.
3. **Prefer keyword-only args for boolean flags.** `def send(*, dry_run: bool = False)` makes call sites readable.
4. **Functions ≤ 30 lines, ≤ 5 args.** When you cross either, split or pass a dataclass.
5. **One return type per function.** `def f() -> int | None` is fine; `def f() -> int | str | dict` is a smell.
6. **Crash early with a good message.** A clean `raise ValueError(f"unknown provider {p!r}")` beats a defensive fallback that hides the bug.
7. **No mutable default arguments.** `def f(xs: list | None = None) -> ...:  xs = xs or []`
8. **No bare `except:` or `except Exception:`** unless you re-raise. Catch what you can actually handle.
9. **Imports: stdlib → third-party → first-party,** alphabetised inside each block. Ruff's `I` rules do this for you.
10. **Module docstring on every non-trivial file.** One sentence is enough.

---

## Naming

- **Modules / files:** `lowercase_with_underscores.py`. Short. Nouns. `embeddings.py`, not `embedding_utilities_v2.py`.
- **Classes:** `PascalCase`. Nouns. `UserRepo`, `RetryPolicy`. Avoid `*Manager`, `*Helper`, `*Util` — they're confessions of "I couldn't think of a name."
- **Functions / methods:** `snake_case`, verbs. `fetch_user`, `compile_graph`. Not `userFetcher`, `data`.
- **Variables:** `snake_case`. Single-letter only for very local loop variables / math.
- **Constants:** `SCREAMING_SNAKE_CASE`, module-level. `MAX_RETRIES = 3`.
- **Private:** prefix with one underscore (`_helper`). Two underscores triggers name mangling — almost never what you want.
- **Type variables:** single capital letter, `T`, `K`, `V`. For domain-specific generics: `UserT`, `ItemT`.
- **Boolean variables / flags:** start with `is_`, `has_`, `should_`. `is_admin`, `has_pending_writes`, `should_retry`.

Avoid:
- abbreviations no one outside the team recognises (`usr`, `cfg`, `qty`)
- Hungarian notation (`strName`, `bFlag`)
- numeric suffixes (`process_data2`, `final_v3`) — fix the original instead

---

## Type hints

```python
from collections.abc import Iterable, Callable
from typing import Annotated, Literal, Protocol

# unions — PEP 604
def get(user_id: str) -> User | None: ...

# generic containers — collections.abc not typing
def total(xs: Iterable[float]) -> float: ...

# callables
on_done: Callable[[Result], None]

# literal enums (cheap & lint-friendly)
def set_mode(mode: Literal["fast", "safe"]) -> None: ...

# protocol = structural type ("anything with these methods")
class SupportsClose(Protocol):
    def close(self) -> None: ...

# annotated — attach metadata for libraries like Pydantic/FastAPI
PositiveInt = Annotated[int, Field(gt=0)]
```

Don't over-genericise:

```python
# meh — too generic to be useful
def process(data: Any) -> Any: ...

# better — say what you mean
def process(data: dict[str, list[float]]) -> pd.DataFrame: ...
```

---

## Choosing a data class

| Use case | Pick |
|---|---|
| Plain value object inside your code | `@dataclass(frozen=True, slots=True)` |
| Loaded from JSON / forms / env vars; needs validation | `pydantic.BaseModel` |
| Lightweight typed dict — no methods, no behaviour | `TypedDict` |
| Function return with named fields, no methods | `NamedTuple` |
| Mutable, behaviour-rich entity in a domain model | plain `class` |

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Citation:
    source: str
    text: str
    confidence: float = 1.0
```

```python
from pydantic import BaseModel, Field

class CreateUserRequest(BaseModel):
    email: str = Field(pattern=r".+@.+")
    age: int = Field(ge=0, lt=150)
```

`slots=True` saves memory; `frozen=True` makes it hashable and
catches accidental mutations.

---

## Error handling

```python
# good — narrow, named, with context
try:
    data = json.loads(payload)
except json.JSONDecodeError as e:
    raise InvalidPayloadError(f"bad JSON at offset {e.pos}") from e

# also good — convert and re-raise so the original cause is preserved
try:
    return repo.get(user_id)
except DatabaseError as e:
    raise UserLookupFailed(user_id) from e
```

Rules:
- **One thing per `try` block.** Three lines in `try`, six in `except`.
- **`raise X from e`** to preserve causes. Plain `raise X` loses the chain.
- **Make exceptions specific.** `class RefundDenied(Exception)` beats generic `RuntimeError`.
- **Don't catch `Exception` to "be safe."** That's the symptom of a bug, not a safety measure.

**Never:**

```python
try: ...
except Exception:
    pass    # ← silent failure, will haunt you
```

**Rarely-OK** (for explicit best-effort code paths):

```python
try: _send_metric(...)
except Exception as e: logger.warning("metric send failed", exc_info=e)
```

---

## Docstrings

Use **Google style** (readable, widely supported by mkdocs/IDE):

```python
def retrieve(query: str, k: int = 4) -> list[Document]:
    """Return the top-k chunks most relevant to `query`.

    Args:
        query: User question. Should already be in the corpus language.
        k: Number of chunks to return. Defaults to 4.

    Returns:
        Chunks ordered by descending relevance.

    Raises:
        RetrievalError: If the index is unavailable.
    """
```

- One-line summary on the same line as `"""`, full stop at the end.
- Blank line before sections.
- Skip docstrings on truly self-explanatory helpers (`_format_row`).

For modules:

```python
"""Switchable LLM provider layer — see shared/llm/README.md for the pattern."""
```

---

## Function shape

A function should fit on **one screen**. If it doesn't:

1. Pull each "step" out into a named helper.
2. If you find yourself passing 6+ arguments, group them in a dataclass.
3. If you find a comment "# now do step 2", that comment wants to be a function name.

```python
# before: 60 lines, deep nesting
def run(): ...

# after: 8 lines reading like prose
def run():
    payload = _load_payload()
    if _is_stale(payload): _refresh(payload)
    _validate(payload)
    return _publish(payload)
```

---

## Module structure

```python
"""One-line module docstring."""

# 1. __future__ imports (only if needed)
from __future__ import annotations

# 2. stdlib
import os
from pathlib import Path

# 3. third-party
import httpx
from pydantic import BaseModel

# 4. first-party
from shared.settings import settings

# 5. module constants
DEFAULT_TIMEOUT = 30

# 6. public types / dataclasses

# 7. private helpers (_lowercase)

# 8. public functions / classes
```

Group related modules into packages with **focused** `__init__.py`
that **re-exports the public surface**:

```python
# shared/llm/__init__.py
from shared.llm.base import get_llm
__all__ = ["get_llm"]
```

Callers do `from shared.llm import get_llm` — clean. They don't
need to know the file layout inside the package.

---

## Imports

Rules Ruff/isort enforce; you should still know them:

- Absolute imports for first-party (`from shared.settings import settings`).
- Relative imports only for tightly-coupled siblings inside a package — keep them one level (`from .base import X`).
- **Never** use `from x import *` outside `__init__.py` re-exports.
- Put a module name in `__all__` to declare your public API.

---

## Code smells (fix on sight)

| Smell | Fix |
|---|---|
| `def f(data: dict): data["a"] + data["b"]` — unstructured dict | Replace with dataclass/Pydantic |
| 5-deep nested `if/else` | Early-return guard clauses |
| Functions that take `**kwargs` then read random keys | Make them real parameters |
| Comments explaining *what* (not *why*) | Rename the variable/function instead |
| `TODO` older than 60 days | Either do it or delete it |
| `print()` in production code | Use `logging` (or `rich.console` for CLIs) |
| Catching exception by class name string | Import the class and catch the type |
| Long parameter lists with mixed types | Group with a dataclass |

---

## Tools the codebase should run

- **Black** (or **Ruff format**) — line length 100, no debate.
- **Ruff** — lint + import sort + simple fixes. Enable rule sets `E,W,F,I,N,UP,B,SIM,RUF`.
- **mypy** or **pyright** — type check. Strict mode on new code; gradually tighten old code.
- **pytest** — see the `fastapi-pytest-functional` skill.

Pin them in `pyproject.toml`'s dev extra so everyone runs the same versions.
