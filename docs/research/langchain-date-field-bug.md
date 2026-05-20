# LangChain `with_structured_output()` + `datetime.date` — the bug, root cause, and fix

**Investigation date:** 2026-05-20
**Reproduced against:** `langchain==1.3.1`, `langchain-core==1.4.0`, `langchain-openai==1.2.1`, `langchain-mistralai` (per issue), `pydantic==2.13.4`.
**Upstream issue:** [#29604 — "Pydantic model with a datetime.date value using json_schema raises a 400 bad request"](https://github.com/langchain-ai/langchain/issues/29604) — **closed as "not planned"**.

This is the companion doc to [`lessons/21_date_parsing/`](../../lessons/21_date_parsing/README.md). The lesson README explains why we use `str` not `date` in one sentence; this doc explains it in full so the next person who hits this bug doesn't have to repeat the investigation.

---

## TL;DR

1. The bug is **real and unfixed**. Schemas with `"format": "date"` get rejected with HTTP 400 by Mistral's strict mode and (variably) by OpenAI's strict structured-output mode.
2. The root cause lives at exactly one line in `langchain_core.utils.function_calling._rm_titles` — it strips `title` keys but **does not strip `format`, `$ref`, or several other keywords** that OpenAI/Mistral strict mode rejects.
3. **LangChain has declined to fix this** (issue closed "not planned"). You need a workaround.
4. **Three working solutions**, in order of preference:
   - **A.** Type the field as `str` + run `dateparser.parse(...)` in a Pydantic `field_validator`. Works everywhere. *Lesson 21's default.*
   - **B.** Use a **schema sanitizer** that strips `format` (+ a few others) before the call. Keeps the native `date` type. *Lesson 21's optional path.*
   - **C.** Don't use `json_schema` mode — use `method="function_calling"` instead. Sometimes works; provider-dependent.

---

## 1 · Reproducing the bug

```python
from datetime import date
from pydantic import BaseModel
# from langchain_mistralai.chat_models import ChatMistralAI  # the original report

class DummyClass(BaseModel):
    date: date

print(DummyClass.model_json_schema())
```

Pydantic emits:

```json
{
  "properties": {
    "date": {
      "format": "date",       ← THIS is the offending key
      "title": "Date",
      "type": "string"
    }
  },
  "required": ["date"],
  "title": "DummyClass",
  "type": "object"
}
```

When this schema is passed to `ChatMistralAI(...).with_structured_output(DummyClass, method='json_schema')`, the Mistral API responds:

> **HTTP 400 Bad Request:** "Received unsupported keyword `format` in schema."

OpenAI's strict structured-output mode rejects schemas containing keywords outside [its documented subset](https://platform.openai.com/docs/guides/structured-outputs#supported-schemas) (most notably `format`, `pattern` for non-string types, `default`, `$ref` to siblings, and `additionalProperties: {}` with no defined type). The exact tolerance varies between API versions; some calls succeed when the schema is small, fail when it's wrapped in further nesting.

**Verified empirically** in this repo against `pydantic==2.13.4` — the schema above is what gets generated; nothing strips the `format` key downstream.

---

## 2 · Root cause in the LangChain source

The path from "Pydantic model" to "what the provider receives" runs through:

```
BaseModel.model_json_schema()
        │
        ▼
langchain_core.utils.function_calling.convert_to_openai_function(model)
        │
        ▼
_rm_titles(schema)              ← strips ONLY 'title' keys
        │
        ▼
provider-side serializer (langchain_openai / langchain_mistralai / ...)
        │
        ▼
HTTP request body
```

The relevant code (`langchain_core/utils/function_calling.py`, lines 89-118):

```python
def _rm_titles(kv: dict, prev_key: str = "") -> dict:
    """Recursively removes 'title' fields from a JSON schema dictionary."""
    new_kv = {}
    for k, v in kv.items():
        if k == "title":
            if isinstance(v, dict) and prev_key == "properties":
                new_kv[k] = _rm_titles(v, k)
            else:
                continue                    # ← only 'title' gets dropped
        elif isinstance(v, dict):
            new_kv[k] = _rm_titles(v, k)
        else:
            new_kv[k] = v
    return new_kv
```

**The function strips `title` because OpenAI used to reject it.** It was never extended to handle the other keys that newer strict-mode endpoints also reject. Neither `langchain_openai/chat_models/base.py` nor the `openai` Python client strips `format` either — `grep -rn "format.*pop\|del.*format" .venv/Lib/site-packages/langchain_openai/` returns nothing relevant.

**No reachable code path strips `format`.** That's the bug.

---

## 3 · The complete list of offending Pydantic-emitted keys

Pydantic v2 emits these keys that **some** strict-mode endpoints reject. Beyond `date`, the same class of bug shows up with `Email`, `IPv4Address`, `UUID`, `Url`, etc.:

| Pydantic field type | Emitted JSON Schema key | Strict-mode response |
|---|---|---|
| `datetime.date` | `"format": "date"` | Mistral 400; OpenAI sometimes 400 |
| `datetime.datetime` | `"format": "date-time"` | Same |
| `datetime.time` | `"format": "time"` | Same |
| `pydantic.EmailStr` | `"format": "email"` | Same |
| `pydantic.HttpUrl` | `"format": "uri"`, `"minLength": 1` | Same; also `minLength` on a string field can fail |
| `pydantic.UUID4` | `"format": "uuid"`, `"pattern": "<uuid regex>"` | Same |
| `pydantic.IPvAnyAddress` | `"format": "ipvanyaddress"` | Same |
| `Optional[int]` with `Field(ge=0)` | `{"type": ["integer", "null"], "minimum": 0}` | OpenAI strict mode rejects numeric constraints on nullable types |
| Self-referential model | `"$ref"` to a sibling | OpenAI strict mode rejects |
| `dict[str, Any]` | `"additionalProperties": {}` | OpenAI strict mode rejects (must be `False` or have a type) |
| Mixed `Union[str, MyModel]` | `"anyOf": [...]` | Strict mode rejects across most providers |

[*Source: aviadr1 / "How to Fix OpenAI Structured Outputs Breaking Your Pydantic Models", Medium*](https://medium.com/@aviadr1/how-to-fix-openai-structured-outputs-breaking-your-pydantic-models-bdcd896d43bd)

---

## 4 · The three solutions

### Solution A — `str` + Pydantic `field_validator` (recommended default)

```python
from datetime import date
from pydantic import BaseModel, Field, field_validator
import dateparser

TODAY = date.today()

class BirthDateExtraction(BaseModel):
    birth_date: str | None = Field(
        default=None,
        description="ISO 8601 (YYYY-MM-DD). Null if not present.",
    )

    @field_validator("birth_date")
    @classmethod
    def _parse_and_sanity(cls, v):
        if v is None or v == "":
            return None
        parsed = dateparser.parse(v, settings={"STRICT_PARSING": True})
        if parsed is None:
            raise ValueError(f"unparseable: {v!r}")
        if parsed.date() > TODAY:
            raise ValueError(f"future date {parsed.date()}")
        return parsed.date().isoformat()
```

| ✅ Pro | ❌ Con |
|---|---|
| Works on **every** provider — no `format` key emitted | You lose Pydantic's native `date` coercion (downstream code receives `str`) |
| Total control over the parse step (locales, calendars, relative dates) | Two-stage thinking — schema field + validator |
| Zero new dependencies beyond `dateparser` (already in lesson 20) | Slight overhead on every parse (negligible) |

**This is what lesson 21 ships with by default.** It also matches the GitHub issue reporter's own workaround ("manually removing format and adding a description field resolves the issue").

### Solution B — schema sanitizer (keeps native `date` typing)

If your downstream code really wants a `datetime.date` object, you can keep the Pydantic field as `date` and strip the offending keys *before* LangChain hands them to the provider. This repo ships [`shared/llm/schema_sanitizer.py`](../../shared/llm/schema_sanitizer.py) with a reusable `with_structured_output_safe()` wrapper.

```python
from datetime import date
from pydantic import BaseModel
from shared.llm import with_structured_output_safe

class BirthDateExtraction(BaseModel):
    birth_date: date          # ← native date type works again

llm = with_structured_output_safe(BirthDateExtraction)
result = llm.invoke("Born April 5, 1987.")
print(type(result.birth_date), result.birth_date)
# <class 'datetime.date'>  1987-04-05
```

The sanitizer recursively strips `format`, `$ref` to siblings, fixes `additionalProperties: {}`, and patches numeric constraints on nullable types — i.e. the full catalogue from §3.

| ✅ Pro | ❌ Con |
|---|---|
| Native `date` typing — downstream code uses `result.birth_date.year` etc. | Strips info the schema *meant* — model can no longer use the format hint |
| Reusable across schemas — write once, apply everywhere | Adds a (small) amount of LangChain-coupling to your code |
| Closest to "the upstream fix" | One more thing that can break on a LangChain version bump |

### Solution C — `method="function_calling"` instead of `"json_schema"`

For some provider × model combinations, swapping the structured-output method bypasses the bug because function-calling endpoints parse schemas more leniently.

```python
llm = get_llm().with_structured_output(MySchema, method="function_calling")
```

| ✅ Pro | ❌ Con |
|---|---|
| Zero schema changes; one-line fix | Provider-dependent — works on OpenAI tool-calling, *may* still fail on Mistral |
| Easy to A/B test | You lose the *deterministic* guarantee of strict mode |

**Recommended when:** you can't change the Pydantic schema (it's owned by another team) and you've confirmed your specific model accepts it.

---

## 5 · Which solution to pick

```
You control the schema?            ┐
   │                                │
   │  yes ──► Need datetime.date    │
   │            in downstream code? │
   │                │               │
   │                │  yes ─────► Solution B (sanitizer)
   │                │  no  ─────► Solution A (str + validator)   ← default
   │                                │
   no ───► Solution C (method="function_calling") + provider test
```

**Defaults baked into this repo:**
- Lesson 21 ships **Solution A** as the headline example (universal, no LangChain-coupling, easy to read).
- `shared/llm/schema_sanitizer.py` ships **Solution B** for when you need it.
- The lesson README mentions Solution C in the "alternatives" section.

---

## 6 · A note on long-term stability

This bug has been open since **January 2025** and was [closed-as-"not planned"](https://github.com/langchain-ai/langchain/issues/29604) by maintainers. That signals:

- LangChain considers the Pydantic-to-provider-schema translation *out of scope* for `langchain-core`.
- Each provider integration package (`langchain-mistralai`, `langchain-openai`, etc.) is expected to handle its own quirks — but in practice none of them strip `format`.
- The community-accepted solution is the sanitizer/workaround pattern, not waiting for an upstream fix.

**Re-verify quarterly.** If LangChain or provider strict-mode behavior changes, update this doc and re-run the reproducer in §1.

---

## References

| # | Source |
|---|---|
| [1] | [LangChain issue #29604](https://github.com/langchain-ai/langchain/issues/29604) — original bug report, reproducer, status "closed: not planned" |
| [2] | [`langchain_core/utils/function_calling.py` `_rm_titles`](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/utils/function_calling.py#L89) — source of the bug |
| [3] | [OpenAI structured outputs — supported schemas](https://platform.openai.com/docs/guides/structured-outputs#supported-schemas) — the documented subset that strict mode accepts |
| [4] | [Mistral docs — JSON schema mode](https://docs.mistral.ai/capabilities/structured-output/json_schema/) — provider-side restrictions |
| [5] | [aviadr1 — "How to Fix OpenAI Structured Outputs Breaking Your Pydantic Models"](https://medium.com/@aviadr1/how-to-fix-openai-structured-outputs-breaking-your-pydantic-models-bdcd896d43bd) — community-known list of breaking patterns + sanitizer reference |
| [6] | [LangChain v1.3 structured-output docs](https://docs.langchain.com/oss/python/langchain/structured-output) — current canonical API surface |
