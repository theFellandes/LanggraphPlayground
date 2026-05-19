# Lesson 04 · Structured output

## What you'll learn

- How to declare an output shape with a Pydantic `BaseModel`
- How `model.with_structured_output(SchemaClass)` returns a Runnable that yields a typed instance
- Common patterns: extraction (long free-text → structured record) and classification (short text → small enum)
- Why `Field(description=...)` matters — those descriptions are part of the prompt the model sees

## Why it matters

The moment your app needs to make a *decision* based on model output —
"is this ticket urgent?", "does this email need a refund?" — parsing
free text becomes a liability. Structured output gives you a typed
object, validated by Pydantic, every time.

## Key concepts

- **Pydantic `BaseModel`** — the schema you want back. Subclass it, annotate fields, optionally add validation (`ge=`, `le=`, `Literal[...]`).
- **`Field(description=...)`** — populates the JSON-schema description the model sees. Treat it like an inline prompt.
- **`with_structured_output(Schema)`** — returns a Runnable that calls the model with the schema attached (tool-use mode under the hood) and parses the result into a `Schema` instance.
- **`Literal[...]`** in your schema — the cleanest way to restrict a field to a known enum (e.g. `Literal["billing", "bug"]`).

## Walk through `example.py`

| Part | What it shows |
|---|---|
| 1 | Extract a multi-field `MovieReview` from a paragraph of free text. |
| 2 | Classify three support tickets into a 3-field `SupportTicket`. |

## Run it

```bash
uv run python -m lessons.04_structured_output.example
```

## Debug it

Put a `breakpoint()` right after `llm.invoke(...)` and inspect `review`
— it's a real Pydantic model. Try `review.model_dump_json(indent=2)`.

## Try it yourself

- Add a `release_year: int | None` field to `MovieReview` and re-run.
- Change `priority` in `SupportTicket` to an `int` from 1–5 (use `Field(ge=1, le=5)`) and observe the model adapt.

## Next →

[Lesson 05 · Tools](../05_tools/README.md)
