# Lesson 05 · Tools

## What you'll learn

- The `@tool` decorator and what it produces (a `BaseTool` with a JSON schema generated from the type hints + docstring)
- `llm.bind_tools([t1, t2, ...])` — how to give the model a tool palette
- The full **tool-call round-trip**: model decides → you execute → you feed results back → model answers
- Why every later "agent" abstraction (`create_agent`, `ToolNode`, supervisors) is built on this primitive

## Why it matters

If you only learn one thing about LLM apps, learn the tool-call
round-trip. Every agent framework — `create_agent` in lesson 10,
ReAct, supervisor patterns in lesson 16 — is a fancier wrapper around
the four steps you'll see here.

## Key concepts

- **`@tool`** — wraps a function into a tool. Type hints become the JSON schema; the docstring becomes the tool description. The model sees both.
- **`Annotated[type, "description"]`** — per-argument descriptions, surfaced into the schema for each parameter.
- **`llm.bind_tools([...])`** — returns a new Runnable that will, on every call, include the tool definitions in the request so the model *can* decide to call them.
- **`AIMessage.tool_calls`** — a list of `{name, args, id}` dicts. Empty when the model didn't call any tool.
- **`ToolMessage(content, tool_call_id)`** — your answer to a single tool call. The `tool_call_id` matches the model's request back to the response.

## Walk through `example.py`

`manual_round_trip()` does the whole dance step-by-step so you see the
moving parts:

1. **Bind tools** to the model.
2. **First call** — the model returns an `AIMessage` whose `tool_calls` lists what it wants to run.
3. **Execute** each call locally and produce a `ToolMessage` per call.
4. **Second call** — feed the appended messages back; the model now has tool outputs and can write a final natural-language answer.

The three tools (`current_time`, `add`, `lookup_city_population`) are
deliberately tiny so the focus stays on the protocol.

## Run it

```bash
uv run python -m lessons.05_tools.example
```

## Debug it

Put `breakpoint()` between steps 1 and 2 and inspect `ai.tool_calls`.
Try `pp ai.tool_calls` — that list **is** the model's reasoning made
machine-readable.

## Try it yourself

- Add a `multiply` tool. Ask the model "what's (17 + 25) × 3?" and watch it chain calls.
- Have `lookup_city_population` raise an exception for unknown cities. Notice how the model recovers when you wrap the call site with `try/except`.

## Next →

[Lesson 06 · RAG basics](../06_rag_basics/README.md)
