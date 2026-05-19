# Lesson 10 · `create_agent` (the LangChain v1 prebuilt)

## What you'll learn

- The new `create_agent(model, tools, system_prompt)` API in LangChain 1.x — the **canonical** way to build a tool-using agent
- What `create_agent` is hiding (compare to lesson 05's manual round-trip)
- The input shape — `{"messages": [...]}` — and the output shape — the full message history
- How to visualise the agent's underlying LangGraph graph

## Why it matters

In LangChain 1.x, `create_agent` *replaces* `create_react_agent` and
the older agent constructors. It's the recommended starting point for
any "model + tools" agent. Internally it builds a LangGraph state
graph — so once you understand lessons 08 and 09, you also understand
what's happening under the hood here.

## Key concepts

- **`create_agent(model, tools, system_prompt=...)`** — returns a compiled LangGraph agent (a Runnable with `.invoke / .stream / .ainvoke`).
- **Input format** — `{"messages": [{"role": "user", "content": "..."}]}` or a list of `BaseMessage` objects.
- **Output format** — `{"messages": [...]}` — the full conversation including model turns, tool calls, and tool results.
- **Under the hood** — `create_agent` wires together `MessagesState`, a model node, a `ToolNode`, and a conditional edge based on whether the last message contains tool calls. You'll meet the same parts in lessons 08–09 in unwrapped form.

## Walk through `example.py`

1. Define three `@tool` functions (arithmetic + a stub knowledge base).
2. One call to `create_agent(...)` produces the compiled agent.
3. Print the agent's graph with `agent.get_graph().draw_ascii()` — you'll see the model node, the tool node, and the conditional edge between them.
4. Invoke with a single user message; print the full message trace to see the model call multiple tools, receive results, and synthesise a final answer.

## Run it

```bash
uv run python -m lessons.10_create_agent.example
```

## Debug it

Put `breakpoint()` right before `agent.invoke(...)`. Try at the prompt:

```text
ipdb> pp agent.get_graph().nodes
```

This shows you the node names `create_agent` set up for you — exactly
the building blocks you'll customise with middleware in lesson 11.

## Try it yourself

- Add a tool that intentionally raises an exception. Run the agent. Note how it recovers (or doesn't) — this is the kind of thing middleware (lesson 11) lets you control.
- Replace the question with one that needs no tools — confirm the agent answers directly without a tool round-trip.

## Next →

[Lesson 11 · Agent middleware](../11_agent_middleware/README.md)
