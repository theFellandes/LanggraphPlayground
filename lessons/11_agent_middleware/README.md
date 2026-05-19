# Lesson 11 · `create_agent` middleware

## What you'll learn

- LangChain v1's **middleware system** for `create_agent`
- The six hook points and when each fires:
  - `before_agent` · `before_model` · `wrap_model_call` · `wrap_tool_call` · `after_model` · `after_agent`
- How to write a **custom middleware** by subclassing `AgentMiddleware`
- How to use prebuilt middleware (e.g. `summarization_middleware`, `pii_redaction_middleware`, `human_in_the_loop_middleware`)
- Why middleware is composable — pass several in a list, they layer cleanly

## Why it matters

Middleware is the right place for cross-cutting concerns that don't
belong inside any one tool: logging, cost caps, prompt injection
defences, PII redaction, conversation summarisation, and HITL
approvals. Before middleware, each of those required rebuilding the
agent loop by hand.

## Key concepts

- **Middleware** — a class that subclasses `AgentMiddleware` and overrides any subset of the six hooks. Hooks receive the current state and a runtime handle.
- **`wrap_model_call(request, handler)`** — the most powerful hook. You see the outbound request, decide whether and how to call `handler(request)`, and can return any response — including a cached one.
- **`wrap_tool_call(call, handler)`** — same shape but for tool execution. Use it for retries, timeouts, or to swap a tool's implementation conditionally.
- **Composition** — `middleware=[a, b, c]` runs them in declaration order before each hook and reverse order after each hook (think onion).
- **Prebuilt middlewares** — `SummarizationMiddleware`, `HumanInTheLoopMiddleware`, `PIIMiddleware`, `ModelCallLimitMiddleware`, `ToolRetryMiddleware`, and more ship with LangChain. Drop them in instead of writing your own where they fit.

## Walk through `example.py`

1. **`LoggingMiddleware`** — implements `before_agent`, `before_model`, `after_model`, `after_agent` to print a one-liner at each. Read the output and you'll see the agent loop spelled out.
2. **`CostGuardrail`** — implements `wrap_model_call`. Counts calls and raises if you exceed `max_calls`. Demonstrates the *full-control* pattern.
3. Both are passed via `middleware=[...]` to `create_agent`. The agent solves a two-step math problem and you see every hook fire.

## Run it

```bash
uv run python -m lessons.11_agent_middleware.example
```

You should see the dim trace lines (`┌── agent starting`, `│ before_model`, etc.) interleaved with the printed messages.

## Debug it

Put `breakpoint()` inside `LoggingMiddleware.after_model` and inspect
`state["messages"][-1]` — that's the model's most recent reply, before
the agent loop decides what to do next.

## When to use what

| Goal | Mechanism |
|---|---|
| Log every step | `before_model` / `after_model` |
| Cache or short-circuit model calls | `wrap_model_call` |
| Add a per-tool retry policy | `wrap_tool_call` |
| Cap total cost / call count | `wrap_model_call` (count + raise) |
| Redact PII before model sees prompt | prebuilt `PIIMiddleware(pii_type="email", strategy="redact")` |
| Summarise long chat history | prebuilt `SummarizationMiddleware(model=..., trigger=("tokens", 1500))` |
| Pause for human approval before risky tools | prebuilt `HumanInTheLoopMiddleware(interrupt_on={"my_tool": True})` (and see lesson 13) |

## Try it yourself

- Add a prebuilt: `from langchain.agents.middleware import SummarizationMiddleware` and append `SummarizationMiddleware(model=get_llm(), trigger=("tokens", 1500))` to the list. Now ask the agent a 10-turn conversation and watch it auto-summarise.
- Write a `wrap_tool_call` middleware that retries any tool that raises an exception, up to 2 attempts.

## Next →

[Lesson 12 · Persistence](../12_persistence/README.md)
