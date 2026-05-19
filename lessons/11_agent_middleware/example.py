"""Lesson 11 · `create_agent` middleware.

LangChain v1's middleware system lets you intercept the agent loop at
defined points without rewriting the whole agent.

The hooks you can implement:
  - before_agent       runs once when the agent starts
  - before_model       runs before every model call
  - wrap_model_call    wraps the model call (you call `handler(request)` yourself)
  - wrap_tool_call     wraps each tool execution
  - after_model        runs after every model call (inspect/mutate the response)
  - after_agent        runs once when the agent finishes

Run:
    uv run python -m lessons.11_agent_middleware.example
"""

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.tools import tool

from shared import get_llm
from shared.pretty import console, print_messages, section


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# ---------------------------------------------------------------------------
# 1 · A custom logging middleware — subclass `AgentMiddleware`.
# ---------------------------------------------------------------------------
class LoggingMiddleware(AgentMiddleware):
    """Print a one-liner at every hook so you can see the agent's loop."""

    def before_agent(self, state, runtime) -> None:
        console.print("[dim]┌── agent starting[/]")

    def before_model(self, state, runtime) -> None:
        n = len(state.get("messages", []))
        console.print(f"[dim]│   before_model     (messages so far: {n})[/]")

    def after_model(self, state, runtime) -> None:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            names = ", ".join(c["name"] for c in tool_calls)
            console.print(f"[dim]│   after_model      → tool_calls: {names}[/]")
        else:
            console.print("[dim]│   after_model      → final reply[/]")

    def after_agent(self, state, runtime) -> None:
        console.print("[dim]└── agent finished[/]")


# ---------------------------------------------------------------------------
# 2 · A `wrap_model_call` middleware — full control over the model invocation.
# ---------------------------------------------------------------------------
class CostGuardrail(AgentMiddleware):
    """Cap the number of model calls per run."""

    def __init__(self, max_calls: int = 4) -> None:
        self.max_calls = max_calls
        self.calls = 0

    def wrap_model_call(self, request: ModelRequest, handler):
        self.calls += 1
        if self.calls > self.max_calls:
            raise RuntimeError(
                f"CostGuardrail tripped: more than {self.max_calls} model calls."
            )
        console.print(f"[dim]│   wrap_model_call  ({self.calls}/{self.max_calls})[/]")
        return handler(request)


def main() -> None:
    section("Lesson 11 · custom middleware + cost guardrail")

    agent = create_agent(
        model=get_llm(),
        tools=[add, divide],
        system_prompt="Solve math problems step by step. Use tools when helpful.",
        middleware=[LoggingMiddleware(), CostGuardrail(max_calls=5)],
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is (12 + 8) / 4, and then that result + 17?"}]}
    )

    section("final messages")
    print_messages(result["messages"])


if __name__ == "__main__":
    main()
