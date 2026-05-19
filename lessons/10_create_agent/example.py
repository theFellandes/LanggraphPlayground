"""Lesson 10 · LangChain v1 `create_agent` — the new standard prebuilt agent.

Recall the manual tool-call round-trip from lesson 05. This lesson
shows what `create_agent` replaces it with: a single function call
that returns a compiled LangGraph agent ready to invoke.

Run:
    uv run python -m lessons.10_create_agent.example
"""

from langchain.agents import create_agent
from langchain_core.tools import tool

from shared import get_llm
from shared.pretty import console, print_messages, section


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def lookup_city_population(city: str) -> str:
    """Stub knowledge-base lookup. Returns a string."""
    fake_db = {
        "tokyo":    "Tokyo: ~13.96 million",
        "istanbul": "Istanbul: ~15.84 million",
        "lagos":    "Lagos: ~15.39 million",
    }
    return fake_db.get(city.lower(), f"No data for {city}.")


def main() -> None:
    section("Lesson 10 · `create_agent` (LangChain v1)")

    # One call. Compiled agent comes back ready to invoke.
    agent = create_agent(
        model=get_llm(),
        tools=[add, multiply, lookup_city_population],
        system_prompt=(
            "You are a careful research assistant. Use tools whenever they help. "
            "Show your reasoning briefly, then give a final answer."
        ),
    )

    console.print(agent.get_graph().draw_ascii())

    question = (
        "What's (17 + 25) * 3, and what's the population of Istanbul minus the "
        "population of Lagos (in millions)?"
    )

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    section("Full message trace")
    print_messages(result["messages"])


if __name__ == "__main__":
    main()
