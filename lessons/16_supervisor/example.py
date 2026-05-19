"""Lesson 16 · Supervisor — `langgraph-supervisor`.

One boss + many workers. The supervisor reads the user request and
delegates to the worker whose skills match. Workers can hand back
to the supervisor; the supervisor decides when to stop.

Run:
    uv run python -m lessons.16_supervisor.example
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph_supervisor import create_supervisor

from shared import get_llm
from shared.pretty import console, print_messages, section


# --- Worker 1: math --------------------------------------------------------
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


# --- Worker 2: travel ------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Return a (fake) weather snapshot for the given city."""
    return f"Weather in {city}: 22°C, mostly sunny."


@tool
def get_local_dish(city: str) -> str:
    """Return a (fake) signature dish."""
    fake = {"istanbul": "lahmacun", "tokyo": "ramen", "lima": "ceviche"}
    return fake.get(city.lower(), f"No data for {city}.")


# --- Supervisor ------------------------------------------------------------
def main() -> None:
    section("Lesson 16 · supervisor + two workers")

    # Build agents inside main so the module imports cleanly without API keys.
    math_agent = create_agent(
        model=get_llm(),
        tools=[add, multiply],
        system_prompt="You are a precise calculator. Use tools for all arithmetic.",
        name="math_expert",
    )
    travel_agent = create_agent(
        model=get_llm(),
        tools=[get_weather, get_local_dish],
        system_prompt="You answer travel questions about weather and local food.",
        name="travel_expert",
    )

    supervisor = create_supervisor(
        agents=[math_agent, travel_agent],
        model=get_llm(),
        prompt=(
            "You are a routing supervisor. Delegate the user's request to "
            "math_expert for arithmetic or travel_expert for travel questions. "
            "When you have a complete answer, reply directly to the user."
        ),
    ).compile()

    console.print(supervisor.get_graph().draw_ascii())

    result = supervisor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I'm flying to Istanbul next week. What's the weather, "
                        "and what's 3 × 17 in case I split a bill three ways?"
                    ),
                }
            ]
        }
    )

    section("full message trace")
    print_messages(result["messages"])


if __name__ == "__main__":
    main()
