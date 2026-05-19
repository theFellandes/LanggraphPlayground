"""Lesson 05 · Tools — `@tool`, `bind_tools`, the tool-call round-trip.

Run:
    uv run python -m lessons.05_tools.example
"""

from datetime import datetime
from typing import Annotated

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from shared import get_llm
from shared.pretty import console, print_messages, section


@tool
def current_time(timezone: str = "UTC") -> str:
    """Return the current ISO timestamp. `timezone` is informational only."""
    return f"{datetime.utcnow().isoformat()}Z ({timezone})"


@tool
def add(a: Annotated[int, "first number"], b: Annotated[int, "second number"]) -> int:
    """Add two integers and return the sum."""
    return a + b


@tool
def lookup_city_population(city: str) -> str:
    """Stub knowledge-base lookup — pretend this hits a real DB."""
    fake_db = {
        "tokyo":    "Tokyo: ~13.96 million",
        "istanbul": "Istanbul: ~15.84 million",
        "lagos":    "Lagos: ~15.39 million",
    }
    return fake_db.get(city.lower(), f"No data for {city}.")


TOOLS = [current_time, add, lookup_city_population]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def manual_round_trip() -> None:
    """Step through one tool-call round-trip by hand.

    This is *exactly* what `create_agent` (lesson 10) automates for you.
    Doing it once by hand makes that lesson click.
    """
    section("Tool-call round-trip, the manual way")

    llm_with_tools = get_llm().bind_tools(TOOLS)

    history = [HumanMessage("What's the population of Istanbul, and what's 17 + 25?")]

    # --- step 1: model decides which tools to call -----------------------
    ai = llm_with_tools.invoke(history)
    history.append(ai)
    console.print("[bold]After model's first turn:[/]")
    print_messages(history)

    if not getattr(ai, "tool_calls", None):
        console.print("[yellow]Model chose not to call any tool — nothing more to do.[/]")
        return

    # --- step 2: run each tool the model asked for -----------------------
    for call in ai.tool_calls:
        tool_fn = TOOLS_BY_NAME[call["name"]]
        result = tool_fn.invoke(call["args"])
        history.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # --- step 3: feed tool outputs back so the model can finalise --------
    final = llm_with_tools.invoke(history)
    history.append(final)

    section("After feeding tool outputs back")
    print_messages(history)


def main() -> None:
    manual_round_trip()


if __name__ == "__main__":
    main()
