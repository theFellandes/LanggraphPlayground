"""Lesson 00 · Setup — your first LLM call.

Run:
    uv run python -m lessons.00_setup.example
"""

from shared import get_llm
from shared.pretty import console, section


def main() -> None:
    section("Lesson 00 · Setup")

    llm = get_llm()
    console.print(f"Provider in use: [bold cyan]{llm.__class__.__name__}[/]")
    console.print(f"Model:           [bold cyan]{getattr(llm, 'model', '<unknown>')}[/]\n")

    response = llm.invoke("In one sentence, what is LangGraph?")
    console.print("[bold green]Model said:[/]")
    console.print(response.content)


if __name__ == "__main__":
    main()
