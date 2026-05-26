"""Lesson 28 · Dynamic prompting with Jinja2.

Four demos. Demos 1-3 are offline. Demo 4 stubs the LLM so it also runs
without an API key.

Run:
    uv run python -m lessons.28_dynamic_prompting.example                  # all
    uv run python -m lessons.28_dynamic_prompting.example --inline
    uv run python -m lessons.28_dynamic_prompting.example --file-based
    uv run python -m lessons.28_dynamic_prompting.example --inheritance
    uv run python -m lessons.28_dynamic_prompting.example --dynamic
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from langchain_core.prompts import PromptTemplate

from shared.pretty import console, section

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Build the Jinja environment once. StrictUndefined makes typos loud
# (you get UndefinedError at render time, not a silent empty string).
env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    autoescape=select_autoescape(disabled_extensions=("j2",)),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=StrictUndefined,
)


# --- Demo 1 · Inline Jinja in a LangChain PromptTemplate --------------------
def demo_inline() -> None:
    section("1 · Inline Jinja in PromptTemplate")

    tpl = PromptTemplate.from_template(
        """You are answering on behalf of {{ company }}.
{% if vip %}This user is a VIP — be extra-attentive.{% endif %}

Retrieved context:
{% for doc in docs %}
[{{ loop.index }}] {{ doc.title }} — {{ doc.snippet | truncate(120) }}
{% endfor %}

Question: {{ question }}
""",
        template_format="jinja2",
    )

    rendered = tpl.invoke(
        {
            "company": "Acme",
            "vip": True,
            "docs": [
                {"title": "PTO Policy", "snippet": "Full-time employees receive 20 PTO days per year, accruing monthly from the start date."},
                {"title": "Remote Work", "snippet": "All employees may work remotely up to 3 days/week with manager approval."},
            ],
            "question": "How many PTO days do I get?",
        }
    )
    console.print(rendered.to_string())


# --- Demo 2 · File-based templates ------------------------------------------
@dataclass
class FakeTool:
    name: str
    description: str


def demo_file_based() -> None:
    section("2 · File-based templates with FileSystemLoader")

    tools = [
        FakeTool("search_web", "Search the web. Use it sparingly; rate-limited."),
        FakeTool("cite", "Record a (claim, source) pair for the writer."),
    ]

    out = env.get_template("agents/researcher.j2").render(
        agent_name="Researcher",
        company="Acme",
        tools=tools,
        locale="en-US",
        vip=False,
        conversation_summary="",
    )
    console.print(out)


# --- Demo 3 · Inheritance + locale switching --------------------------------
def demo_inheritance() -> None:
    section("3 · Template inheritance + locale + VIP block")

    common = dict(
        agent_name="Researcher",
        company="Acme",
        tools=[FakeTool("search_web", "Search.")],
        vip=True,
        tier="enterprise",
        conversation_summary="User asked about refund policy last week; got a $200 refund approved.",
    )

    for locale in ("en-US", "tr-TR", "de-DE"):
        console.rule(f"[bold]locale = {locale}[/]")
        out = env.get_template("agents/researcher.j2").render(locale=locale, **common)
        console.print(out)


# --- Demo 4 · Dynamic per-call system_prompt -------------------------------
def demo_dynamic() -> None:
    section("4 · Dynamic system_prompt — re-rendered every turn")

    # We simulate what `create_agent(system_prompt=callable)` does internally:
    # render the system message *just before* each LLM call, so it sees the
    # current state.

    template = env.get_template("agents/researcher.j2")

    def system_for_turn(state: dict[str, Any]) -> str:
        return template.render(
            agent_name="Researcher",
            company="Acme",
            tools=state["tools"],
            locale=state.get("locale", "en-US"),
            vip=state.get("vip", False),
            conversation_summary=_summarise(state["messages"]),
        )

    def _summarise(messages: list[str]) -> str:
        if not messages:
            return ""
        return f"User has sent {len(messages)} messages. Last: {messages[-1][:80]!r}"

    state = {
        "tools": [FakeTool("search_web", "Search.")],
        "vip": False,
        "messages": [],
        "locale": "en-US",
    }

    # Turn 1: empty history, free tier.
    console.rule("[bold]turn 1 (no history, free tier)[/]")
    console.print(system_for_turn(state))

    # Turn 2: VIP, conversation history, switched locale.
    state["messages"].extend(
        ["Hi, I need help with a refund.", "It's about order #ABC-99, $250."]
    )
    state["vip"] = True
    state["locale"] = "tr-TR"
    console.rule("[bold]turn 2 (VIP, 2 prior messages, Turkish)[/]")
    console.print(system_for_turn(state))


# --- entry point ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inline", action="store_true")
    parser.add_argument("--file-based", action="store_true")
    parser.add_argument("--inheritance", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    selected = []
    if args.inline:       selected.append(demo_inline)
    if args.file_based:   selected.append(demo_file_based)
    if args.inheritance:  selected.append(demo_inheritance)
    if args.dynamic:      selected.append(demo_dynamic)
    if not selected:
        selected = [demo_inline, demo_file_based, demo_inheritance, demo_dynamic]

    for fn in selected:
        fn()


if __name__ == "__main__":
    main()
