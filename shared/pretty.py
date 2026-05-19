"""Rich-based printers so lesson output is readable.

Lessons import these to display messages, graph state, and tool calls
without rolling their own formatting.
"""

from __future__ import annotations

from typing import Any, Iterable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.rule import Rule

console = Console()

_ROLE_STYLE = {
    HumanMessage: ("Human",  "bold cyan"),
    AIMessage:    ("AI",     "bold green"),
    SystemMessage:("System", "bold magenta"),
    ToolMessage:  ("Tool",   "bold yellow"),
}


def print_message(msg: BaseMessage) -> None:
    label, style = next(
        ((lbl, st) for cls, (lbl, st) in _ROLE_STYLE.items() if isinstance(msg, cls)),
        (type(msg).__name__, "white"),
    )
    body = msg.content if isinstance(msg.content, str) else Pretty(msg.content)
    console.print(Panel(body, title=label, title_align="left", border_style=style))

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        for call in tool_calls:
            console.print(
                f"  [dim]→ tool_call[/]  [yellow]{call['name']}[/]"
                f"({Pretty(call.get('args', {}))})"
            )


def print_messages(messages: Iterable[BaseMessage]) -> None:
    for m in messages:
        print_message(m)


def print_state(label: str, state: Any) -> None:
    console.print(Rule(f"[bold]{label}[/]"))
    console.print(Pretty(state, expand_all=True))


def section(title: str) -> None:
    console.print()
    console.print(Rule(f"[bold bright_white]{title}[/]"))
