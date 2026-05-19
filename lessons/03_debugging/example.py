"""Lesson 03 · Debugging — ipdb, breakpoints, listeners, LangSmith preview.

Run normally:
    uv run python -m lessons.03_debugging.example

Run with ipdb at every `breakpoint()`:
    PYTHONBREAKPOINT=ipdb.set_trace uv run python -m lessons.03_debugging.example

Inside ipdb (most useful commands):
    n      next line
    s      step into
    c      continue
    p x    print x
    pp x   pretty-print x
    w      where (stack trace)
    l      list source around current line
    u/d    move up/down the stack
    !x=1   execute arbitrary Python (mutate state)
    q      quit
"""

import asyncio
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tracers.schemas import Run

from shared import get_llm
from shared.pretty import console, section


def with_breakpoint() -> None:
    """A planted `breakpoint()` drops you into ipdb when PYTHONBREAKPOINT=ipdb.set_trace."""
    section("Part 1 · breakpoint() + ipdb")

    chain = (
        ChatPromptTemplate.from_template("Translate {text!r} into pirate English.")
        | get_llm()
        | StrOutputParser()
    )

    payload = {"text": "Where is the nearest library?"}

    # Try: pp chain  /  pp payload  /  n  to step over the next line
    breakpoint()

    result = chain.invoke(payload)
    console.print(result)


def with_listeners() -> None:
    """Listeners are the non-interactive way to peek inside a Runnable.

    They fire on `on_start` / `on_end` / `on_error`. Use them when you want
    to log every step of a chain without stopping execution.
    """
    section("Part 2 · Runnable.with_listeners (non-interactive)")

    def on_start(run: Run, config: RunnableConfig | None = None) -> None:
        console.print(f"[dim]→ start[/]  [yellow]{run.name}[/]  inputs={run.inputs}")

    def on_end(run: Run, config: RunnableConfig | None = None) -> None:
        console.print(f"[dim]← end  [/]  [yellow]{run.name}[/]  outputs={run.outputs}")

    chain = (
        ChatPromptTemplate.from_template("Reverse the word: {word}")
        | get_llm()
        | StrOutputParser()
    ).with_listeners(on_start=on_start, on_end=on_end)

    chain.invoke({"word": "octopus"})


async def async_breakpoint() -> None:
    """ipdb works in async too — `await` resolves as expected inside the prompt."""
    section("Part 3 · debugging async code")

    chain = (
        ChatPromptTemplate.from_template("Say hi in {language}.")
        | get_llm()
        | StrOutputParser()
    )

    breakpoint()  # try `await chain.ainvoke({'language': 'French'})` from the prompt

    out = await chain.ainvoke({"language": "French"})
    console.print(out)


def langsmith_preview() -> None:
    """LangSmith is the production analogue of ipdb — every Runnable run is captured.

    Enable it by setting in your .env:
        LANGSMITH_TRACING=true
        LANGSMITH_API_KEY=ls_...

    Once enabled, every chain you run becomes a clickable trace in the
    LangSmith UI. That's how you 'debug' in production where you can't
    drop a breakpoint.
    """
    section("Part 4 · LangSmith (the prod-grade debugger)")

    console.print(
        "If LANGSMITH_TRACING=true in your .env, every run above appeared as a trace "
        "in your LangSmith project. Open https://smith.langchain.com/ to inspect "
        "inputs, outputs, latency, and token counts for each step."
    )


def main() -> None:
    with_breakpoint()
    with_listeners()
    asyncio.run(async_breakpoint())
    langsmith_preview()


if __name__ == "__main__":
    main()
