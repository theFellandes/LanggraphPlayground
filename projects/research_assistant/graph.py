"""Multi-agent research assistant.

Architecture (supervisor pattern, lesson 16):

    user
     │
     ▼
   supervisor
   ┌────┼────┐
   ▼    ▼    ▼
 researcher writer critic
   (Tavily) (LLM) (LLM)

The supervisor delegates: researcher gathers sources, writer drafts
the report, critic checks for missing citations. The supervisor
synthesises the final Markdown output.

Run:
    uv run python -m projects.research_assistant.graph "your question here"
"""

from __future__ import annotations

import sys

from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph_supervisor import create_supervisor

from shared import get_llm, settings
from shared.pretty import console, print_messages, section


def _build_research_tool():
    if not settings.tavily_api_key:
        raise RuntimeError(
            "TAVILY_API_KEY missing in .env. Get one from https://tavily.com/."
        )
    return TavilySearchResults(max_results=5, tavily_api_key=settings.tavily_api_key)


@tool
def cite(source_url: str, claim: str) -> str:
    """Record a (claim, source) pair the writer can later cite."""
    return f"[cited] {claim} — {source_url}"


def build_graph():
    research = _build_research_tool()

    researcher = create_agent(
        model=get_llm(),
        tools=[research, cite],
        system_prompt=(
            "You are a research analyst. Search the web for the user's topic, "
            "extract 3-5 specific claims with URLs, and call `cite(url, claim)` "
            "for each. Return a tidy bullet list of cited claims."
        ),
        name="researcher",
    )

    writer = create_agent(
        model=get_llm(),
        tools=[],
        system_prompt=(
            "You are a science writer. Turn the cited claims you receive into a "
            "well-structured Markdown report with headings and inline citations "
            "in (Source: <url>) format. Be concise — under 400 words."
        ),
        name="writer",
    )

    critic = create_agent(
        model=get_llm(),
        tools=[],
        system_prompt=(
            "You are an editor. Read the draft report. Reply with either "
            "'APPROVED' if every claim has a citation, or a short bullet list of "
            "missing citations / unsupported claims for the writer to fix."
        ),
        name="critic",
    )

    return create_supervisor(
        agents=[researcher, writer, critic],
        model=get_llm(),
        prompt=(
            "You coordinate a small research team. Workflow:\n"
            "  1. Delegate to `researcher` to gather cited claims.\n"
            "  2. Delegate to `writer` to draft a report from those claims.\n"
            "  3. Delegate to `critic` to review. If critic returns issues, "
            "loop back to writer with the feedback.\n"
            "  4. When critic returns APPROVED, present the final report to the user."
        ),
    ).compile()


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What are the latest advances in fusion energy in 2026?"

    section(f"Research request: {question}")
    graph = build_graph()
    result = graph.invoke({"messages": [{"role": "user", "content": question}]})

    section("Full transcript")
    print_messages(result["messages"])

    section("Final report")
    console.print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
