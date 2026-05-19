"""Lesson 04 · Structured output — Pydantic + `with_structured_output`.

Run:
    uv run python -m lessons.04_structured_output.example
"""

from typing import Literal

from pydantic import BaseModel, Field

from shared import get_llm
from shared.pretty import console, section


class MovieReview(BaseModel):
    """A structured review extracted from free-text user feedback."""

    title: str = Field(description="Name of the movie being reviewed.")
    rating: int = Field(ge=1, le=10, description="Score from 1 (worst) to 10 (best).")
    sentiment: Literal["positive", "neutral", "negative"]
    pros: list[str] = Field(default_factory=list, description="Things the reviewer liked.")
    cons: list[str] = Field(default_factory=list, description="Things the reviewer disliked.")


def extract_review() -> None:
    section("Part 1 · single-shot extraction")

    llm = get_llm().with_structured_output(MovieReview)

    raw = (
        "I finally saw 'Dune: Part Two' last night. Visually stunning and the "
        "score is unreal, but the pacing in the second act dragged. Probably "
        "an 8 out of 10 — I'd watch it again, but I wouldn't queue up for a third."
    )
    review = llm.invoke(raw)

    console.print(f"[bold cyan]Title[/]    {review.title}")
    console.print(f"[bold cyan]Rating[/]   {review.rating}/10  ({review.sentiment})")
    console.print(f"[bold green]Pros[/]    {review.pros}")
    console.print(f"[bold red]Cons[/]     {review.cons}")


class SupportTicket(BaseModel):
    category: Literal["billing", "bug", "feature_request", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    one_line_summary: str


def classify_ticket() -> None:
    section("Part 2 · classification into a small schema")

    llm = get_llm().with_structured_output(SupportTicket)

    tickets = [
        "Your service charged me twice for the same plan last week. Please refund.",
        "It would be nice if dark mode was available on the mobile app.",
        "The export button on the dashboard does nothing in Firefox.",
    ]
    for t in tickets:
        out = llm.invoke(t)
        console.print(
            f"[dim]→[/] [{out.priority:>6}] [{out.category:>15}]  {out.one_line_summary}"
        )


def main() -> None:
    extract_review()
    classify_ticket()


if __name__ == "__main__":
    main()
