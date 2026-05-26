"""Lesson 36 · The AI engineering library landscape — opt-in demos.

Each demo is gated on whether its library is installed. The script
tells you the `uv add ...` command if it's not.

Run:
    uv run python -m lessons.36_library_landscape.example
    uv run python -m lessons.36_library_landscape.example --instructor
    uv run python -m lessons.36_library_landscape.example --semantic-router
    uv run python -m lessons.36_library_landscape.example --marvin
    uv run python -m lessons.36_library_landscape.example --litellm
"""

from __future__ import annotations

import argparse

from shared.pretty import console, section


def _need(pkg: str, install: str) -> bool:
    """Return True if `pkg` is importable; print install hint otherwise."""
    try:
        __import__(pkg)
        return True
    except ImportError:
        console.print(f"[yellow]missing[/] {pkg!r} — install with: [bold]{install}[/]")
        return False


def demo_instructor() -> None:
    section("Instructor · Pydantic + retries")
    if not _need("instructor", "uv add instructor anthropic"):
        return

    import instructor
    from anthropic import Anthropic
    from pydantic import BaseModel, Field

    class SupportTicket(BaseModel):
        category: str = Field(description="refund | policy | technical | other")
        severity: int = Field(ge=1, le=5)
        needs_human: bool

    client = instructor.from_anthropic(Anthropic())
    out = client.messages.create(
        model="claude-haiku-4-5",
        response_model=SupportTicket,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": "Hello, I would like to be refunded for my $250 broken headphones order #99.",
        }],
    )
    console.print(out)


def demo_semantic_router() -> None:
    section("semantic-router · fast intent routing, no LLM")
    if not _need("semantic_router", "uv add semantic-router"):
        return

    from semantic_router import Route
    from semantic_router.routers import SemanticRouter
    from semantic_router.encoders import FastEmbedEncoder

    routes = [
        Route(name="refund",   utterances=["I want a refund", "Refund my order",
                                            "Can I get my money back?", "Cancel and refund"]),
        Route(name="policy",   utterances=["How many PTO days?", "What's the remote-work policy?",
                                            "Can I expense lunch?", "Holiday calendar"]),
        Route(name="technical",utterances=["My app crashes", "Site is down", "Can't log in",
                                            "Bug report"]),
    ]
    rl = SemanticRouter(encoder=FastEmbedEncoder(), routes=routes, auto_sync="local")

    queries = [
        "I'd like my money back please",
        "How much paid time off do I get per year?",
        "The page won't load",
        "What's the meaning of life?",
    ]
    for q in queries:
        result = rl(q)
        console.print(f"  {q[:50]:50}  → [bold]{result.name or '(no match)'}[/]")


def demo_marvin() -> None:
    section("Marvin · declarative AI functions")
    if not _need("marvin", "uv add marvin"):
        return

    import marvin

    @marvin.fn
    def extract_amount(text: str) -> float:
        """Extract the dollar amount from the text, as a float."""

    @marvin.fn
    def categorise(text: str) -> str:
        """Return one of: refund, policy, technical, other."""

    samples = [
        "I want to be refunded $250 for my broken order",
        "How many PTO days do new hires get?",
        "Login button doesn't work",
    ]
    for s in samples:
        cat = categorise(s)
        amt = extract_amount(s) if cat == "refund" else None
        console.print(f"  {s[:50]:50}  → cat={cat:10}  amt={amt}")


def demo_litellm() -> None:
    section("LiteLLM · one provider-agnostic call")
    if not _need("litellm", "uv add litellm"):
        return

    from litellm import completion

    # The point: same function call, different providers.
    for model in ("anthropic/claude-haiku-4-5", "openai/gpt-4o-mini"):
        try:
            r = completion(
                model=model,
                messages=[{"role": "user", "content": "Reply with exactly: pong"}],
                max_tokens=10,
            )
            console.print(f"  {model:30}  → {r.choices[0].message.content[:50]!r}")
        except Exception as e:
            console.print(f"  {model:30}  → [yellow]skipped:[/] {type(e).__name__}")


def demo_pydantic_ai() -> None:
    section("Pydantic AI · typed agents")
    if not _need("pydantic_ai", "uv add pydantic-ai"):
        return

    from pydantic_ai import Agent

    agent = Agent(
        "anthropic:claude-haiku-4-5",
        system_prompt="You are a helpful assistant. Be concise.",
    )
    result = agent.run_sync("What's the capital of Belgium?")
    console.print(f"  output: {result.output}")


def demo_dspy_smoke() -> None:
    section("DSPy · declarative signature (smoke test, no compile)")
    if not _need("dspy", "uv add dspy"):
        return

    import dspy

    dspy.settings.configure(lm=dspy.LM("anthropic/claude-haiku-4-5"))

    class CitedAnswer(dspy.Signature):
        """Answer the question with an inline citation."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        citation: str = dspy.OutputField()

    qa = dspy.Predict(CitedAnswer)
    out = qa(question="What's the chemical symbol for gold?")
    console.print(f"  answer:   {out.answer}")
    console.print(f"  citation: {out.citation}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instructor", action="store_true")
    parser.add_argument("--semantic-router", action="store_true")
    parser.add_argument("--marvin", action="store_true")
    parser.add_argument("--litellm", action="store_true")
    parser.add_argument("--pydantic-ai", action="store_true")
    parser.add_argument("--dspy", action="store_true")
    args = parser.parse_args()

    selected = []
    if args.instructor:      selected.append(demo_instructor)
    if args.semantic_router: selected.append(demo_semantic_router)
    if args.marvin:          selected.append(demo_marvin)
    if args.litellm:         selected.append(demo_litellm)
    if args.pydantic_ai:     selected.append(demo_pydantic_ai)
    if args.dspy:            selected.append(demo_dspy_smoke)
    if not selected:
        selected = [demo_instructor, demo_semantic_router, demo_marvin,
                    demo_litellm, demo_pydantic_ai, demo_dspy_smoke]

    for fn in selected:
        try:
            fn()
        except Exception as e:
            console.print(f"[red]demo failed:[/] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
