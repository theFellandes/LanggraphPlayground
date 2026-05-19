"""Lesson 19 · Guardrails — middleware + decorators + judge node.

A single banking-support agent protected by FOUR guardrail axes
(input / output / tool / conversation) implemented via THREE patterns:

  1. **Middleware**       — PIIMiddleware (prebuilt) + custom CostCap (subclass AgentMiddleware)
  2. **Tool decorators**  — @validate_tool_args wraps a @tool to do arg checks before it runs
  3. **Layer-as-node**    — a Pydantic schema validates the agent's *final* answer

The script runs the agent on three inputs:
  - a clean, valid request                → all guardrails pass
  - a request with embedded PII           → PIIMiddleware redacts before the model sees it
  - a refund above the policy cap         → @validate_tool_args refuses the tool call

Run:
    uv run python -m lessons.19_guardrails.example
"""

from __future__ import annotations

from functools import wraps
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    PIIMiddleware,
)
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

from shared import get_llm
from shared.pretty import console, print_messages, section


# ───────────────────────────────────────────────────────────────────────────
# 1 · Tool guardrail — a *decorator* that validates args before the tool runs.
# ───────────────────────────────────────────────────────────────────────────
def validate_tool_args(check: Callable[..., None]) -> Callable:
    """Decorator: run `check(**kwargs)` before the tool body.

    `check` should raise `ValueError` (or any Exception) to reject. Anything
    that escapes here becomes a ToolMessage with `status="error"` that the
    model can read and react to — the agent doesn't crash.
    """
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            check(**kwargs)
            return fn(*args, **kwargs)
        return wrapper
    return deco


# Policy: refunds above $200 are not allowed without an explicit approval flow.
REFUND_CAP = 200.0


def _refund_policy(order_id: str, amount: float, **_) -> None:
    if not order_id:
        raise ValueError("order_id is required")
    if amount <= 0:
        raise ValueError(f"amount must be positive (got {amount})")
    if amount > REFUND_CAP:
        raise ValueError(
            f"REFUND_BLOCKED: amount ${amount:.2f} exceeds the ${REFUND_CAP:.0f} "
            "auto-approve cap. Escalate to a human."
        )


@tool
@validate_tool_args(_refund_policy)
def process_refund(order_id: str, amount: float, reason: str = "") -> str:
    """Refund the order. Refunds above $200 require human approval."""
    return f"OK — refunded ${amount:.2f} for order {order_id}. Reason: {reason or 'n/a'}"


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by id."""
    fake = {
        "ABC-1": "order ABC-1: $42 — shoes — shipped",
        "ABC-2": "order ABC-2: $250 — laptop — delivered",
    }
    return fake.get(order_id, f"no order matching {order_id}")


# ───────────────────────────────────────────────────────────────────────────
# 2 · Conversation guardrail — a *custom middleware* enforces a cost cap.
# ───────────────────────────────────────────────────────────────────────────
class CostCap(AgentMiddleware):
    """Hard cap on model calls per run. Demonstrates wrap_model_call."""

    def __init__(self, max_calls: int = 4) -> None:
        self.max_calls = max_calls
        self._calls = 0

    def wrap_model_call(self, request: ModelRequest, handler):
        self._calls += 1
        if self._calls > self.max_calls:
            raise RuntimeError(
                f"CostCap tripped: more than {self.max_calls} model calls."
            )
        console.print(f"  [dim]· model call {self._calls}/{self.max_calls}[/]")
        return handler(request)


# ───────────────────────────────────────────────────────────────────────────
# 3 · Output guardrail — a Pydantic schema validates the *final* answer.
#     (This is the "layer-as-node" pattern, run after the agent finishes.)
# ───────────────────────────────────────────────────────────────────────────
class FinalAnswer(BaseModel):
    summary:        str  = Field(min_length=4, max_length=400)
    contains_pii:   bool = Field(description="True iff PII is present in the summary.")
    action_taken:   str  = Field(description="Plain-text summary of the side effect.")


def judge_output(raw: str) -> FinalAnswer:
    """Have the LLM coerce the final answer into a typed schema.

    A judge node is the cleanest place to enforce: shape, length,
    presence/absence of forbidden content. If parsing fails, raise —
    a real app would log + retry + fall back to a safe canned reply.
    """
    llm = get_llm().with_structured_output(FinalAnswer)
    return llm.invoke(
        "Rewrite the assistant's reply as a strict JSON object matching the "
        "schema. If the original contains anything that looks like PII (emails, "
        "credit cards, IDs), set contains_pii=true and DO NOT include the PII "
        "in the summary.\n\n"
        f"Original reply: {raw}"
    )


# ───────────────────────────────────────────────────────────────────────────
# Build the protected agent.
# ───────────────────────────────────────────────────────────────────────────
def build_agent():
    return create_agent(
        model=get_llm(),
        tools=[lookup_order, process_refund],
        system_prompt=(
            "You are a banking support assistant. You may look up orders and, "
            "for orders under $200, issue refunds. Be concise and never leak "
            "personal data back to the user."
        ),
        middleware=[
            # Input guardrail — redact obvious PII before the model sees it.
            PIIMiddleware(pii_type="email",       strategy="redact"),
            PIIMiddleware(pii_type="credit_card", strategy="block"),
            # Conversation guardrail — hard cap on model calls.
            CostCap(max_calls=6),
        ],
    )


# ───────────────────────────────────────────────────────────────────────────
# Run three scenarios.
# ───────────────────────────────────────────────────────────────────────────
SCENARIOS = [
    (
        "1 · clean request",
        "Hi! Can you tell me the status of order ABC-1, please?",
    ),
    (
        "2 · request containing PII (email)",
        "Refund order ABC-1 for $30 and email me at alice@example.com when done.",
    ),
    (
        "3 · refund above the policy cap",
        "Refund order ABC-2 for $250 — laptop arrived broken.",
    ),
]


def run_one(label: str, prompt: str) -> None:
    section(label)
    agent = build_agent()  # fresh state per scenario so CostCap counter resets

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    except Exception as e:
        console.print(f"[bold red]Agent raised:[/] {type(e).__name__}: {e}")
        return

    raw_reply = result["messages"][-1].content
    console.print(f"[bold]raw agent reply:[/]\n{raw_reply}\n")

    # Output guardrail — run the judge.
    try:
        judged = judge_output(raw_reply)
    except ValidationError as e:
        console.print(f"[bold red]judge rejected the reply:[/] {e}")
        return

    console.print(f"[bold green]judge → FinalAnswer[/]:")
    console.print(f"  summary       : {judged.summary}")
    console.print(f"  contains_pii  : {judged.contains_pii}")
    console.print(f"  action_taken  : {judged.action_taken}")


def main() -> None:
    section("Lesson 19 · guardrails (middleware + decorator + judge node)")
    for label, prompt in SCENARIOS:
        run_one(label, prompt)


if __name__ == "__main__":
    main()
