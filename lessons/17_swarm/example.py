"""Lesson 17 · Swarm — peer-to-peer handoffs.

Two agents that hand off directly to each other. No supervisor in the
middle.

    user → triage  ⇄  refunder

Run:
    uv run python -m lessons.17_swarm.example
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph_swarm import create_handoff_tool, create_swarm

from shared import get_llm
from shared.pretty import console, print_messages, section


@tool
def process_refund(order_id: str, amount: float) -> str:
    """Issue a refund. Pretend this hits a real payments API."""
    return f"Refunded ${amount:.2f} for order {order_id}."


# Handoff tools are just specially-shaped @tools that the swarm interprets.
hand_to_refunder = create_handoff_tool(
    agent_name="refunder",
    description="Transfer the conversation to the refunder agent.",
)
hand_to_triage = create_handoff_tool(
    agent_name="triage",
    description="Transfer the conversation back to the triage agent.",
)


def main() -> None:
    section("Lesson 17 · swarm with handoffs")

    # Build agents inside main so the module imports cleanly without API keys.
    triage = create_agent(
        model=get_llm(),
        tools=[hand_to_refunder],
        system_prompt=(
            "You triage customer-support requests. If the customer wants a refund, "
            "transfer to the refunder agent. Otherwise answer directly."
        ),
        name="triage",
    )

    refunder = create_agent(
        model=get_llm(),
        tools=[process_refund, hand_to_triage],
        system_prompt=(
            "You handle refund requests. Ask for any missing details (order id, "
            "amount), then call process_refund. Hand back to triage when done."
        ),
        name="refunder",
    )

    swarm = create_swarm(
        agents=[triage, refunder],
        default_active_agent="triage",
    ).compile()

    console.print(swarm.get_graph().draw_ascii())

    result = swarm.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Hi! I'd like a refund of $42 on order #ABC-123, please.",
                }
            ]
        }
    )

    section("full message trace")
    print_messages(result["messages"])


if __name__ == "__main__":
    main()
