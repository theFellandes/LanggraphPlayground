"""Lesson 01 · Chat models — messages, prompt templates, roles.

Run:
    uv run python -m lessons.01_chat_models.example
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from shared import get_llm
from shared.pretty import console, print_message, section


def raw_messages() -> None:
    """Send a list of messages directly — no template."""
    section("Part 1 · raw messages")

    llm = get_llm()
    messages = [
        SystemMessage("You are a terse Renaissance polymath. Reply in one line."),
        HumanMessage("What is the best way to learn LangGraph?"),
    ]
    reply = llm.invoke(messages)
    for m in (*messages, reply):
        print_message(m)


def with_prompt_template() -> None:
    """A `ChatPromptTemplate` makes the prompt parameterised + reusable."""
    section("Part 2 · ChatPromptTemplate")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You translate English into {language}. Answer with the translation only."),
            ("human", "{text}"),
        ]
    )
    chain = prompt | get_llm()

    reply = chain.invoke({"language": "Turkish", "text": "I love stateful agents."})
    console.print(f"[bold green]Translation:[/] {reply.content}")


def with_message_history() -> None:
    """`MessagesPlaceholder` is how chat agents thread history into the prompt."""
    section("Part 3 · MessagesPlaceholder (conversation history)")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a friendly mentor for programmers."),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | get_llm()

    history = [
        HumanMessage("Hi, I'm new to AI agents."),
        # The model's previous reply would normally go here as an AIMessage.
    ]
    reply = chain.invoke(
        {"history": history, "question": "Where should I start: LCEL or LangGraph?"}
    )
    print_message(reply)


def main() -> None:
    raw_messages()
    with_prompt_template()
    with_message_history()


if __name__ == "__main__":
    main()
