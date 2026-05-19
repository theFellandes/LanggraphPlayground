"""Lesson 02 · LCEL chains — pipe, parallel, stream, batch, async.

Run:
    uv run python -m lessons.02_lcel_chains.example
"""

import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from shared import get_llm
from shared.pretty import console, section


def simple_pipe() -> None:
    section("Part 1 · the pipe operator")

    prompt = ChatPromptTemplate.from_template("Give me one fun fact about {topic}.")
    chain = prompt | get_llm() | StrOutputParser()

    console.print(chain.invoke({"topic": "octopuses"}))


def parallel_chain() -> None:
    """RunnableParallel fans-out the same input to multiple branches."""
    section("Part 2 · RunnableParallel (fan-out)")

    llm = get_llm()
    poem  = ChatPromptTemplate.from_template("Write a 2-line poem about {topic}.") | llm | StrOutputParser()
    joke  = ChatPromptTemplate.from_template("Tell a one-liner joke about {topic}.") | llm | StrOutputParser()
    facts = ChatPromptTemplate.from_template("Give one fact about {topic}.")        | llm | StrOutputParser()

    fan_out = RunnableParallel(poem=poem, joke=joke, fact=facts)
    result = fan_out.invoke({"topic": "Saturn"})

    for key, value in result.items():
        console.print(f"[bold cyan]{key}[/]: {value}")


def upstream_with_lambda() -> None:
    """RunnableLambda + RunnablePassthrough.assign lets you add fields mid-chain."""
    section("Part 3 · RunnablePassthrough.assign + RunnableLambda")

    word_count = RunnableLambda(lambda inputs: len(inputs["text"].split()))

    pipeline = (
        RunnablePassthrough.assign(words=word_count)
        | RunnableLambda(lambda inputs: f"{inputs['text']!r} has {inputs['words']} words.")
    )
    console.print(pipeline.invoke({"text": "LangGraph turns agents into graphs."}))


def streaming() -> None:
    """`.stream()` yields chunks as the model produces them."""
    section("Part 4 · streaming")

    chain = (
        ChatPromptTemplate.from_template("List {n} surprising uses of duct tape.")
        | get_llm()
        | StrOutputParser()
    )
    for chunk in chain.stream({"n": 5}):
        console.print(chunk, end="")
    console.print()  # final newline


def batching() -> None:
    """`.batch()` runs many inputs in parallel (provider permitting)."""
    section("Part 5 · batching")

    chain = (
        ChatPromptTemplate.from_template("In 5 words, define: {term}")
        | get_llm()
        | StrOutputParser()
    )
    answers = chain.batch(
        [{"term": t} for t in ("monad", "transformer", "eigenvalue")]
    )
    for term, ans in zip(("monad", "transformer", "eigenvalue"), answers):
        console.print(f"[cyan]{term}[/] → {ans}")


async def async_invoke() -> None:
    """`.ainvoke()` is the async twin of `.invoke()`."""
    section("Part 6 · async (ainvoke)")

    chain = ChatPromptTemplate.from_template("Greet me in {language}.") | get_llm() | StrOutputParser()
    out = await chain.ainvoke({"language": "Japanese"})
    console.print(out)


def main() -> None:
    simple_pipe()
    parallel_chain()
    upstream_with_lambda()
    streaming()
    batching()
    asyncio.run(async_invoke())


if __name__ == "__main__":
    main()
