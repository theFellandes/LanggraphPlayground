"""Sample pytest for lesson 08.

Shows the canonical pattern for testing a StateGraph:
  1. Build the graph with a FAKE LLM so tests are deterministic and cheap.
  2. Assert on the final state — not on the model's prose.

Note: lesson folders start with digits (e.g. `08_langgraph_basics`),
which is fine for `python -m lessons.08_…` on the CLI but NOT a valid
Python identifier in `import` statements. So we use `importlib` here.

Run:
    uv run --extra dev pytest tests/test_lesson_08_graph.py -v
"""

import importlib
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

lesson = importlib.import_module("lessons.08_langgraph_basics.example")
build_graph = lesson.build_graph


def _fake_llm(responses: list[str]) -> FakeListChatModel:
    return FakeListChatModel(responses=responses)


def test_graph_runs_both_nodes_in_order() -> None:
    fake = _fake_llm(["A short intro paragraph.", "- bullet one\n- bullet two"])

    with patch.object(lesson, "get_llm", return_value=fake):
        graph = build_graph()
        out = graph.invoke({"topic": "fake topic"})

    assert out["draft"] == "A short intro paragraph."
    assert out["critique"].startswith("- bullet")


def test_graph_topology() -> None:
    """The compiled graph has exactly the nodes and edges we declared."""
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    # `__start__` / `__end__` are LangGraph sentinels.
    assert {"draft", "critique"}.issubset(nodes)
