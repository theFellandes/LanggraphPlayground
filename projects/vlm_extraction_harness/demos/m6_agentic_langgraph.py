"""Method 6 — Agentic extraction with a self-correcting LangGraph StateGraph.

Setup (uv only — never pip)::

    uv add langgraph langchain-google-genai
    uv sync --extra extraction          # pulls PyMuPDF for page rendering

What this method is
-------------------
A *real* LangGraph ``StateGraph`` that treats PDF extraction as an agentic loop
instead of a single shot:

    ingest  →  (Send fan-out: one worker per page)
            →  extract_page   (VLM via get_llm)
            →  self_check     (confidence heuristic)
            →  conditional retry: low-confidence pages are re-rendered at a
               higher DPI / alternate model and re-extracted, bounded by a
               shared ``asyncio.Semaphore`` so we never hammer the API
            →  assemble       (deferred fan-in: runs only after every page
               branch — including retries — has settled)

This maps to the course like so:
  * **Lesson 30 (fan-out / map-reduce)** — ``Send("extract_page", {...})`` spawns
    one branch per page; results merge through an ``operator.add`` list reducer.
  * **Lesson 27 (bounded concurrency)** — a module-level ``asyncio.Semaphore``
    caps how many VLM calls run at once across all fan-out branches.

Design constraints honoured here
--------------------------------
* The VLM is reached **only** through ``get_llm("google", ...)`` — never a raw
  google-genai / OpenAI SDK call. It inherits the project fallback chain.
* Heavy deps (``langgraph``, ``pymupdf``) are imported **lazily** inside the
  functions that need them, each guarded with an install hint.
* The graph is **importable and compilable with no API key**. Run it with
  ``--mock`` (or simply without ``GOOGLE_API_KEY``) and every page is "extracted"
  by a deterministic stub so you can watch the fan-out / retry / assemble flow.
"""

from __future__ import annotations

import asyncio
import base64
import operator
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

# --- repo wiring: run as a plain script ------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.llm import get_llm  # noqa: E402

# Reuse the exact rendering + prompt the rest of the harness uses.
from projects.vlm_extraction_harness.extractors import (  # noqa: E402
    GEMINI_PROMPT,
    render_page_png,
)

# ---------------------------------------------------------------------------
# Tunables (Lesson 27: bounded concurrency). One semaphore shared by every
# fan-out branch so total in-flight VLM calls never exceed MAX_CONCURRENCY.
# ---------------------------------------------------------------------------
MAX_CONCURRENCY = 3
MAX_RETRIES = 1
BASE_DPI = 200
RETRY_DPI = 350  # re-render harder pages at a higher resolution
PRIMARY_MODEL = "gemini-2.0-flash"
RETRY_MODEL = "gemini-2.0-flash"  # swap to a stronger model here if desired

# Created lazily inside the async run so it binds to the right event loop.
_SEMAPHORE: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)
    return _SEMAPHORE


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
@dataclass
class PageResult:
    page: int
    markdown: str
    confidence: float
    attempts: int
    dpi: int


class GraphState(TypedDict, total=False):
    pdf_path: str
    num_pages: int
    mock: bool
    # Fan-in target: each worker branch appends exactly one PageResult.
    # The operator.add reducer concatenates the per-branch lists (Lesson 30).
    results: Annotated[list[PageResult], operator.add]
    document_markdown: str


class PageState(TypedDict, total=False):
    """The private payload Send delivers to one extract_page branch."""

    pdf_path: str
    page: int
    dpi: int
    model: str
    attempt: int
    mock: bool


# ---------------------------------------------------------------------------
# Confidence heuristic (stands in for a learned quality model)
# ---------------------------------------------------------------------------
_UNREADABLE = "[UNREADABLE]"


def score_confidence(markdown: str) -> float:
    """Cheap, deterministic proxy for "did this page extract cleanly?".

    A production system would use logprobs or a verifier model (that is Method
    7). Here we punish empty output and unreadable markers — exactly the signal
    that a scanned page (G2) or a dense table (G4) is in trouble.
    """
    text = (markdown or "").strip()
    if not text:
        return 0.0
    score = 1.0
    if _UNREADABLE in text:
        score -= 0.5 * text.count(_UNREADABLE)
    if len(text) < 40:  # suspiciously short for a real page
        score -= 0.4
    return max(0.0, min(1.0, score))


CONFIDENCE_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# The single VLM call, guarded by the shared semaphore.
# ---------------------------------------------------------------------------
async def _extract_one(state: PageState) -> PageResult:
    page = int(state["page"])
    dpi = int(state.get("dpi", BASE_DPI))
    attempt = int(state.get("attempt", 1))
    model = state.get("model", PRIMARY_MODEL)

    if state.get("mock"):
        # Deterministic stub: make page 1 look like a hard/scanned page on the
        # first attempt so the retry branch is exercised end-to-end.
        if page == 1 and attempt == 1:
            md = _UNREADABLE
        else:
            md = f"# Page {page}\n\n(mock extraction @ {dpi} dpi, attempt {attempt})"
        return PageResult(page, md, score_confidence(md), attempt, dpi)

    from langchain_core.messages import HumanMessage

    png = await asyncio.to_thread(render_page_png, state["pdf_path"], page, dpi)
    b64 = base64.b64encode(png).decode()
    vlm = get_llm("google", model=model, temperature=0)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": GEMINI_PROMPT},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        ]
    )
    async with _semaphore():  # Lesson 27: never exceed MAX_CONCURRENCY in flight
        out = await vlm.ainvoke([msg])
    md = out.content if hasattr(out, "content") else str(out)
    return PageResult(page, md, score_confidence(md), attempt, dpi)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
def ingest(state: GraphState) -> GraphState:
    """Resolve the page count (the only thing the fan-out router needs)."""
    if state.get("mock"):
        n = int(state.get("num_pages", 3))
    else:
        try:
            import pymupdf
        except ModuleNotFoundError as exc:  # pragma: no cover - install guard
            raise RuntimeError(
                "PyMuPDF is needed to count pages. Run: uv sync --extra extraction"
            ) from exc
        with pymupdf.open(state["pdf_path"]) as doc:
            n = doc.page_count
    return {"num_pages": n}


def fan_out(state: GraphState):
    """Router → one Send per page (Lesson 30 map step).

    Returns a list of ``Send`` objects; LangGraph launches one ``extract_page``
    branch per item in the same superstep.
    """
    from langgraph.types import Send

    return [
        Send(
            "extract_page",
            PageState(
                pdf_path=state.get("pdf_path", ""),
                page=p,
                dpi=BASE_DPI,
                model=PRIMARY_MODEL,
                attempt=1,
                mock=state.get("mock", False),
            ),
        )
        for p in range(int(state["num_pages"]))
    ]


async def extract_page(state: PageState) -> GraphState:
    """Worker branch (one per page) — extract, self-check, and self-correct.

    The retry loop lives *inside* the node on purpose. A node reached via
    ``Send`` receives its own private payload, but any conditional-edge router
    placed *after* the node sees the **merged** graph state (the accumulated
    ``results`` list, not this branch's page id). That makes a graph-level
    per-page retry edge ambiguous once branches interleave. Keeping the
    self-correction loop local to the branch keeps page identity unambiguous,
    while the *fan-out* (Send) and *fan-in* (deferred assemble) stay at the
    graph level where they belong.

    Self-check: if confidence is below threshold, re-render the page at a
    higher DPI (and optionally a stronger model) and re-extract, up to
    ``MAX_RETRIES``. Every VLM call passes through the shared semaphore.
    """
    payload = dict(state)
    result = await _extract_one(payload)  # attempt 1

    attempt = 1
    while result.confidence < CONFIDENCE_THRESHOLD and attempt <= MAX_RETRIES:
        attempt += 1
        payload = {
            **payload,
            "dpi": RETRY_DPI,      # re-render harder
            "model": RETRY_MODEL,  # alternate/stronger model
            "attempt": attempt,
        }
        result = await _extract_one(payload)

    # One PageResult appended to the shared list (operator.add reducer merges
    # every branch's single-element list — Lesson 30 reduce step).
    return {"results": [result]}


def assemble(state: GraphState) -> GraphState:
    """Deferred fan-in: concatenate pages in order into one Markdown document.

    Registered with ``defer=True`` so it runs only after every fan-out branch
    (and any retry branches they spawned) has completed — a synchronization
    barrier, not a race.
    """
    pages = sorted(state.get("results", []), key=lambda r: r.page)
    doc = "\n\n---\n\n".join(
        f"<!-- page {r.page} | conf={r.confidence:.2f} | "
        f"attempt {r.attempts} @ {r.dpi}dpi -->\n{r.markdown}"
        for r in pages
    )
    return {"document_markdown": doc}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------
def build_graph():
    """Construct and compile the self-correcting extraction graph.

    Topology::

        START → ingest → (fan_out: Send per page) ⇒ extract_page*
              → assemble(defer) → END

    ``extract_page`` runs its own bounded self-check/retry loop internally; the
    assemble node is deferred so it runs only after every page branch (and its
    in-branch retries) has completed — a synchronization barrier, not a race.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "langgraph is not installed. Run: uv add langgraph"
        ) from exc

    builder = StateGraph(GraphState)

    builder.add_node("ingest", ingest)
    builder.add_node("extract_page", extract_page)
    builder.add_node("assemble", assemble, defer=True)

    builder.add_edge(START, "ingest")
    # Conditional edge whose router returns a list[Send] → dynamic fan-out
    # (Lesson 30 map step). One extract_page branch per page.
    builder.add_conditional_edges("ingest", fan_out, ["extract_page"])
    # Every branch flows into the deferred assembler (Lesson 30 reduce step).
    builder.add_edge("extract_page", "assemble")
    builder.add_edge("assemble", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Primary entry points
# ---------------------------------------------------------------------------
async def extract_document(pdf_path: str | Path, *, mock: bool = False) -> dict[str, Any]:
    """Run the agentic graph over a whole PDF and return the assembled result."""
    graph = build_graph()
    init: GraphState = {"pdf_path": str(pdf_path), "mock": mock}
    final = await graph.ainvoke(init)
    return {
        "markdown": final.get("document_markdown", ""),
        "pages": sorted(final.get("results", []), key=lambda r: r.page),
    }


def demo(pdf_path: str | Path) -> None:
    """Entry point matching the harness convention: run and print a short result."""
    has_key = bool(os.getenv("GOOGLE_API_KEY"))
    mock = not has_key
    if mock:
        print(
            "[m6] GOOGLE_API_KEY not set — running the graph in MOCK mode so you "
            "can still see fan-out → retry → deferred-assemble.\n"
        )

    result = asyncio.run(extract_document(pdf_path, mock=mock))
    pages = result["pages"]
    retried = [p for p in pages if p.attempts > 1]
    print(f"[m6] agentic extraction over {len(pages)} page(s)")
    for p in pages:
        flag = "  <-- retried" if p.attempts > 1 else ""
        print(
            f"   page {p.page}: conf={p.confidence:.2f} "
            f"attempt={p.attempts} dpi={p.dpi}{flag}"
        )
    print(f"[m6] {len(retried)} page(s) triggered a self-correcting retry.")
    preview = result["markdown"][:400].replace("\n", " ")
    print(f"[m6] document preview: {preview!r}")


if __name__ == "__main__":
    # Pick argv[1], else the first present golden-set PDF; degrade gracefully.
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        try:
            from projects.vlm_extraction_harness.golden_set import GOLDEN_SET

            target = next(
                (g.pdf_path for g in GOLDEN_SET if g.exists), None
            )  # type: ignore[assignment]
        except Exception:
            target = None

    if target is None:
        print(
            "[m6] No PDF found. Running a pure-mock demo (no file needed) so the "
            "graph topology still executes.\n"
        )
        out = asyncio.run(extract_document("<none>", mock=True))
        for p in out["pages"]:
            print(f"   page {p.page}: conf={p.confidence:.2f} attempt={p.attempts}")
        print(f"[m6] assembled {len(out['pages'])} mock page(s).")
        sys.exit(0)

    if not Path(target).is_file():
        print(f"[m6] Not a file: {target}")
        sys.exit(1)

    demo(target)
