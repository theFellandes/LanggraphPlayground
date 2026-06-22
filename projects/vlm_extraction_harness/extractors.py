"""The three methods under test (TEST-PLAN.md §3).

  A. ``gemini_direct``      — page image → Markdown via ``get_llm("google", ...)``
                              (adapter only; no raw SDK call — TEST-PLAN §1).
  B. ``docling_extract``    — Docling pipeline → native Markdown (the recommended
                              primary extractor, FINDINGS §8).
  C. ``pymupdf4llm_extract``— text-layer baseline (proves when a VLM is overkill,
                              and its hard floor on scanned pages).

Heavy deps (``docling``, ``pymupdf``/``pymupdf4llm``) are imported **lazily** so
importing this module never forces the ``extraction`` extra. Install with::

    uv sync --extra extraction
"""

from __future__ import annotations

import base64
from pathlib import Path

from shared.llm import get_llm

# Locked Gemini-direct prompt (TEST-PLAN §3 — structured-output discipline:
# column-first reading order, HTML tables, LaTeX formulas, anti-hallucination).
GEMINI_PROMPT = (
    "Return the content of this page as GitHub-flavored Markdown in natural "
    "reading order.\n"
    "- If the page has multiple columns, read the leftmost column fully before "
    "the next.\n"
    "- Render every table as valid HTML (<table>/<th>/<td> with colspan/rowspan). "
    "Never use Markdown pipes for merged cells.\n"
    "- Render formulas as LaTeX ($...$).\n"
    "- For each figure, emit ![<one-sentence caption>](figure) and nothing invented.\n"
    "- Do not hallucinate. If a region is unreadable, write [UNREADABLE]."
)


def render_page_png(pdf_path: str | Path, page_number: int = 0, dpi: int = 300) -> bytes:
    """Render one PDF page to PNG bytes at ``dpi`` (TEST-PLAN: 300 for dense content)."""
    try:
        import pymupdf  # PyMuPDF (installed transitively by pymupdf4llm)
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "PyMuPDF is needed to render pages. Run: uv sync --extra extraction"
        ) from exc
    with pymupdf.open(str(pdf_path)) as doc:
        page = doc[page_number]
        return page.get_pixmap(dpi=dpi).tobytes("png")


def gemini_direct(
    pdf_path: str | Path,
    page_number: int = 0,
    dpi: int = 300,
    model: str = "gemini-2.0-flash",
) -> str:
    """Method A — render the page and send the image to Gemini through the adapter.

    The VLM is reached **only** via ``get_llm("google", ...)`` so it is switchable
    and inherits the ``Runnable.with_fallbacks`` chain (TEST-PLAN §1). Requires
    ``GOOGLE_API_KEY`` and ``uv add langchain-google-genai``.
    """
    from langchain_core.messages import HumanMessage

    png = render_page_png(pdf_path, page_number, dpi)
    b64 = base64.b64encode(png).decode()
    vlm = get_llm("google", model=model, temperature=0)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": GEMINI_PROMPT},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        ]
    )
    out = vlm.invoke([msg])
    return out.content if hasattr(out, "content") else str(out)


def docling_extract(pdf_path: str | Path) -> str:
    """Method B — Docling pipeline → native Markdown."""
    try:
        from docling.document_converter import DocumentConverter
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "docling is not installed. Run: uv sync --extra extraction"
        ) from exc
    result = DocumentConverter().convert(str(pdf_path))
    return result.document.export_to_markdown()


def pymupdf4llm_extract(pdf_path: str | Path, page_number: int | None = 0) -> str:
    """Method C — text-layer baseline. Returns empty-ish text on scanned pages."""
    try:
        import pymupdf4llm
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "pymupdf4llm is not installed. Run: uv sync --extra extraction"
        ) from exc
    pages = None if page_number is None else [page_number]
    return pymupdf4llm.to_markdown(str(pdf_path), pages=pages)


# Registry consumed by run.py. Each entry: (label, callable taking a pdf_path).
METHODS = {
    "gemini_direct": gemini_direct,
    "docling": docling_extract,
    "pymupdf4llm": pymupdf4llm_extract,
}
