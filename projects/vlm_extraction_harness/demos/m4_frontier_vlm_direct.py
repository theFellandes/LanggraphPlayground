"""Method 4 — Frontier-VLM direct (page-as-image → Markdown).

Setup (uv ONLY — never pip)::

    uv sync --extra extraction          # PyMuPDF, for rendering the page
    uv add langchain-google-genai        # the Gemini adapter behind get_llm("google")

The single most flexible extractor in the taxonomy (FINDINGS §4, row 4): we
rasterise one PDF page to a PNG, base64-encode it, and hand the *pixels* — not
the glyph stream — to a frontier multimodal model with a locked Markdown prompt.
Nothing about the page's internal structure is parsed by us; the model reads the
image the way a human would. That is its power (handles scanned, RTL, dense
tables, figure-heavy, arbitrary scripts with zero local infra) and its danger
(cost/throughput at volume, privacy, and a measured **12.4% hallucination rate
on dense text** — invented cells that are undetectable downstream).

The VLM is reached **only** through ``get_llm("google", ...)`` so it stays
switchable and inherits the project's ``Runnable.with_fallbacks`` chain — never a
raw ``ChatGoogleGenerativeAI`` / google-genai / ``openai`` SDK call.

This module is a focused teaching demo, not a framework. It deliberately mirrors
``extractors.gemini_direct`` and reuses ``render_page_png`` from that module.
"""

from __future__ import annotations

import base64
import pathlib
import sys

# Make the demo runnable as a plain script: add the repo root to sys.path so the
# absolute ``shared.*`` / ``projects.*`` imports resolve.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.llm import get_llm  # noqa: E402  (after sys.path shim)

# Reuse the harness renderer so we render identically to the rest of the suite
# (PyMuPDF, page.get_pixmap(dpi=dpi).tobytes("png")).
from projects.vlm_extraction_harness.extractors import render_page_png  # noqa: E402


# Locked extraction prompt. Encodes the FINDINGS §5 prompting playbook:
# explicit column-first reading order, HTML tables for merged cells, LaTeX for
# formulas, figure captions only (never invented pixels), and an explicit
# anti-hallucination clause with an [UNREADABLE] escape hatch. Temperature is 0.
FRONTIER_PROMPT = (
    "You are a precise document-extraction engine. Transcribe THIS PAGE IMAGE "
    "into GitHub-flavored Markdown, in natural human reading order.\n"
    "- Multi-column layout: read the leftmost column top-to-bottom BEFORE moving "
    "to the next column. Never interleave columns left-to-right.\n"
    "- Right-to-left scripts (Arabic/Hebrew): preserve correct logical reading "
    "order; do not reverse digits or Latin runs embedded in the text.\n"
    "- Tables: emit valid HTML (<table>/<tr>/<th>/<td> with colspan/rowspan for "
    "merged cells). Never use Markdown pipe tables when cells are merged.\n"
    "- Formulas: render as LaTeX inside $...$ (inline) or $$...$$ (display).\n"
    "- Figures/charts: emit ![<one-sentence factual caption>](figure). Describe "
    "only what is visibly present; never invent data points or values.\n"
    "- Do NOT hallucinate. If any region is illegible, write [UNREADABLE] there "
    "rather than guessing.\n"
    "Output the Markdown only — no preamble, no code fences around the whole answer."
)

# Gemini scales any image down/pads to a 3072x3072 box preserving aspect ratio
# (verified, see api_sources). 300 DPI on Letter/A4 stays under that, so we do
# NOT pre-downscale — letting the provider rasterise is the anti-pattern
# (FINDINGS §5: providers internally drop to ~90 DPI). We render crisp ourselves.
DEFAULT_DPI = 300
DEFAULT_MODEL = "gemini-2.0-flash"


def frontier_vlm_direct(
    pdf_path: str | pathlib.Path,
    page_number: int = 0,
    dpi: int = DEFAULT_DPI,
    model: str = DEFAULT_MODEL,
) -> str:
    """Render one page to an image and ask a frontier VLM for Markdown.

    Pipeline: render PNG (PyMuPDF) → base64 data URI → ``get_llm("google")``
    multimodal ``HumanMessage`` → ``.content`` Markdown string.

    Requires ``GOOGLE_API_KEY`` (the google adapter guards this) and the
    ``langchain-google-genai`` package. Both raise a clear RuntimeError if
    missing, so callers degrade gracefully.
    """
    # langchain_core is always installed (core dep); import locally to keep the
    # module's import surface tiny and consistent with the lazy-import house style.
    from langchain_core.messages import HumanMessage

    png = render_page_png(pdf_path, page_number=page_number, dpi=dpi)
    b64 = base64.b64encode(png).decode()

    # temperature=0 for deterministic transcription (raise only on retry to break
    # repetition loops — out of scope for this single-shot teaching demo).
    vlm = get_llm("google", model=model, temperature=0)

    # The multimodal content shape langchain-google-genai expects: a text part
    # plus an "image_url" part carrying a base64 data URI (verified, see
    # api_sources). This is the exact shape used across the harness.
    message = HumanMessage(
        content=[
            {"type": "text", "text": FRONTIER_PROMPT},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        ]
    )
    out = vlm.invoke([message])
    return out.content if hasattr(out, "content") else str(out)


def demo(pdf_path: str | pathlib.Path) -> None:
    """Run the frontier-VLM-direct extractor on page 0 and print a short result.

    Degrades gracefully: a missing dependency or ``GOOGLE_API_KEY`` becomes a
    clear skip message instead of a traceback.
    """
    print(f"[m4] frontier-VLM direct on: {pdf_path}")
    print(f"[m4] render {DEFAULT_DPI} DPI PNG -> base64 -> get_llm('google', "
          f"model='{DEFAULT_MODEL}') multimodal")
    try:
        markdown = frontier_vlm_direct(pdf_path)
    except RuntimeError as exc:
        # Raised by render_page_png (no PyMuPDF), the google adapter (no
        # GOOGLE_API_KEY), or the lazy langchain-google-genai guard.
        print(f"[m4] SKIPPED — {exc}")
        return
    except Exception as exc:  # pragma: no cover - network/quota/etc.
        print(f"[m4] SKIPPED — VLM call failed: {type(exc).__name__}: {exc}")
        return

    preview = markdown.strip()
    print(f"[m4] OK — {len(markdown)} chars of Markdown. First 600 chars:\n")
    print(preview[:600])
    if len(preview) > 600:
        print("\n[... truncated ...]")


def _pick_pdf() -> pathlib.Path | None:
    """First CLI arg, else the first present golden-set PDF."""
    if len(sys.argv) > 1:
        return pathlib.Path(sys.argv[1])
    try:
        from projects.vlm_extraction_harness.golden_set import GOLDEN_SET
    except Exception:  # pragma: no cover - if golden_set import fails, bail clean
        return None
    for page in GOLDEN_SET:
        if page.exists:
            return page.pdf_path
    return None


if __name__ == "__main__":
    target = _pick_pdf()
    if target is None:
        print(
            "[m4] No PDF to run on. Pass one explicitly:\n"
            "    python projects/vlm_extraction_harness/demos/m4_frontier_vlm_direct.py "
            "path/to/page.pdf\n"
            "or drop a golden-set PDF (e.g. g1_born_digital_prose.pdf) into "
            "data/sample_docs/."
        )
        sys.exit(0)
    if not pathlib.Path(target).is_file():
        print(f"[m4] File not found: {target}")
        sys.exit(0)
    demo(target)
