"""Method 1 - Text-layer extraction (FINDINGS.md §4, method #1).

Setup (uv only - never pip):

    uv sync --extra extraction
    # or, if adding the deps explicitly:
    uv add pymupdf4llm pdfplumber

Gist
----
Read the PDF's *embedded glyph stream* directly. The page already carries its
text as positioned characters (font + Unicode codepoint + x/y), so we never
rasterize the page and never run OCR. This is the fastest, cheapest, and most
faithful method that exists - but ONLY for born-digital PDFs that actually have
a text layer. On a scanned (image-only) page there is no glyph stream, so the
output is silently empty: that is this method's hard floor, not a bug.

This module shows two complementary readers of the same glyph stream:

  * ``pymupdf4llm_text_layer`` - the primary path. pymupdf4llm walks PyMuPDF's
    text dict, infers headings from font sizes, detects tables via the vector
    ``lines_strict`` strategy, and serializes straight to GitHub-flavored
    Markdown. Best default for prose + light tables.

  * ``pdfplumber_text_layer`` - a transparent, lower-level cross-check. It
    exposes ``page.chars`` / ``page.images`` / ``page.extract_tables`` so you
    can see *exactly* which glyphs and image XObjects live on the page. Useful
    for auditing reading order and for grid-ruled tables.

Both are pure text-layer reads. Neither touches a VLM - by design. (The harness
``get_llm`` adapter is intentionally NOT imported here: method #1's whole point
is that no model is needed. The frontier-VLM path is method #4.)

Heavy deps are imported LAZILY inside functions and guarded, so importing this
module never forces the ``extraction`` extra.
"""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

# Make the repo importable when run as a bare script (python m1_text_layer.py).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# Primary reader: pymupdf4llm                                                  #
# --------------------------------------------------------------------------- #
def pymupdf4llm_text_layer(
    pdf_path: str | Path,
    page_number: int | None = 0,
    *,
    table_strategy: str = "lines_strict",
) -> str:
    """Read the embedded text layer and serialize to Markdown via pymupdf4llm.

    No rendering, no OCR. ``force_ocr`` is left False and ``ignore_images`` True
    so figures are skipped (method #1 does not interpret pixels - see
    ``image_handling`` in the architecture notes). On a scanned page this
    returns near-empty Markdown, which the demo flags as the text-layer floor.

    Args:
        pdf_path: path to the PDF.
        page_number: 0-based page to read, or ``None`` for the whole document.
        table_strategy: PyMuPDF table detection; ``lines_strict`` uses real
            vector rules (no guessing from text gaps).
    """
    try:
        import pymupdf4llm
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "pymupdf4llm not installed. Run: uv sync --extra extraction"
        ) from exc

    pages = None if page_number is None else [page_number]
    return pymupdf4llm.to_markdown(
        str(pdf_path),
        pages=pages,
        table_strategy=table_strategy,
        ignore_images=True,   # method #1 does not extract figure pixels
        force_ocr=False,      # never OCR - that would make this method #2
        show_progress=False,
    )


# --------------------------------------------------------------------------- #
# Cross-check reader: pdfplumber                                              #
# --------------------------------------------------------------------------- #
def pdfplumber_text_layer(
    pdf_path: str | Path,
    page_number: int = 0,
    *,
    layout: bool = False,
) -> dict:
    """Low-level glyph-stream read with pdfplumber, for auditing.

    Returns a dict so you can *see* the raw evidence of the text layer:
        {
          "text":         str,    # extract_text() output
          "char_count":   int,    # number of positioned glyphs found
          "tables":       list,   # extract_tables() - ruled-grid tables
          "image_count":  int,    # image XObjects present (NOT decoded here)
          "image_bboxes": list,   # bbox of each embedded figure (see notes)
        }

    A ``char_count`` of 0 is the unambiguous signal that this page has no text
    layer (i.e. it is scanned) and method #1 cannot serve it.
    """
    try:
        import pdfplumber
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "pdfplumber not installed. Run: uv sync --extra extraction"
        ) from exc

    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[page_number]
        text = page.extract_text(layout=layout) or ""
        tables = page.extract_tables()
        # page.images are image XObject placements; we record their bboxes only.
        image_boxes = [
            {
                "name": img.get("name"),
                "bbox": (img.get("x0"), img.get("top"), img.get("x1"), img.get("bottom")),
                "srcsize": img.get("srcsize"),
            }
            for img in page.images
        ]
        return {
            "text": text,
            "char_count": len(page.chars),
            "tables": tables,
            "image_count": len(page.images),
            "image_boxes": image_boxes,
        }


def demo(pdf_path: str | Path) -> None:
    """Run both text-layer readers on one PDF and print a short result."""
    pdf_path = Path(pdf_path)
    print(f"== Method 1: text-layer extraction on {pdf_path.name} ==\n")

    # --- pdfplumber cross-check first: it tells us if a text layer even exists.
    try:
        audit = pdfplumber_text_layer(pdf_path, page_number=0)
    except RuntimeError as exc:
        print(f"[skip] pdfplumber path unavailable: {exc}\n")
        audit = None

    if audit is not None:
        print(f"[pdfplumber] glyphs on page 1 : {audit['char_count']}")
        print(f"[pdfplumber] ruled tables     : {len(audit['tables'])}")
        print(f"[pdfplumber] embedded figures : {audit['image_count']}")
        if audit["char_count"] == 0:
            print(
                "\n[!] No glyphs found -> this page has NO text layer (scanned).\n"
                "    Method #1 cannot serve this page. Use OCR (method #2) or a\n"
                "    VLM (method #4). This empty result IS the text-layer floor.\n"
            )
        snippet = audit["text"].strip().replace("\n", " ")[:200]
        print(f"[pdfplumber] text preview     : {snippet!r}\n")

    # --- primary Markdown serialization via pymupdf4llm.
    try:
        md = pymupdf4llm_text_layer(pdf_path, page_number=0)
    except RuntimeError as exc:
        print(f"[skip] pymupdf4llm path unavailable: {exc}")
        return

    md = md.strip()
    print("[pymupdf4llm] Markdown preview (first 600 chars):")
    print("-" * 60)
    print(md[:600] if md else "(empty - no text layer on this page)")
    print("-" * 60)


def _first_present_pdf() -> Path | None:
    """Pick argv[1] if given, else the first golden-set PDF that exists on disk."""
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        return candidate if candidate.is_file() else None
    try:
        from projects.vlm_extraction_harness.golden_set import GOLDEN_SET
    except Exception:  # pragma: no cover - fall back to a flat scan
        return None
    for page in GOLDEN_SET:
        if page.exists:
            return page.pdf_path
    return None


if __name__ == "__main__":
    pdf = _first_present_pdf()
    if pdf is None:
        print(
            "No PDF to run on. Pass one explicitly:\n"
            "    python projects/vlm_extraction_harness/demos/m1_text_layer.py path/to/file.pdf\n"
            "or drop a golden-set PDF (e.g. g1_born_digital_prose.pdf) into\n"
            "data/sample_docs/. Born-digital PDFs work best for method #1."
        )
        sys.exit(0)
    demo(pdf)
