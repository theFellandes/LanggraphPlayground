"""Method 2 — Classical OCR + layout model (the pre-VLM SOTA).

Setup (uv ONLY — never pip)::

    uv sync --extra extraction          # PyMuPDF (render) + Pillow
    uv add pytesseract paddleocr        # the two OCR backends shown here
    # pytesseract also needs the native engine on PATH:
    #   Windows: install "Tesseract-OCR" (UB-Mannheim build) + lang packs
    #   macOS:   brew install tesseract tesseract-lang
    #   Linux:   apt-get install tesseract-ocr tesseract-ocr-tur tesseract-ocr-ara

Pipeline (no LLM, fully offline / CPU-friendly)::

    rasterize page -> detect layout regions + reading order -> OCR each region
    -> reassemble Markdown in reading order.

Two backends, same shape:

  * ``ppstructure_extract`` — PaddleOCR's **PP-StructureV3** pipeline. A real
    layout model (text/title/table/figure/formula detection + reading-order
    sort) feeding PP-OCR recognition; emits Markdown natively. This is the
    "layout model" half of the method done properly.

  * ``tesseract_layout_extract`` — a transparent, dependency-light teaching
    fallback: Tesseract's own page-segmentation (PSM 1, auto + OSD) gives
    block/paragraph/line structure via ``image_to_data`` TSV, which we
    reassemble into Markdown by block -> paragraph -> line. Shows the moving
    parts of the classical pipeline when a heavy layout model is unavailable.

Heavy deps are imported LAZILY inside functions so importing this module never
forces the ``extraction`` extra. This method calls **no** VLM — it is the CPU /
offline / Latin-scan baseline that frontier VLMs are measured against.
"""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

# Make the demo runnable as a plain script: add repo root to sys.path so
# ``shared`` / ``projects`` imports resolve.
# (demos/ -> vlm_extraction_harness/ -> projects/ -> repo root = 3 parents up.)
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def render_page_png(pdf_path: str | Path, page_number: int = 0, dpi: int = 300) -> bytes:
    """Render one PDF page to PNG bytes at ``dpi`` (mirrors the harness helper).

    Classical OCR is resolution-sensitive: 300 DPI is the floor for small type,
    and going below ~200 DPI collapses recognition accuracy.
    """
    try:
        import pymupdf  # PyMuPDF, installed by the ``extraction`` extra
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "PyMuPDF is needed to render pages. Run: uv sync --extra extraction"
        ) from exc
    with pymupdf.open(str(pdf_path)) as doc:
        page = doc[page_number]
        return page.get_pixmap(dpi=dpi).tobytes("png")


# Tesseract language codes per golden-set stress. Latin default is "eng";
# Turkish/Arabic need their packs installed natively or recognition is garbage.
_TESS_LANG_BY_FILE = {
    "g6_turkish.pdf": "tur",
    "g6_arabic.pdf": "ara",
}


def tesseract_layout_extract(
    pdf_path: str | Path,
    page_number: int = 0,
    dpi: int = 300,
    lang: str | None = None,
) -> str:
    """Backend A — rasterize, then use Tesseract's page segmentation as the
    layout model and reassemble Markdown from the TSV word boxes.

    ``image_to_data`` returns one row per detected token with the standard
    Tesseract TSV columns: ``level, page_num, block_num, par_num, line_num,
    word_num, left, top, width, height, conf, text``. We group word -> line ->
    paragraph -> block to recover reading order without any external layout net.
    PSM 1 = automatic page segmentation **with** orientation/script detection.
    """
    try:
        import pytesseract
        from pytesseract import Output
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "pytesseract not installed. Run: uv add pytesseract "
            "(and install the native Tesseract engine + language packs)."
        ) from exc
    try:
        import io

        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "Pillow not installed. Run: uv sync --extra extraction"
        ) from exc

    if lang is None:
        lang = _TESS_LANG_BY_FILE.get(Path(pdf_path).name, "eng")

    png = render_page_png(pdf_path, page_number, dpi)
    image = Image.open(io.BytesIO(png))

    try:
        # PSM 1: auto page segmentation with OSD (orientation + script).
        data = pytesseract.image_to_data(
            image, lang=lang, config="--psm 1", output_type=Output.DICT
        )
    except pytesseract.TesseractNotFoundError as exc:  # native engine missing
        raise RuntimeError(
            "The native Tesseract engine is not on PATH. Install it "
            "(e.g. UB-Mannheim build on Windows; `brew install tesseract` on macOS) "
            "plus the language pack you need (tur/ara)."
        ) from exc

    # Reassemble: walk tokens, break lines on (block, par, line) changes and
    # paragraph breaks on (block, par) changes -> Markdown paragraphs.
    n = len(data["text"])
    blocks: dict[tuple[int, int], list[list[str]]] = {}
    current_line_key: tuple[int, int, int] | None = None
    for i in range(n):
        word = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if not word or conf < 0:  # -1 rows are layout boxes, not words
            continue
        par_key = (data["block_num"][i], data["par_num"][i])
        line_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        para = blocks.setdefault(par_key, [])
        if line_key != current_line_key:
            para.append([])  # start a new line
            current_line_key = line_key
        para[-1].append(word)

    paragraphs = []
    for _key, lines in blocks.items():
        text = " ".join(" ".join(line) for line in lines if line).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def ppstructure_extract(
    pdf_path: str | Path,
    page_number: int = 0,
    dpi: int = 300,
    lang: str = "en",
) -> str:
    """Backend B — PP-StructureV3: a real layout model + PP-OCR -> Markdown.

    PP-StructureV3 runs document layout detection (text/title/table/figure/
    formula), reading-order sorting, table-structure recognition and OCR, then
    emits Markdown directly. We render the page to PNG and feed it the image so
    behaviour matches the rasterize-first classical pipeline (and so a single
    page can be targeted), then concatenate the per-image Markdown.
    """
    try:
        from paddleocr import PPStructureV3
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "paddleocr not installed. Run: uv add paddleocr "
            "(plus a paddlepaddle build for your platform)."
        ) from exc
    try:
        import io

        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "Pillow/numpy not installed. Run: uv sync --extra extraction"
        ) from exc

    png = render_page_png(pdf_path, page_number, dpi)
    # PP-StructureV3 accepts a path or an ndarray; feed the rendered page as RGB.
    image = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    pipeline = PPStructureV3(lang=lang)
    output = pipeline.predict(input=image)

    # ``res.markdown`` is a dict per page; concatenate_markdown_pages stitches
    # them (handling cross-page continuation) into one Markdown string.
    md_list = [res.markdown for res in output]
    try:
        return pipeline.concatenate_markdown_pages(md_list)
    except (AttributeError, TypeError):  # older builds: markdown dict carries the text directly
        parts = []
        for md in md_list:
            if isinstance(md, dict):
                parts.append(md.get("markdown_texts", "") or md.get("markdown", ""))
            else:
                parts.append(str(md))
        return "\n\n".join(p for p in parts if p)


def demo(pdf_path: str | Path) -> str:
    """Run the classical pipeline on one page and print a short result.

    Prefers PP-StructureV3 (the proper layout model); transparently falls back
    to the Tesseract page-segmentation path when PaddleOCR is unavailable, so
    the demo still shows the method end to end.
    """
    pdf_path = Path(pdf_path)
    print(f"[m2] classical OCR + layout on: {pdf_path.name}")

    text = None
    backend = None
    try:
        text = ppstructure_extract(pdf_path)
        backend = "PP-StructureV3 (layout model + PP-OCR)"
    except RuntimeError as exc:
        print(f"[m2] PP-StructureV3 unavailable -> {exc}")
        print("[m2] falling back to Tesseract page-segmentation backend.")
        try:
            text = tesseract_layout_extract(pdf_path)
            backend = "Tesseract PSM-1 (page segmentation)"
        except RuntimeError as exc2:
            print(f"[m2] Tesseract backend also unavailable -> {exc2}")
            print("[m2] Install a backend, then re-run. See the module docstring.")
            return ""

    text = text or ""
    print(f"[m2] backend: {backend}")
    print(f"[m2] extracted {len(text)} chars. First 600:\n")
    print(text[:600])
    return text


if __name__ == "__main__":
    # Run on argv[1] if given, else the first present golden-set PDF.
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        try:
            from projects.vlm_extraction_harness.golden_set import GOLDEN_SET

            target = next((g.pdf_path for g in GOLDEN_SET if g.exists), None)
        except Exception as exc:  # pragma: no cover - import/setup guard
            print(f"[m2] could not load golden set: {exc}")
            target = None

    if target is None:
        print(
            "[m2] No PDF to run on. Drop a golden-set PDF into "
            "data/sample_docs/ (e.g. g1_born_digital_prose.pdf) or pass a path: "
            "python m2_classical_ocr_layout.py <file.pdf>"
        )
        sys.exit(0)

    if not Path(target).is_file():
        print(f"[m2] file not found: {target}")
        sys.exit(0)

    demo(target)
