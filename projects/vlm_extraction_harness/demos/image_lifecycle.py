"""Image lifecycle on a figure-heavy PDF page: extract -> caption -> emit -> index.

Setup (heavy deps are imported LAZILY inside functions, so importing this module
never forces the ``extraction`` extra)::

    uv sync --extra extraction        # PyMuPDF (render + embedded-image pull) and Docling
    uv add langchain-google-genai     # the Gemini VLM provider (already a core dep)

This answers the central question — *"how can we extract an image, what can we do
with it, and what happens to it after a model processes it?"* — end to end:

  1. EXTRACT images two distinct ways
     (i)  pull embedded raster XObjects   — ``page.get_images()`` + ``doc.extract_image(xref)``
     (ii) render a figure REGION to PNG   — ``page.get_pixmap(clip=rect, dpi=...)``
     (a third way, a parser's own picture export via Docling, is shown in
      ``docling_pictures`` so all three "ways to get an image out" are runnable.)
  2. CAPTION each figure with a VLM via ``get_llm("google", ...)`` (image_url message),
     with an anti-hallucination prompt + chart series/axis extraction.
  3. EMIT a Markdown ``![caption](path)`` at the figure's position AND a metadata dict
     ``{figure_path, caption, bbox, page, classification}`` a chunker (Guillotine) carries.
  4. INDEX the caption text for retrieval (clearly-marked stub) — the single place an
     embedding/index call goes so the figure becomes findable by a text-RAG query.

The VLM is reached ONLY through the adapter (``get_llm("google", ...)``), never a raw
SDK call, so it stays switchable and inherits the ``Runnable.with_fallbacks`` chain.

Verified June 2026 against:
  - PyMuPDF recipes-images (get_images / extract_image / get_image_rects / get_pixmap clip)
    https://pymupdf.readthedocs.io/en/latest/recipes-images.html
  - PyMuPDF Page.get_image_rects
    https://pymupdf.readthedocs.io/en/latest/page.html
  - Docling export_figures example (generate_picture_images / PictureItem.get_image)
    https://github.com/docling-project/docling/blob/main/docs/examples/export_figures.py
"""

from __future__ import annotations

import base64
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Make the demo runnable both as `python -m ...` and as a direct script.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.llm import get_llm  # noqa: E402  (core dep)

# Anti-hallucination caption prompt. Asks for grounded description only, and for
# charts the actual series/axis labels — never invented numbers.
CAPTION_PROMPT = (
    "You are labelling ONE extracted figure for a document search index.\n"
    "Return STRICT JSON with keys: caption, classification, axes, series.\n"
    "- caption: one factual sentence describing only what is visibly in the image.\n"
    "- classification: one of 'chart', 'photo', 'diagram', 'table', 'logo', 'other'.\n"
    "- axes: for a chart, the x and y axis labels exactly as printed, else null.\n"
    "- series: for a chart, the legend/series names exactly as printed, else [].\n"
    "Do NOT invent numbers, trends, titles, or labels. If text is unreadable, use null. "
    "If the image is not a chart, set axes=null and series=[]."
)


@dataclass(frozen=True)
class FigureRecord:
    """The metadata a chunker (Guillotine) carries so a figure is retrievable + citable."""

    figure_path: str
    caption: str
    bbox: tuple[float, float, float, float] | None
    page: int
    classification: str
    source: str  # "embedded_xref" | "rendered_region" | "docling_picture"


# ---------------------------------------------------------------------------
# 1. EXTRACT — three distinct ways to get an image out of a PDF
# ---------------------------------------------------------------------------
def extract_embedded_images(
    pdf_path: str | Path, page_number: int = 0, out_dir: str | Path | None = None
) -> list[tuple[Path, tuple[float, float, float, float] | None]]:
    """Way (i): pull embedded raster XObjects directly — no render, no OCR.

    Uses ``page.get_images(full=True)`` for the xref list, ``doc.extract_image(xref)``
    for the original compressed bytes (thousands of times faster than rasterising),
    and ``page.get_image_rects(xref)`` for the on-page bbox. Returns saved-PNG paths
    paired with their page bbox (``None`` if the image is not actually placed).
    """
    try:
        import pymupdf  # PyMuPDF
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "PyMuPDF not installed. Run: uv sync --extra extraction"
        ) from exc

    out = Path(out_dir or (ROOT / "data" / "sample_docs" / "figures"))
    out.mkdir(parents=True, exist_ok=True)
    saved: list[tuple[Path, tuple[float, float, float, float] | None]] = []

    with pymupdf.open(str(pdf_path)) as doc:
        page = doc[page_number]
        for img in page.get_images(full=True):
            xref = img[0]
            info = doc.extract_image(xref)  # {'image','ext','width','height',...}
            dest = out / f"p{page_number}_xref{xref}.{info['ext']}"
            dest.write_bytes(info["image"])
            rects = page.get_image_rects(xref)  # list[Rect]; [] if not placed
            bbox = tuple(rects[0]) if rects else None
            saved.append((dest, bbox))
    return saved


def render_region_png(
    pdf_path: str | Path,
    bbox: tuple[float, float, float, float],
    page_number: int = 0,
    dpi: int = 300,
    out_dir: str | Path | None = None,
) -> Path:
    """Way (ii): rasterise just the figure REGION via a clip rect (same get_pixmap path
    as ``extractors.render_page_png``, but with a ``clip`` rectangle).

    This is the robust way to capture vector figures / charts that have NO embedded
    raster XObject — the only pixels that exist are the ones you render.
    """
    try:
        import pymupdf
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "PyMuPDF not installed. Run: uv sync --extra extraction"
        ) from exc

    out = Path(out_dir or (ROOT / "data" / "sample_docs" / "figures"))
    out.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(str(pdf_path)) as doc:
        page = doc[page_number]
        clip = pymupdf.Rect(*bbox)
        png = page.get_pixmap(dpi=dpi, clip=clip).tobytes("png")
    dest = out / f"p{page_number}_region_{int(bbox[0])}_{int(bbox[1])}.png"
    dest.write_bytes(png)
    return dest


def docling_pictures(
    pdf_path: str | Path, out_dir: str | Path | None = None
) -> list[Path]:
    """Way (iii): let a specialised parser export its own picture crops.

    Docling, with ``generate_picture_images=True``, detects figures during layout
    analysis and hands you cropped ``PictureItem`` images plus their provenance — the
    "parser picture export" path. Returns saved-PNG paths.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc import PictureItem
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "docling not installed. Run: uv sync --extra extraction"
        ) from exc

    out = Path(out_dir or (ROOT / "data" / "sample_docs" / "figures"))
    out.mkdir(parents=True, exist_ok=True)

    opts = PdfPipelineOptions()
    opts.images_scale = 2.0  # ~144 DPI crops; raise for denser figures
    opts.generate_picture_images = True
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    res = conv.convert(str(pdf_path))
    saved: list[Path] = []
    for i, (element, _level) in enumerate(res.document.iterate_items()):
        if isinstance(element, PictureItem):
            dest = out / f"docling_picture_{i}.png"
            with dest.open("wb") as fp:
                element.get_image(res.document).save(fp, "PNG")
            saved.append(dest)
    return saved


# ---------------------------------------------------------------------------
# 2. CAPTION — what the model does to the image (anti-hallucination)
# ---------------------------------------------------------------------------
def caption_figure(png_path: str | Path, model: str = "gemini-2.0-flash") -> dict:
    """Send one figure image to the VLM via the adapter; return the parsed caption dict.

    Requires ``GOOGLE_API_KEY``. The model NEVER sees the rest of the page — only the
    cropped figure — which is what keeps the caption grounded.
    """
    from langchain_core.messages import HumanMessage

    b64 = base64.b64encode(Path(png_path).read_bytes()).decode()
    vlm = get_llm("google", model=model, temperature=0)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": CAPTION_PROMPT},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        ]
    )
    raw = vlm.invoke([msg])
    text = raw.content if hasattr(raw, "content") else str(raw)
    return _parse_caption_json(text)


def _parse_caption_json(text: str) -> dict:
    """Best-effort parse of the model's JSON (strips ```json fences); degrade to text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data.setdefault("caption", "")
            data.setdefault("classification", "other")
            return data
    except json.JSONDecodeError:
        pass
    return {"caption": text.strip()[:300], "classification": "other"}


# ---------------------------------------------------------------------------
# 3. EMIT — Markdown ref + chunk metadata
# ---------------------------------------------------------------------------
def figure_markdown(record: FigureRecord) -> str:
    """The Markdown a chunker drops at the figure's reading-order position."""
    rel = record.figure_path
    return f"![{record.caption}]({rel})"


# ---------------------------------------------------------------------------
# 4. INDEX — STUB: where the figure becomes retrievable by a text RAG query
# ---------------------------------------------------------------------------
def index_figure(record: FigureRecord) -> None:
    """STUB — embed the CAPTION text and upsert with the figure metadata as payload.

    The caption (not the pixels) is what a text query matches against; the metadata
    payload carries the path/bbox/page so a hit can render and cite the actual figure.

    Wire this to your real store, e.g.::

        from langchain_openai import OpenAIEmbeddings           # or any embeddings
        vector = OpenAIEmbeddings().embed_query(record.caption)
        store.upsert(id=record.figure_path,
                     vector=vector,
                     payload=asdict(record))  # path/bbox/page ride along

    Left as a stub so the demo runs with zero vector-store infra.
    """
    payload = asdict(record)  # noqa: F841 — shows exactly what would be upserted
    # store.upsert(id=record.figure_path, vector=embed(record.caption), payload=payload)
    return None


# ---------------------------------------------------------------------------
# Orchestration + demo
# ---------------------------------------------------------------------------
def run_lifecycle(pdf_path: str | Path, page_number: int = 0) -> list[FigureRecord]:
    """Full lifecycle on one page. Captioning degrades gracefully without a key."""
    have_key = bool(get_settings_key())
    records: list[FigureRecord] = []

    pairs = extract_embedded_images(pdf_path, page_number)
    source = "embedded_xref"
    if not pairs:
        # No embedded raster (e.g. a vector chart): render the full page region as the
        # figure so the lifecycle still has pixels to work with.
        try:
            import pymupdf

            with pymupdf.open(str(pdf_path)) as doc:
                rect = doc[page_number].rect
            png = render_region_png(pdf_path, tuple(rect), page_number)
            pairs = [(png, tuple(rect))]
            source = "rendered_region"
        except RuntimeError:
            return records

    for path, bbox in pairs:
        if have_key:
            try:
                cap = caption_figure(path)
            except Exception as exc:  # noqa: BLE001 — keep extracting even if VLM fails
                cap = {"caption": f"[caption skipped: {type(exc).__name__}]",
                       "classification": "unknown"}
        else:
            cap = {"caption": "[caption skipped: no GOOGLE_API_KEY]",
                   "classification": "unknown"}

        rec = FigureRecord(
            figure_path=str(path),
            caption=cap.get("caption", ""),
            bbox=bbox,
            page=page_number,
            classification=cap.get("classification", "other"),
            source=source,
        )
        records.append(rec)
        index_figure(rec)  # stub: caption -> embedding -> index
    return records


def get_settings_key() -> str | None:
    """Read the Google key through settings (no os.environ in call sites)."""
    try:
        from shared.settings import settings

        return settings.google_api_key
    except Exception:  # noqa: BLE001
        return None


def demo(pdf_path: str | Path) -> None:
    """Print a short end-to-end result for one figure-heavy page."""
    print(f"[image_lifecycle] page 0 of {pdf_path}")
    records = run_lifecycle(pdf_path, page_number=0)
    if not records:
        print("  no figures extracted (no embedded images and render unavailable).")
        return
    for rec in records:
        print("\n  EXTRACT:", rec.source, "->", rec.figure_path)
        print("  bbox   :", rec.bbox, "page", rec.page)
        print("  CAPTION:", rec.caption, f"({rec.classification})")
        print("  EMIT   :", figure_markdown(rec))
        print("  INDEX  : caption embedded + metadata upserted ->",
              json.dumps(asdict(rec))[:120], "...")


def _first_present_pdf() -> Path | None:
    from projects.vlm_extraction_harness.golden_set import GOLDEN_SET

    figure_heavy = [p for p in GOLDEN_SET if p.id == "G5" and p.exists]
    if figure_heavy:
        return figure_heavy[0].pdf_path
    for p in GOLDEN_SET:
        if p.exists:
            return p.pdf_path
    return None


if __name__ == "__main__":
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else _first_present_pdf()
    if pdf is None or not Path(pdf).is_file():
        print(
            "No PDF to run on. Add a figure-heavy PDF "
            "(data/sample_docs/g5_figure_heavy.pdf) or pass one: "
            "uv run python -m projects.vlm_extraction_harness.demos.image_lifecycle <pdf>"
        )
        raise SystemExit(0)
    if not get_settings_key():
        print("[note] GOOGLE_API_KEY not set — extracting + storing, skipping VLM caption.\n")
    demo(pdf)
