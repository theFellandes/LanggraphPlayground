"""Method 5 — Hybrid pipeline (layout → region routing → VLM on hard regions only).

  uv sync --extra extraction          # PyMuPDF (render + text-layer + image rects)
  uv add langchain-google-genai       # the Gemini VLM provider (reached via get_llm)

The production sweet spot (FINDINGS.md §4, row 5). Instead of paying a frontier
VLM to read *every* page (Method 4) — slow, costly, and 12.4% hallucination on
dense text — we **route by region difficulty**:

  * EASY  regions  (born-digital prose with a real glyph stream) → text-layer.
            Free, instant, lossless. This is Method 1 applied per-region.
  * HARD  regions  (figures / charts / scanned images) → render just that crop
            and send the PNG to a VLM via ``get_llm("google", ...)`` for a caption.

Then we **merge by bbox into reading order** so the cheap prose and the expensive
figure captions reassemble into one coherent Markdown stream.

This file ships a *simplified, runnable* router that uses PyMuPDF as the layout
oracle: ``page.get_text("dict")`` yields text blocks (``type == 0``) with spans +
bboxes for the cheap path, and image blocks (``type == 1``) / ``get_image_info``
give figure rectangles for the VLM path. In a real system the segmenter is a
trained layout model (DocLayNet/PP-StructureV3/Docling layout) that *also* tags
``Table`` and ``Title`` regions and detects borderless tables — see
``WHERE_A_REAL_LAYOUT_MODEL_SLOTS_IN`` below. The routing/merge logic is identical;
only the source of the region list changes.

Heavy deps are imported LAZILY inside functions so importing this module never
forces the ``extraction`` extra.
"""

from __future__ import annotations

import base64
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Make the demo runnable as a plain script (`python .../m5_hybrid_region_routing.py`)
# as well as via `-m`. Repo root is three levels up from this file.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.llm import get_llm  # noqa: E402  (core dep, no heavy import)

# Where a *real* layout model replaces the PyMuPDF heuristic segmenter below.
WHERE_A_REAL_LAYOUT_MODEL_SLOTS_IN = """\
Swap `segment_regions()` for a trained detector. Each detector emits typed,
bbox'd regions; routing/merge stay the same:
  * Docling   : DocumentConverter(...).convert(pdf); walk doc.iterate_items()
                — each item has a `prov[0].bbox` + a label (Text/Title/Table/
                Picture/Caption). Route Table/Picture to the VLM, Text to text-layer.
  * PP-StructureV3 (PaddleOCR): PPStructureV3()(img) → list of {type, bbox, ...}
                with type in {text, title, table, figure, header, footer}.
  * DocLayNet/YOLO: a layout YOLO returns class+bbox per region.
A real model adds the two region classes PyMuPDF cannot see on its own:
borderless TABLES (route to VLM as HTML) and column structure (for reading order).
"""

# VLM prompt for a single cropped figure region — caption only, no invention.
FIGURE_PROMPT = (
    "This image is a single figure or chart cropped from a PDF page. "
    "In one sentence, describe what it shows. If it is a chart, name the chart "
    "type and the quantities on each axis. Do not invent numbers or labels you "
    "cannot read. If it is unreadable, reply exactly: [UNREADABLE]."
)


@dataclass
class Region:
    """One segmented page region with its routing decision."""

    kind: str  # "text" | "figure"
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points
    route: str = ""  # "text_layer" | "vlm" — filled by route_region()
    text: str = ""  # cheap-path text, or VLM caption
    xref: int | None = None  # image xref for figure regions (for region render)

    @property
    def reading_key(self) -> tuple[float, float]:
        # Top-to-bottom, then left-to-right. A real multi-column page would
        # bucket by column first; PyMuPDF blocks are already roughly ordered.
        return (round(self.bbox[1], 1), round(self.bbox[0], 1))


@dataclass
class HybridResult:
    markdown: str
    regions: list[Region] = field(default_factory=list)

    @property
    def n_text(self) -> int:
        return sum(r.route == "text_layer" for r in self.regions)

    @property
    def n_vlm(self) -> int:
        return sum(r.route == "vlm" for r in self.regions)


# --------------------------------------------------------------------------- #
# Stage 1 — SEGMENT: ask the layout oracle for regions (here: PyMuPDF).        #
# --------------------------------------------------------------------------- #
def segment_regions(pdf_path: str | Path, page_number: int = 0) -> list[Region]:
    """Return typed, bbox'd regions for one page.

    Text blocks (``block["type"] == 0``) carry their own glyph stream and become
    EASY ``text`` regions. Image blocks (``type == 1``) plus anything from
    ``get_image_info`` become HARD ``figure`` regions to be rendered + captioned.
    A real layout model (see ``WHERE_A_REAL_LAYOUT_MODEL_SLOTS_IN``) would also
    emit ``Table`` regions; PyMuPDF cannot reliably detect borderless ones.
    """
    try:
        import pymupdf  # PyMuPDF, installed by the `extraction` extra
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "PyMuPDF not installed. Run: uv sync --extra extraction"
        ) from exc

    regions: list[Region] = []
    with pymupdf.open(str(pdf_path)) as doc:
        page = doc[page_number]
        data = page.get_text("dict")

        for block in data["blocks"]:
            bbox = tuple(round(v, 2) for v in block["bbox"])  # (x0,y0,x1,y1)
            if block.get("type") == 0:  # text block
                spans = [
                    span["text"]
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                ]
                text = " ".join(s for s in spans if s).strip()
                if text:
                    regions.append(Region(kind="text", bbox=bbox, text=text))
            elif block.get("type") == 1:  # image block
                regions.append(Region(kind="figure", bbox=bbox))

        # `get_image_info(xrefs=True)` catches images that `get_text` may miss
        # (e.g. drawn via XObject reuse) and gives us the xref for region render.
        seen = {r.bbox for r in regions if r.kind == "figure"}
        for info in page.get_image_info(xrefs=True):
            bbox = tuple(round(v, 2) for v in info["bbox"])
            if bbox in seen:
                continue
            xref = info.get("xref") or None
            regions.append(Region(kind="figure", bbox=bbox, xref=xref))
            seen.add(bbox)

    regions.sort(key=lambda r: r.reading_key)
    return regions


# --------------------------------------------------------------------------- #
# Stage 2 — ROUTE: classify each region's difficulty → cheap or VLM path.     #
# --------------------------------------------------------------------------- #
def route_region(region: Region) -> Region:
    """Decide which extractor handles this region (cost-aware routing).

    The whole point of the method: spend VLM dollars only where the cheap path
    cannot win. Text blocks already carry their glyphs → ``text_layer`` (free).
    Figures have no usable text layer → ``vlm`` (render the crop + caption it).
    A real router also sends ``Table`` regions and *scanned* (image-only) pages
    to the VLM; here scanned pages surface as one big ``figure`` region with no
    text blocks, which routes to the VLM automatically.
    """
    region.route = "text_layer" if region.kind == "text" else "vlm"
    return region


# --------------------------------------------------------------------------- #
# Stage 3 — EXTRACT: run each region through its chosen path.                  #
# --------------------------------------------------------------------------- #
def _render_region_png(pdf_path: str | Path, region: Region, page_number: int, dpi: int) -> bytes:
    """Render ONLY this region's bbox to PNG bytes (mirrors render_page_png, clipped)."""
    import pymupdf  # already guarded by segment_regions()

    with pymupdf.open(str(pdf_path)) as doc:
        page = doc[page_number]
        clip = pymupdf.Rect(*region.bbox)
        return page.get_pixmap(clip=clip, dpi=dpi).tobytes("png")


def _caption_figure_with_vlm(png: bytes, model: str) -> str:
    """Send one cropped figure PNG to the VLM via the adapter. Caption-only."""
    from langchain_core.messages import HumanMessage

    b64 = base64.b64encode(png).decode()
    vlm = get_llm("google", model=model, temperature=0)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": FIGURE_PROMPT},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        ]
    )
    out = vlm.invoke([msg])
    return (out.content if hasattr(out, "content") else str(out)).strip()


def extract_region(
    region: Region,
    pdf_path: str | Path,
    page_number: int = 0,
    dpi: int = 200,
    model: str = "gemini-2.0-flash",
    vlm_enabled: bool = True,
) -> Region:
    """Fill ``region.text`` using the routed path.

    Text regions are already populated by the segmenter (the glyph stream *is*
    the answer — no work, no cost). Figure regions get rendered and captioned by
    the VLM. If the VLM is unavailable (no key/provider), we degrade to a bbox
    placeholder rather than crash — the merge step still produces valid Markdown.
    """
    if region.route == "text_layer":
        return region  # already extracted at segment time

    # VLM path.
    if not vlm_enabled:
        region.text = f"[figure at {region.bbox} — VLM disabled]"
        return region
    try:
        png = _render_region_png(pdf_path, region, page_number, dpi)
        region.text = _caption_figure_with_vlm(png, model)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 — a single bad region must not sink the page
        region.text = f"[figure at {region.bbox} — caption failed: {type(exc).__name__}]"
    return region


# --------------------------------------------------------------------------- #
# Stage 4 — MERGE: reassemble regions by bbox into one reading-order stream.   #
# --------------------------------------------------------------------------- #
def merge_regions(regions: list[Region]) -> str:
    """Stitch text + figure captions back together in reading order.

    Text regions emit their prose; figure regions emit a Markdown image ref
    ``![caption](figure)`` so the chunker keeps a stable figure reference with
    the VLM caption carried inline (FINDINGS.md §5: figures as file-refs +
    captions). Ordering is by bbox top-then-left.
    """
    ordered = sorted(regions, key=lambda r: r.reading_key)
    parts: list[str] = []
    for r in ordered:
        if r.route == "text_layer":
            parts.append(r.text)
        else:
            caption = r.text or "[UNREADABLE]"
            parts.append(f"![{caption}](figure)")
    return "\n\n".join(p for p in parts if p).strip()


# --------------------------------------------------------------------------- #
# Primary function — the whole hybrid pipeline for one page.                   #
# --------------------------------------------------------------------------- #
def hybrid_extract(
    pdf_path: str | Path,
    page_number: int = 0,
    dpi: int = 200,
    model: str = "gemini-2.0-flash",
    vlm_enabled: bool = True,
) -> HybridResult:
    """SEGMENT → ROUTE → EXTRACT (per region) → MERGE, for one page."""
    regions = [route_region(r) for r in segment_regions(pdf_path, page_number)]
    for r in regions:
        extract_region(r, pdf_path, page_number, dpi, model, vlm_enabled)
    markdown = merge_regions(regions)
    return HybridResult(markdown=markdown, regions=regions)


def _google_available() -> bool:
    """True iff the google provider has a key (so the VLM path can actually run)."""
    try:
        from shared.llm import available_providers
    except Exception:  # noqa: BLE001
        return False
    return "google" in available_providers()


def demo(pdf_path: str | Path) -> None:
    """Run the hybrid router on the first page and print a short routing report."""
    vlm_on = _google_available()
    if not vlm_on:
        print(
            "GOOGLE_API_KEY / google provider not configured — running the cheap "
            "path only (figures become bbox placeholders).\n"
            "Set GOOGLE_API_KEY and `uv add langchain-google-genai` for VLM captions.\n"
        )
    try:
        result = hybrid_extract(pdf_path, vlm_enabled=vlm_on)
    except RuntimeError as exc:  # missing dep — expected, explain and stop
        print(f"skipped: {exc}")
        return

    print(f"Hybrid region routing on: {Path(pdf_path).name}")
    print(
        f"  regions: {len(result.regions)}  "
        f"(text-layer: {result.n_text}, VLM: {result.n_vlm})"
    )
    print("  --- merged Markdown (first 800 chars) ---")
    preview = result.markdown[:800]
    print(preview if preview else "[empty — likely a scanned page with no text layer]")
    if len(result.markdown) > 800:
        print(f"  ... (+{len(result.markdown) - 800} more chars)")


def _first_present_pdf() -> Path | None:
    """Pick the first golden-set PDF that exists on disk."""
    try:
        from projects.vlm_extraction_harness.golden_set import GOLDEN_SET
    except Exception:  # noqa: BLE001 — fall back to a bare sample dir scan
        from shared.settings import settings

        sample_dir = settings.data_dir / "sample_docs"
        pdfs = sorted(sample_dir.glob("*.pdf"))
        return pdfs[0] if pdfs else None
    for page in GOLDEN_SET:
        if page.exists:
            return page.pdf_path
    return None


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else _first_present_pdf()
    if target is None or not Path(target).is_file():
        print(
            "No PDF to run on. Pass a path as argv[1], or add a golden-set PDF "
            "(e.g. g5_figure_heavy.pdf) under data/sample_docs/ per its README."
        )
        raise SystemExit(0)
    demo(target)
