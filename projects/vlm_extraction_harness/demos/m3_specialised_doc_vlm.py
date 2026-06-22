"""Method 3 — Specialised document VLM (OCR-free image→markup transformer).

Setup (uv only — never pip)::

    uv add docling docling-core pymupdf
    uv sync --extra extraction

Gist
----
One *end-to-end* model turns a page image straight into structured markup
(Markdown / DocTags / HTML), local and self-contained. No separate OCR engine,
no hand-built layout reading-order heuristics, no frontier-API round trip.

Docling is the runnable backbone here. It ships two pipelines:

* the **standard PDF pipeline** (layout + table-structure models, the recommended
  default, FINDINGS §8) — ``primary_extract`` below; and
* the **VLM pipeline** (``VlmPipeline``) — a single Granite-Docling / SmolDocling
  image→DocTags transformer, the purest expression of "specialised doc VLM" —
  ``granite_docling_extract`` below.

Both export *native* Markdown and, crucially for this method, can carve embedded
**figures out as real pixel crops** (``generate_picture_images``) rather than
inventing a caption for them — see ``extract_with_figures``.

Cousins worth knowing (all same family, image→markup, local): **Surya**,
**Nougat** (scientific PDFs / LaTeX-math), **GOT-OCR2.0**, **Granite-Docling**.

Heavy deps (``docling``, ``docling_core``, ``pymupdf``) are imported **lazily**
inside functions so importing this module never forces the ``extraction`` extra.
"""

from __future__ import annotations

import sys
import pathlib

# Make the demo runnable as a plain script: add repo root to sys.path.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path


def primary_extract(pdf_path: str | Path) -> str:
    """Docling **standard** pipeline → native Markdown (the recommended default).

    Layout model + table-structure model reassemble reading order and emit
    GitHub-flavored Markdown. This is the workhorse: fast, deterministic, no API
    key, no GPU strictly required. Verified API:
    https://docling-project.github.io/docling/usage/
    """
    try:
        from docling.document_converter import DocumentConverter
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "docling is not installed. Run: uv sync --extra extraction"
        ) from exc

    result = DocumentConverter().convert(str(pdf_path))
    return result.document.export_to_markdown()


def granite_docling_extract(pdf_path: str | Path) -> str:
    """Docling **VLM** pipeline → DocTags→Markdown via Granite-Docling.

    The purest "specialised document VLM": a *single* image→markup transformer
    (Granite-Docling / SmolDocling) replaces the whole layout+OCR+table stack.
    Downloads model weights on first run (transformers backend). Verified API:
    https://docling-project.github.io/docling/usage/vision_models/
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.datamodel import vlm_model_specs
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "docling VLM pipeline unavailable. Run: uv sync --extra extraction"
        ) from exc

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    return converter.convert(str(pdf_path)).document.export_to_markdown()


def extract_with_figures(pdf_path: str | Path, out_dir: str | Path) -> tuple[str, list[Path]]:
    """Extract Markdown **and** save every embedded figure as a real PNG crop.

    This is what makes method #3 distinct from "ask a model to describe the
    figure": Docling carves the actual figure pixels out of the page. Enable
    ``generate_picture_images``, iterate ``PictureItem`` elements, and save each
    via ``element.get_image(document)``. Markdown is written in REFERENCED mode
    so each ``![](figure-N.png)`` points at the saved crop. Verified against the
    official export_figures example:
    https://github.com/docling-project/docling/blob/main/docs/examples/export_figures.py
    """
    try:
        from docling_core.types.doc import ImageRefMode, PictureItem
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        raise RuntimeError(
            "docling / docling-core not installed. Run: uv sync --extra extraction"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # images_scale=2.0 -> ~144 DPI crops; generate_picture_images keeps the
    # real PIL pixels on each PictureItem instead of dropping them.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    conv_res = converter.convert(str(pdf_path))
    doc = conv_res.document

    saved: list[Path] = []
    fig_no = 0
    for element, _level in doc.iterate_items():
        if isinstance(element, PictureItem):
            fig_no += 1
            fig_path = out_dir / f"figure-{fig_no}.png"
            image = element.get_image(doc)
            if image is not None:
                image.save(fig_path, format="PNG")
                saved.append(fig_path)

    md_path = out_dir / "extract.md"
    doc.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED)
    markdown = md_path.read_text(encoding="utf-8")
    return markdown, saved


def demo(pdf_path: str | Path) -> None:
    """Entry point — run the standard pipeline and report figure extraction."""
    pdf_path = Path(pdf_path)
    print(f"[m3] specialised document VLM (Docling) on: {pdf_path.name}")

    try:
        markdown = primary_extract(pdf_path)
    except RuntimeError as exc:
        print(f"[m3] skipped: {exc}")
        return

    preview = markdown.strip().splitlines()[:12]
    print("[m3] --- standard-pipeline Markdown (first 12 lines) ---")
    for line in preview:
        print("    " + line)
    print(f"[m3] total Markdown chars: {len(markdown)}")

    # Bonus: pull real figure crops next to the source PDF.
    out_dir = pdf_path.parent / f"{pdf_path.stem}_m3_figures"
    try:
        _md, saved = extract_with_figures(pdf_path, out_dir)
        if saved:
            print(f"[m3] saved {len(saved)} figure crop(s) to {out_dir}")
            for p in saved:
                print(f"      - {p.name}")
        else:
            print("[m3] no embedded figures found on this document.")
    except RuntimeError as exc:
        print(f"[m3] figure extraction skipped: {exc}")

    print(
        "[m3] tip: for the pure OCR-free transformer, call "
        "granite_docling_extract(pdf) (downloads Granite-Docling weights)."
    )


def _pick_pdf() -> Path | None:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    try:
        from projects.vlm_extraction_harness.golden_set import GOLDEN_SET
    except ModuleNotFoundError:
        return None
    for page in GOLDEN_SET:
        if page.exists:
            return page.pdf_path
    return None


if __name__ == "__main__":
    pdf = _pick_pdf()
    if pdf is None or not Path(pdf).is_file():
        print(
            "[m3] no PDF found. Pass one explicitly:\n"
            "      uv run python projects/vlm_extraction_harness/"
            "m3_specialised_doc_vlm.py <path-to.pdf>\n"
            "    or drop a golden-set PDF into data/sample_docs/."
        )
        sys.exit(0)
    demo(pdf)
