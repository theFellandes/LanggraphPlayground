# `demos/` — one runnable demo per extraction method

These are **runnable, one-per-method demos** for the 7 PDF-extraction methods plus the image (figure) lifecycle — a hands-on companion to the research write-ups in [`../../../docs/research/vlm-pdf-extraction/METHODS-ARCHITECTURE.md`](../../../docs/research/vlm-pdf-extraction/METHODS-ARCHITECTURE.md) (the per-method architecture, ASCII diagrams, and image-handling deep-dive) and [`FINDINGS.md`](../../../docs/research/vlm-pdf-extraction/FINDINGS.md) (the §4 method taxonomy and §5 reliability playbook). Each file is self-contained: it isolates exactly one method end-to-end, lazy-imports its heavy dependencies so the module always imports clean, runs on the first PDF it finds under `data/sample_docs/` (or a path you pass), and **gracefully skips** with an install hint when a dependency or API key is missing — nothing here ever hard-crashes on a cold checkout.

## The demos

| Demo file | Method | Deps to install | Needs API key? | What it shows |
|---|---|---|---|---|
| `m1_text_layer.py` | 1 — Text-layer extraction | `uv sync --extra extraction` | No | Reads the PDF's embedded glyph stream straight to Markdown — no render, no OCR; born-digital only, with a char-count gate. |
| `m2_classical_ocr_layout.py` | 2 — Classical OCR + layout model | `uv sync --extra extraction` + `uv add pytesseract paddleocr` (+ native Tesseract on PATH) | No | Rasterize → layout-detect regions → OCR each → reassemble Markdown; the CPU/offline pre-VLM baseline (PP-StructureV3 + Tesseract fallback). |
| `m3_specialised_doc_vlm.py` | 3 — Specialised document VLM | `uv add docling docling-core pymupdf` + `uv sync --extra extraction` | No | One local OCR-free transformer (Docling / Granite-Docling) image→DocTags→Markdown, keeping real figure-pixel crops. |
| `m4_frontier_vlm_direct.py` | 4 — Frontier-VLM direct (page-as-image) | `uv sync --extra extraction` + `uv add langchain-google-genai` | **Yes** (`GOOGLE_API_KEY`) | Render a 300-DPI PNG and prompt a frontier VLM (Gemini via `get_llm`) to transcribe the whole page to Markdown. |
| `m5_hybrid_region_routing.py` | 5 — Hybrid pipeline (region routing) | `uv sync --extra extraction` + `uv add langchain-google-genai` | **Yes** (`GOOGLE_API_KEY`) | Segment → route by difficulty → cheap text-layer on prose, VLM on figures/tables only → merge by bbox into reading order. |
| `m6_agentic_langgraph.py` | 6 — Agentic extraction (LangGraph) | `uv add langgraph langchain-google-genai` + `uv sync --extra extraction` | **Yes** (`GOOGLE_API_KEY`) | A `StateGraph` that fans out one page-worker per page, self-checks confidence, re-renders low-confidence pages at higher DPI, and defers a fan-in assembler. Runs in mock mode without a key. |
| `m7_two_pass_verify.py` | 7 — Two-pass / verification | `uv sync --extra extraction` + `uv add langchain-google-genai` | **Yes** (`GOOGLE_API_KEY`) | Pass 1 drafts cheaply (pymupdf4llm); pass 2 a VLM judges the draft against the page image and returns image-grounded corrections only. |
| `image_lifecycle.py` | Figure lifecycle (cross-cutting) | `uv sync --extra extraction` + `uv add langchain-google-genai` | **Yes** (`GOOGLE_API_KEY`) | Extract a figure 3 ways (embedded XObject / clipped render / Docling crop) → caption via VLM → store → emit `![caption](path)` → index, carrying `{figure_path, caption, bbox, page, classification, source}`. |

## Setup

Use **`uv` only — never `pip`**.

```bash
# OSS extractors + the page renderer (PyMuPDF). Covers m1, m2, m3, m5, m7, image_lifecycle.
uv sync --extra extraction        # docling, pymupdf4llm, + PyMuPDF

# The Gemini adapter behind get_llm("google", ...). Needed for the VLM / agentic / two-pass demos
# (m4, m5, m6, m7, image_lifecycle). Then put your key in .env:
uv add langchain-google-genai
#   GOOGLE_API_KEY=...   in .env   (see ../../../shared/llm/README.md)

# Per-method extras on top of the base:
uv add pytesseract paddleocr      # m2 — also install the native Tesseract engine + lang packs on PATH:
#   Windows: UB-Mannheim Tesseract build + tur/ara language data
#   macOS:   brew install tesseract tesseract-lang
#   Linux:   apt-get install tesseract-ocr tesseract-ocr-tur tesseract-ocr-ara
uv add docling docling-core pymupdf   # m3 — specialised doc-VLM pipeline
uv add langgraph langchain-google-genai   # m6 — the agentic StateGraph
```

You only need the extras for the demos you actually run. With just `uv sync --extra extraction` and no key, `m1`/`m2` work fully and the VLM demos skip cleanly.

## Run

Each demo is a module. Run it against the first PDF present in `data/sample_docs/`, or pass an explicit path:

```bash
uv run python -m projects.vlm_extraction_harness.demos.m4_frontier_vlm_direct
uv run python -m projects.vlm_extraction_harness.demos.m4_frontier_vlm_direct data/sample_docs/g5_figure_heavy.pdf
```

Swap `m4_frontier_vlm_direct` for any file in the table above (`m1_text_layer`, `m2_classical_ocr_layout`, `m3_specialised_doc_vlm`, `m5_hybrid_region_routing`, `m6_agentic_langgraph`, `m7_two_pass_verify`, `image_lifecycle`). If a demo's dependency or API key is missing, it prints a one-line install/key hint and exits cleanly instead of crashing — so you can start with whichever method you have set up and grow from there.

## The non-negotiable wiring rule

VLMs are reached **only** through the project adapter:

```python
from shared.llm import get_llm
vlm = get_llm("google", model="gemini-2.0-flash", temperature=0)
```

No demo ever instantiates a provider SDK directly — that keeps every method provider-switchable and inherits the `Runnable.with_fallbacks` chain. The only place a raw SDK is constructed is `shared/llm/google_adapter.py`. This is enforced with a grep assertion that **must return nothing**:

```bash
# Run from the repo root — must print NOTHING for the demos to be considered correctly wired.
grep -rnE --include='*.py' "ChatGoogleGenerativeAI\(|google\.generativeai|import +genai" \
  projects/vlm_extraction_harness/demos
# → empty. (Docstrings naming the SDK in prose are fine; an actual call site is not.)
```
