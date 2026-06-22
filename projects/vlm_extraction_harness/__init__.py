"""VLM PDF→Markdown extraction harness.

Runnable companion to `docs/research/vlm-pdf-extraction/TEST-PLAN.md`. Runs the
three methods under test (Gemini-direct via the `shared/llm` adapter, Docling,
pymupdf4llm) over the golden set in `data/sample_docs/`.

Install the extractor deps:  uv sync --extra extraction
Run:                         uv run python -m projects.vlm_extraction_harness.run
"""
