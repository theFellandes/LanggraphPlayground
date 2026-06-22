# `vlm_extraction_harness/` — Gemini vs Docling vs pymupdf4llm

Runnable companion to the research in
[`docs/research/vlm-pdf-extraction/`](../../docs/research/vlm-pdf-extraction/FINDINGS.md).
It executes the bake-off specified in
[`TEST-PLAN.md`](../../docs/research/vlm-pdf-extraction/TEST-PLAN.md): the three
methods under test, over the hand-curated golden set in
[`data/sample_docs/`](../../data/sample_docs/README.md).

## What's here (and what isn't)

| Piece | Status |
|---|---|
| `extractors.py` — Gemini-direct (via `get_llm("google", …)`), Docling, pymupdf4llm | ✅ implemented |
| `golden_set.py` — the 6 hard-case pages (TEST-PLAN §2) as data | ✅ implemented |
| `run.py` — extract every present PDF with every method, write `*.pred.md`, print a status table | ✅ implemented |
| **Scoring** — Kendall-τ reading order, TEDS tables, Markdown-validity gate, LLM-as-judge (TEST-PLAN §4–§6) | ⏳ **next step, not implemented** |
| The golden-set **PDFs + ground truth** | ⬜ you provide (real docs — see the data README) |

The runner is the *extraction* pass only. Scoring is deliberately left as the
next step because it needs the hand-authored `*.gold.md` / `*.gold.tables.html`
ground truth that does not exist yet (TEST-PLAN is explicit that TR/AR and
per-tool quality are currently undefendable — the golden set is how you close that).

## Setup

```bash
# Gemini provider (lightweight; already declared in core deps)
uv add langchain-google-genai          # uv ONLY — never pip
# heavier OSS extractors + the page renderer
uv sync --extra extraction             # docling, langchain-docling, pymupdf4llm (+ PyMuPDF)
```

Set `GOOGLE_API_KEY` in `.env` (per [`shared/llm/README.md`](../../shared/llm/README.md)).
Gemini is reached **only** through `get_llm("google", …)` — no raw SDK call anywhere
(the non-negotiable wiring rule, TEST-PLAN §1).

## Run

```bash
# drop at least one PDF into data/sample_docs/ first (e.g. g1_born_digital_prose.pdf)
uv run python -m projects.vlm_extraction_harness.run
```

Methods with a missing dependency or API key report as **skipped** — the run never
crashes on a missing extractor, so you can start with just `pymupdf4llm` (no key) or
just Gemini and grow from there. Predictions land in `data/sample_docs/preds/`.

## Wiring assertion (TEST-PLAN §1)

```bash
grep -rE "ChatGoogleGenerativeAI|google\.generativeai|import genai" projects/vlm_extraction_harness
# → must return NOTHING. The only such reference lives in shared/llm/google_adapter.py.
```
