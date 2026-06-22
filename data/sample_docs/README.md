# `data/sample_docs/` — golden set for the VLM extraction harness

Hand-curated PDF pages for the bake-off specified in
[`docs/research/vlm-pdf-extraction/TEST-PLAN.md`](../../docs/research/vlm-pdf-extraction/TEST-PLAN.md) §2.
Six pages, one per hard case. **Keep the set tiny and hard** — this is a
*discriminator*, not a leaderboard: one of each failure mode beats 500 easy pages.

The harness (`projects/vlm_extraction_harness/`) reads PDFs from this folder by the
`filename` below. **The PDFs themselves are not committed** — drop in real documents
(see curation rules) and, ideally, hand-author the ground-truth files next to each.

| # | Filename | Stresses | Expected hard part (where tools break) |
|---|---|---|---|
| G1 | `g1_born_digital_prose.pdf` | Born-digital single-column prose, real text layer | Should be *easy* — the baseline. Catches over-engineering; `pymupdf4llm` must near-ace it. |
| G2 | `g2_scanned_page.pdf` | A **scanned** (image-only, no text layer) page | `pymupdf4llm` emits **silent blank output** — proves the born-digital baseline's hard floor. Only OCR/VLM recover text. |
| G3 | `g3_multicolumn_paper.pdf` | Two-column academic page with footnotes crossing columns | **Reading-order scramble** — columns merged L→R instead of column-first. |
| G4 | `g4_table_heavy.pdf` | Dense table with **merged cells** (colspan/rowspan) + numeric data | Markdown pipes can't express merged cells — ground truth is **HTML**. Catches hallucinated cells. |
| G5 | `g5_figure_heavy.pdf` | Page with figures/charts + captions | Figure-pixel extraction + caption association + chart-to-data. |
| G6 | `g6_turkish.pdf` / `g6_arabic.pdf` | A **Turkish** page (dotless-ı / dotted-İ, ş/ğ/ç) **and/or an Arabic** RTL page | The Guillotine-critical case. No verified TR/AR score exists for any tool — measure it here. |

## Per-page ground-truth files (for scoring — TEST-PLAN §4)

For each page `gN_*.pdf`, store alongside it:

- `gN_*.pdf`            — the source PDF (real document, **not synthetic**)
- `gN_*_page.png`       — rendered page at **300 DPI** (so every method scores the same pixels)
- `gN_*.gold.md`        — hand-authored ground-truth Markdown
- `gN_*.gold.tables.html` — ground-truth tables as **HTML** (G4 only; colspan/rowspan)

## Curation rules

- **Real documents only.** PureDocBench shows synthetic pages mislead — rankings reverse under real degradation.
- **Tables in ground truth are HTML**, not Markdown pipes (merged cells).
- **G6 ground truth is transcribed by a TR/AR reader, not a model** — it is the one case where model-vs-model scoring is circular.
- Note each page's source license.

> The harness runs whatever PDFs are present and skips the rest, so you can start with
> one page (e.g. `g1_born_digital_prose.pdf`) and grow the set.
