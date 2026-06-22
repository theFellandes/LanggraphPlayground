# TEST-PLAN.md — Gemini VLM Test-Harness Spec

> Companion to `FINDINGS.md` §11 (Known open questions). The findings are explicit that
> **Turkish, Arabic, and per-tool quality are currently undefendable** — no verified OCRTurk
> or KITAB-Bench score exists for any candidate tool (§11 gaps 3, 4, 11, 12). This document is
> the **runnable-on-paper** spec for the harness that closes those gaps on a *local* golden set.
>
> Downstream consumers: **Guillotine** (the user's own TR/AR/EN chunker) and
> **`projects/rag_qa_api_pro/`** (the capstone RAG ingestion service).

| Field | Value |
|---|---|
| **Spec date** | 2026-06-22 |
| **Owner** | theFellandes |
| **Purpose** | Decide whether a tool is **recommended for `rag_qa_api_pro`** by measuring it on a hand-curated golden set instead of trusting vendor benchmarks. |
| **Method-under-test (MUT)** | `Gemini-direct` (page-image → Markdown, via `get_llm("google", …)`) vs the top OSS library (`Docling`, §1) vs the born-digital baseline (`pymupdf4llm`). |
| **Repo constraints honoured** | switchable `get_llm("google", …)` adapter • `uv` only • Guillotine TR/AR + chunker-friendliness • `rag_qa_api_pro` production path |
| **Status** | **Spec only.** Building and running the harness is a separate later step (see §8). Nothing here is executed yet. |

---

## 1. Wiring — add Gemini through the documented adapter path

The extractor calls a VLM **only** through `get_llm("google", …)` so it is switchable and
inherits the `Runnable.with_fallbacks` chain. **No hard-coded `ChatGoogleGenerativeAI` /
`google.generativeai` call appears anywhere in the harness** — that is the single
non-negotiable wiring rule. This is the exact recipe from `shared/llm/README.md` →
"Adding a new provider", quoted verbatim so the harness stays in lock-step with the repo.

**Step 1 — install (uv ONLY, never pip):**

```bash
uv add langchain-google-genai
```

**Step 2 — create `shared/llm/google_adapter.py`** (one provider file per backend,
Mitrailleuse/Aletheia style; mirrors `anthropic_adapter.build()`):

```python
# shared/llm/google_adapter.py  (new file)
from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

from shared.settings import settings

DEFAULT_MODEL = "gemini-2.0-flash"


def build(model: str | None = None, **kwargs) -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=model or DEFAULT_MODEL,
        google_api_key=settings.google_api_key,
        **kwargs,
    )
```

**Step 3 — add the key to `shared/settings.py`** (next to `anthropic_api_key` /
`openai_api_key`, and extend the `Provider` literal so the type checker knows `"google"`):

```python
# shared/settings.py
Provider = Literal["anthropic", "openai", "google"]   # ← add "google"

class Settings(BaseSettings):
    ...
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None                 # ← new
```

**Step 4 — register in `shared/llm/base.py`'s `_ADAPTERS`** (insertion order = fallback
preference; the new provider auto-joins the chain when its key is set):

```python
# shared/llm/base.py
from shared.llm import anthropic_adapter, openai_adapter, google_adapter

_ADAPTERS = {
    "anthropic": anthropic_adapter,
    "openai":    openai_adapter,
    "google":    google_adapter,        # ← new; auto-joins the fallback chain
}
```

**Step 5 — the extractor's only entry to a VLM** (matches `FINDINGS.md` §6.3 — captioning
goes through the same door):

```python
from shared.llm import get_llm

# switchable + fallback-able. media_resolution high for dense docs (bestpractices §6).
vlm = get_llm("google", model="gemini-2.0-flash", temperature=0)
# vlm.invoke([HumanMessage(content=[{"type": "text", ...},
#                                   {"type": "image_url", "image_url": "data:image/png;base64,..."}])])
```

> **Wiring assertion (the harness test that proves compliance):**
> `grep -rE "ChatGoogleGenerativeAI|google\.generativeai|import genai" <harness_dir>` must
> return **only** `shared/llm/google_adapter.py`. Any other hit fails the plan.

---

## 2. Golden set — hand-curated hard pages under `data/sample_docs/`

Six pages, one per hard case. Hand-curate a **ground-truth Markdown** (and HTML for tables)
for each — the harness scores *against* this, not against another model. Keep each page's
source license noted. Place rendered page PNGs at **300 DPI** (dense content; bestpractices §6)
beside the PDFs so every method scores the *same* pixels.

| # | Filename | Stresses | Expected-hard-part (where tools break) |
|---|---|---|---|
| G1 | `g1_born_digital_prose.pdf` | Born-digital single-column prose, real text layer | Should be *easy* — the baseline. Catches over-engineering: a VLM that paraphrases or drops a paragraph here is disqualified. `pymupdf4llm` must near-ace it. |
| G2 | `g2_scanned_page.pdf` | A **scanned** (image-only, no text layer) page | `pymupdf4llm` emits **silent blank output** (methodology §4 failure mode) — proves the born-digital baseline's hard floor. Only OCR/VLM paths recover text. |
| G3 | `g3_multicolumn_paper.pdf` | Two-column academic page with footnotes crossing columns | **Reading-order scramble** — columns merged L-to-R instead of column-first; footnotes extracted out of sequence (methodology §1, §4). The classic multi-col failure. |
| G4 | `g4_table_heavy.pdf` | Dense table with **merged cells** (colspan/rowspan) + numeric data | Markdown pipes cannot express merged cells (§9 claim 6) — ground truth is **HTML**. Catches hallucinated cells (the ~12.4% dense-text rate, §9 claim 8) and merged columns. |
| G5 | `g5_figure_heavy.pdf` | Page with figures/charts + captions | Figure-pixel extraction + caption association + chart-to-data. Traditional parsers score <6% on charts; VLMs 64-78% (§11 gap 8). Tests the `caption_figure` metadata path (§6.3). |
| G6 | `g6_turkish_arabic.pdf` | A **Turkish** page (dotless-ı / dotted-İ, ş/ğ/ç diacritics) **and/or an Arabic** RTL page | The Guillotine-critical case. §11 gaps 3, 4: no verified TR/AR score exists. Tests Turkish casing trap and Arabic RTL reading-order + glyph joining. **Curate one of each if budget allows (`g6_turkish.pdf`, `g6_arabic.pdf`).** |

**Curation rules:**

- Pages must be **real documents**, not synthetic — PureDocBench (§benchmarks) shows synthetic
  pages mislead; rankings reverse under real degradation.
- For each page, store: `*.pdf`, `*_page.png` (300 DPI), `*.gold.md`, and `*.gold.tables.html`
  (G4 only). Tables in ground truth are **HTML** (§9 claim 6).
- G6 ground truth must be transcribed by a TR/AR reader, not a model — it is the one case where
  model-vs-model scoring is circular (§5 known biases).
- Keep the set **tiny and hard** (6–8 pages). This is a *discriminator*, not a leaderboard:
  one of each failure mode beats 500 easy pages.

---

## 3. What to compare — three methods, same golden set

All three run on the **identical** rendered PNGs (300 DPI) / source PDFs so differences are
method, not input.

| Method | What it is | Why it's in the bake-off | Wiring |
|---|---|---|---|
| **A. Gemini-direct** | Page-image → Markdown prompt, one call per page | The in-repo VLM path; the thing we're evaluating for the hard cases | `get_llm("google", "gemini-2.0-flash")` — **adapter only** (§1) |
| **B. Docling** (top OSS, §1) | IBM ensemble pipeline → native Markdown + reading-order tree | The recommended primary extractor; MIT-clean, first-party `langchain-docling` loader | `uv add docling langchain-docling`; local, CPU ok |
| **C. pymupdf4llm** | Text-layer extraction → Markdown | The born-digital **baseline** — proves when a VLM is overkill (§FINDINGS "When a VLM is overkill") and exposes its hard floor on G2 | `uv add pymupdf4llm`; local, CPU, sub-second/page |

**The Gemini-direct prompt** (structured-output discipline from bestpractices §6 — metadata
before text, anti-hallucination, HTML tables, column-first reading order):

```text
Return the content of this page as GitHub-flavored Markdown in natural reading order.
- If the page has multiple columns, read the leftmost column fully before the next.
- Render every table as valid HTML (<table>/<th>/<td> with colspan/rowspan). Never use Markdown pipes for merged cells.
- Render formulas as LaTeX ($...$).
- For each figure, emit ![<one-sentence caption>](figure) and nothing invented.
- Do not hallucinate. If a region is unreadable, write [UNREADABLE].
```

Temperature 0 for repeatability; raise to 0.7 only on a retry that hit a repetition loop
(bestpractices §6 / reliability). Each method writes `*.pred.md` (+ `*.pred.tables.html`) per page.

> **A note on prompt-vs-tool fairness:** B and C have no prompt knobs; A does. Lock the prompt
> above for the whole run and version it next to the results so the comparison is reproducible.

---

## 4. Metrics — explicit scoring rule per metric

Seven metrics. Each row gives the **unit, the rule, and the pass direction**. Scores are
per-page then averaged per method; report per-page so G6/G4 outliers stay visible.

| Metric | Unit | Scoring rule | Pass direction |
|---|---|---|---|
| **Reading-order correctness** | 0–1 | Normalized Kendall-τ between the sequence of ground-truth text blocks and their order in `*.pred.md` (match blocks by fuzzy text ≥ 0.9 ratio). 1.0 = perfect order; penalize column-merge and out-of-sequence footnotes. Primary signal on **G3**. | higher better |
| **Table-cell accuracy** | 0–1 (TEDS) | Tree-Edit-Distance Similarity between predicted table tree and `*.gold.tables.html`. Use HTML so colspan/rowspan count (§9 claim 6). Report **per-table**; flag any **hallucinated cell** (cell present in pred, absent in gold) as an automatic sub-fail regardless of TEDS. Primary signal on **G4**. | higher better |
| **Formula correctness** | 0–10 | LLM-judge semantic score (§5) of predicted LaTeX vs gold LaTeX (the ICPR/BMVC 2026 metric correlates r≈0.78–0.93 with humans, vs CDM r≈0.34). 10 = semantically identical. Only scored on pages containing formulas. | higher better |
| **Figure extraction + caption quality** | two sub-scores | (a) **Detection** = figures emitted / figures in gold (recall; missing-figure = 0). (b) **Caption** = LLM-judge 0–5 on whether the caption matches the figure pixels + for charts lists series/axes. Primary signal on **G5**. | higher better |
| **Markdown validity** | pass/fail + 0–1 | Parse `*.pred.md` with a CommonMark parser → must not error (hard gate). Then 0–1 = fraction of expected headings preserved (so `MarkdownHeaderTextSplitter` works) + tables that are well-formed HTML. Guillotine-critical. | higher better |
| **Cost / page** | USD | A: input-tokens × Gemini price + output-tokens × price (log actual token counts from the response; ~258 input tokens/page at media_resolution medium per bestpractices §6). B, C: `$0` API + measured wall-clock × local-compute rate (note as `$0 + <s>` since local). | lower better |
| **Latency / page** | seconds | Wall-clock per page, p50 and p95 across the set. For A, include API round-trip; record retries separately (design for ~12% retry rate, bestpractices §6 reliability). | lower better |

**Aggregation:** no single composite number — report the table per method. A composite hides the
G6 (TR/AR) and G4 (table) failures that actually decide `rag_qa_api_pro` suitability.

---

## 5. LLM-as-judge (optional second pass) — rubric + known biases

For the subjective metrics (formula correctness, caption quality, and a holistic
"extraction-vs-page-image faithfulness" check), an optional judge scores each `*.pred.md`
**against the original page PNG**. Wire the judge through the **same adapter** — either
`get_llm("google", …)` or `get_llm("anthropic", …)` — never a raw SDK call (ties this to the
evaluation discipline already in the repo).

**Judge rubric (0–5 each, page image + prediction supplied):**

1. **Faithfulness** — every fact/number in the prediction is present in the page image (no invention).
2. **Completeness** — no paragraph, row, footnote, or figure dropped.
3. **Structure** — headings, reading order, and table shape match the visual layout.
4. **Table fidelity** — merged cells preserved; numbers in the right cells.
5. **Script integrity** (G6) — Turkish casing (ı/İ, ş/ğ/ç) and Arabic RTL order correct.

Judge prompt fixes temperature 0, requires a per-criterion integer + a one-line justification,
and is told *"If a region is unreadable in the image, do not penalize the prediction for omitting it."*

**Known biases — state them so the judge's numbers are read with caution:**

| Bias | What it does here | Mitigation in this plan |
|---|---|---|
| **Verbosity bias** | Longer extractions score higher even when padded. | Cap by also checking against gold length; flag predictions > 1.3× gold tokens. |
| **Self-preference / self-enhancement** | A Gemini judge favours the Gemini-direct (method A) output. | **Cross-judge:** if A is Gemini, the judge is Claude (`get_llm("anthropic", …)`), and vice-versa. Never let a model judge its own family on the deciding metrics. |
| **Position bias** | The first/last candidate shown scores higher. | Randomize candidate order per page; the judge sees one prediction at a time, not a ranked list. |

> The judge is a **tie-breaker and a smell-test**, not the primary metric. The deterministic
> metrics in §4 (TEDS, Kendall-τ, Markdown-parse gate) are authoritative; the judge explains
> *why* a method lost, especially on G6 where deterministic TR/AR ground truth is scarce.

---

## 6. Pass thresholds — the bar for "recommended for `rag_qa_api_pro`"

A method must clear **all** of these on the golden set to earn a "recommended" verdict. These
are gates, not averages — a method that aces five pages and silently blanks G2 is **not** fit
for a fault-tolerant ingestion path.

| Gate | Threshold | Rationale |
|---|---|---|
| **Markdown validity** | 100% of pages parse as CommonMark (hard gate) + ≥ 0.95 heading preservation | Guillotine's `MarkdownHeaderTextSplitter` breaks on invalid Markdown / lost headings. |
| **Reading-order** | mean Kendall-τ ≥ 0.90; **G3 ≥ 0.85** | Column scramble poisons chunk semantics (methodology §1 rag_symptom). |
| **Table-cell accuracy** | mean TEDS ≥ 0.85; **zero hallucinated cells** on G4 | A hallucinated number in a RAG index is a high-confidence wrong answer (§9 claim 8). Hard zero-tolerance on invented cells. |
| **Formula correctness** | ≥ 8.0 / 10 on formula pages (if any) | Below this, numeric lookups fail silently. |
| **Figure detection** | recall ≥ 0.90; caption quality ≥ 3.5 / 5 | Figures carry as chunk metadata (§6.3); a dropped figure is a dropped citation. |
| **TR/AR (G6)** | script-integrity ≥ 4 / 5 **and** no RTL reading-order inversion | The Guillotine multilingual requirement. §11 gaps 3–4: this is the gate the vendor benchmarks cannot answer — measure it here. |
| **No silent failure** | every page produces non-empty output **or** a logged `[UNREADABLE]` / raised error | `rag_qa_api_pro` must be **observable + fault-tolerant**: a blank page must be detectable, never silently missing. Disqualifies bare `pymupdf4llm` on G2 → proves the hybrid route-by-difficulty design. |
| **Cost / latency** | within budget for the re-index cadence (set per deployment; record p95) | Batchable + cost-bounded; VLM-direct on *every* page is the anti-pattern (§FINDINGS "When a VLM is overkill"). |

**Expected outcome (hypothesis, to be confirmed by the run):** `pymupdf4llm` passes G1, fails the
"no silent failure" gate on G2; `Docling` passes most, contested on G4 dense tables and G6 TR/AR;
`Gemini-direct` strong on G4/G5/G6 but loses on cost/latency and risks hallucinated cells. This is
exactly why the recommended production shape is **hybrid route-by-difficulty** (§FINDINGS §5, §8),
not any single method — the harness exists to *measure* that, not assume it.

---

## 7. Anti-patterns (smell to fix)

- **Hard-coded `ChatGoogleGenerativeAI(...)` in the harness.** Violates the adapter rule; loses
  the fallback chain. → Always `get_llm("google", …)` (§1 wiring assertion).
- **`pip install` anything.** → `uv add` only.
- **Markdown pipe tables in ground truth.** Cannot express merged cells → false TEDS wins for
  tools that flatten structure. → Gold tables are HTML (§9 claim 6).
- **Model-vs-model scoring with no human gold on G6.** Circular; a Gemini judge blesses Gemini
  Turkish. → Human-transcribed TR/AR gold + cross-family judge (§5).
- **One composite score.** Hides the G2/G4/G6 failures that actually disqualify a tool. → Report
  per-page, gate per-metric (§6).
- **Synthetic golden pages.** Rankings reverse under real degradation (PureDocBench, §benchmarks).
  → Real documents only (§2).

---

## 8. Note — this is the runnable-on-paper spec, not the harness

Everything above is **specification**. Building the harness (the runner that fans pages out across
methods, the metric implementations, the judge wiring) and **running it** is a **separate later
step**. When that step happens:

- It will live under the harness directory and call extractors **only** through `shared/llm`
  (Gemini) and the documented OSS loaders (`docling`, `pymupdf4llm`), installed with `uv add`.
- The LangGraph shape for the runner (fan-out pages via Send API, bounded by an
  `asyncio.Semaphore`, fan-in to a per-method scorer) is sketched in `FINDINGS.md` §4 / §11 gap 6
  and is intentionally **out of scope here** — this document only defines *what to measure and
  what bar to clear*.

Until then, this plan is fully reviewable on paper: a reader can check the wiring recipe against
`shared/llm/README.md`, the golden-set coverage against the §11 gaps, and the thresholds against
the `rag_qa_api_pro` requirements — without executing a line.
