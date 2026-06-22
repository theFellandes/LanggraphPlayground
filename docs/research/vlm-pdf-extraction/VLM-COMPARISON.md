# Using your own VLM — and comparing VLMs head-to-head

> Companion to [`FINDINGS.md`](FINDINGS.md), [`METHODS-ARCHITECTURE.md`](METHODS-ARCHITECTURE.md),
> and [`TEST-PLAN.md`](TEST-PLAN.md). Answers one question: **what changes if I run the
> extraction on my own VLM instead of Gemini or Claude — and how do I prove which is
> better on my documents?** It ships a runnable comparison so you don't have to guess.

---

## 1. The short answer

Swapping in your own VLM is a **configuration change, not a code change.** Every
method in this repo reaches the model through one door — `get_llm(provider, …)` — so
the extraction pipeline never knows or cares which model answers. Point a `local`
provider at your server and the *exact same* `m4_frontier_vlm_direct`,
`m5_hybrid_region_routing`, `m6_agentic_langgraph`, `m7_two_pass_verify`, and
`image_lifecycle` demos run on **your** model.

What genuinely changes is the **trade-off profile**, and it is real:

| Dimension | Frontier API (Gemini / Claude / GPT) | Your own open VLM (Qwen2.5-VL, InternVL, MiniCPM-V, Llama-Vision, …) |
|---|---|---|
| **Out-of-box quality** | Strong on dense tables, formulas, reading order, arbitrary scripts | **Varies wildly by model & page type** — often close on prose, behind on complex tables / formulas / RTL |
| **Cost** | Per-token API bill (scales with volume) | **$0 marginal** per page — you pay for the GPU you already run |
| **Privacy** | Document leaves your network | **Stays in your VPC / on-prem** — the usual reason to self-host |
| **Throughput** | Provider rate limits + queue | Bounded by **your** GPUs; batch freely, no rate limit |
| **Control** | Fixed model, fixed updates | **Fine-tune on your own docs**; pin versions; no silent model swaps |
| **Latency** | Network round-trip | Local network — often lower p50, but depends on your hardware |
| **Maintenance** | None (vendor runs it) | **You run it** — serving, GPUs, upgrades, evals |

**The one rule that matters:** you cannot assume your VLM matches a frontier model —
you must **measure it on a golden set**. Open VLMs are competitive on prose and improving
fast, but the gap shows up exactly where extraction is hard: merged-cell tables, LaTeX
formulas, multi-column reading order, and Turkish/Arabic (the Guillotine cases). That is
what the comparison harness below is for.

> **Why this isn't just vibes:** [`FINDINGS.md`](FINDINGS.md) §10 found that even
> *frontier* VLM-direct carries a measured ~12.4% dense-text hallucination rate and that
> "local open models match frontier on tables/formulas" is **mixed/conditional**, not
> settled. So benchmark on *your* documents, not someone's leaderboard.

---

## 2. Wire your VLM — the `local` provider

Any OpenAI-compatible server works: **vLLM, Ollama, LM Studio, HF TGI** all expose the
OpenAI chat-completions API, so the repo reuses `ChatOpenAI` pointed at your `base_url`
([`shared/llm/local_adapter.py`](../../../shared/llm/local_adapter.py)) — no new SDK.

**Step 1 — serve your VLM** (example with vLLM):

```bash
# your box, not the repo — serves an OpenAI-compatible endpoint on :8000
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000
# Ollama equivalent: `ollama serve` then pull a vision model (e.g. llama3.2-vision)
```

**Step 2 — point the repo at it** (`.env`, see [`.env.example`](../../../.env.example)):

```bash
LOCAL_VLM_BASE_URL=http://localhost:8000/v1
LOCAL_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
LOCAL_API_KEY=EMPTY            # any token your server accepts
```

**Step 3 — use it like any provider** (already wired in `_ADAPTERS`):

```python
from shared.llm import get_llm
vlm = get_llm("local", with_fallback=False, temperature=0)   # YOUR model, no silent fallback
```

That's the whole change. `uv` only — never `pip`.

> **`with_fallback=False` matters.** `get_llm()` auto-promotes an *unconfigured* provider
> to a configured one. For a benchmark that would silently test the wrong model, so the
> comparison runner verifies a provider is genuinely available before building it and
> uses `with_fallback=False`. Set `LOCAL_API_KEY` (any token) or `local` is treated as
> "not configured" and skipped.

---

## 3. Run the comparison

[`projects/vlm_extraction_harness/compare_vlms.py`](../../../projects/vlm_extraction_harness/compare_vlms.py)
runs each available provider over every golden page that has a gold reference, scores it,
and writes a side-by-side report.

```bash
uv sync --extra extraction              # PyMuPDF page renderer
uv add langchain-google-genai           # only if you want the google column
# set provider keys in .env, then:
uv run python -m projects.vlm_extraction_harness.compare_vlms --providers google,anthropic,local
```

- `--providers google,anthropic,openai,local` — any subset; missing keys are **skipped
  with a reason**, never silently substituted.
- `--page 0` — which page of each golden PDF to score (samples are single-page).
- `--judge` — also run a cross-family LLM-as-judge (wiring provided; see §5).

**Output** (`data/sample_docs/comparison/`): a `rich` table to the console, plus
`report.md` and `results.json`. With no keys or no gold files it prints what's missing and
exits cleanly — nothing crashes on a cold checkout.

Example shape (numbers illustrative):

| Provider | RO τ | Table F1 | Halluc cells | Fig recall | MD valid % | Char sim | Latency s | $/page |
|---|---|---|---|---|---|---|---|---|
| google | 0.94 | 0.88 | 0 | 1.0 | 100% | 0.91 | 2.1 | 0.0004 |
| anthropic | 0.95 | 0.86 | 0 | 1.0 | 100% | 0.90 | 3.4 | 0.012 |
| local (Qwen2.5-VL-7B) | 0.88 | 0.71 | 2 | 0.8 | 90% | 0.83 | 0.9 | 0.0 |

---

## 4. The metrics ([`metrics.py`](../../../projects/vlm_extraction_harness/metrics.py))

Stdlib-only so it runs with zero extra installs. Each maps to TEST-PLAN §4; each row notes
the **rigorous upgrade** to swap in when you take it further.

| Metric | What it measures | Want | Rigorous upgrade |
|---|---|---|---|
| **`reading_order_tau`** | Kendall-τ of pred block order vs gold (fuzzy-matched) | →1.0 | `scipy.stats.kendalltau` on layout-model blocks |
| **`table_cell_f1`** | Multiset cell F1 of Markdown/HTML table cells | →1.0 | **TEDS** via `apted` + `lxml` (tree-edit, honors colspan/rowspan) |
| **`hallucinated_cells`** | Cells in pred absent from gold | **0 (hard gate)** | same, with cell-level alignment |
| **`figure_recall`** | `![](…)` / `<img>` refs found vs gold | →1.0 | bbox-matched figure detection (IoU) |
| **`formula_recall`** | LaTeX-span presence vs gold (coarse) | →1.0 | LLM-as-judge on rendered LaTeX (§5) |
| **`markdown_parses`** | CommonMark parse gate (+ heading preservation) | true / →1.0 | full `markdown-it-py` AST + `MarkdownHeaderTextSplitter` round-trip |
| **`char_similarity`** | Normalized char-level similarity (holistic) | →1.0 | semantic similarity / edit distance on normalized text |
| **latency / cost** | Wall-clock per page; token-price estimate | lower | measured live; prices in `PRICING` **churn — verify** |

These are the **honest floor**, not the last word — good enough to rank candidates and
catch regressions, explicitly labeled where a heavier metric belongs. The comparison
script is the "metric creation script" you asked for: take it from here.

---

## 5. LLM-as-judge (the subjective metrics)

For formula correctness, caption quality, and a holistic "extraction-vs-page-image"
check, score with a judge VLM — **cross-family** to dodge self-preference bias (if you're
testing Gemini, judge with Claude, and vice-versa; TEST-PLAN §5). `--judge` wires the
cross-family selection; plug your rubric (faithfulness / completeness / structure /
table-fidelity / script-integrity, 0–5 each) once your golden set is in place. Known
biases to control: verbosity, self-preference, position — mitigate with length caps,
cross-family judging, and order randomization.

---

## 6. You need a golden set first

The comparison scores against a **hand-authored `<stem>.gold.md`** beside each PDF in
[`data/sample_docs/`](../../../data/sample_docs/README.md). Without a gold reference there
is nothing to measure, so:

1. Drop real (not synthetic) PDFs covering the hard cases — born-digital, scanned,
   multi-column, table-heavy (merged cells), figure-heavy, and **Turkish/Arabic** (the
   Guillotine cases). See [`TEST-PLAN.md`](TEST-PLAN.md) §2.
2. Hand-transcribe each to `<stem>.gold.md` (tables as **HTML**); for TR/AR, a human
   transcribes — never a model (circular).
3. Re-run `compare_vlms`. Pages without a gold file are reported and skipped.

**Pass bar (TEST-PLAN §6):** to call your VLM "recommended for `rag_qa_api_pro`" it must
clear — `markdown_parses` 100%, mean RO τ ≥ 0.90, table F1 ≥ 0.85 with **zero
hallucinated cells**, figure recall ≥ 0.90, and the TR/AR gate — not just win on average.

---

## 7. So: should you use your own VLM?

- **Yes** when privacy/on-prem is non-negotiable, volume is high (marginal cost → your
  GPU, not a per-token bill), you can **fine-tune on your own document distribution**, or
  you need version stability. Self-hosting pays off most on *prose-heavy, high-volume*
  corpora.
- **Lean frontier** when you need best-in-class **tables / formulas / RTL** out of the box
  with zero infra, or your volume is low enough that the API bill is noise.
- **Best of both (the repo's recommendation):** the **hybrid** method (#5) — cheap local
  text-layer/OCR on the easy majority of pixels, and a VLM (yours *or* a frontier model
  via the same `get_llm` switch) only on the hard regions. Same fallback chain, lowest
  cost, and you decide per-corpus with the numbers from `compare_vlms`, not a vendor chart.
