# Research Prompt — VLM-Based PDF → Markdown Extraction ("All-In")

> **Status:** PROMPT ONLY. No research has been executed yet. This file
> is the brief that a research agent (or a human) runs to produce the
> companion deliverables listed in §9.
>
> **Created:** 2026-06-22
> **Owner:** theFellandes
> **Target deliverable folder:** `docs/research/vlm-pdf-extraction/`
> **Feeds into:** [Lesson 20 · Chunking & Parsing](../../../lessons/20_chunking_and_parsing/README.md) ·
> [Lesson 37 · Multimodal](../../../lessons/37_multimodal/README.md) ·
> [Lesson 36 · Library Landscape](../../../lessons/36_library_landscape/README.md) ·
> the [Guillotine](https://github.com/theFellandes/Guillotine) chunking project ·
> the `rag_qa_api_pro` capstone.

---

## 0 · How to run this prompt

This is a **gated** research brief. The work is divided into gates
**G0–G7**. Each gate has **entry criteria**, a **task**, and **exit
criteria**. **Do not advance to the next gate until the current gate's
exit criteria are fully met and checked off.** If a gate's exit
criteria cannot be met, stop and report the blocker rather than
proceeding on assumptions.

You may execute this prompt three ways:

1. **`deep-research` skill** — paste §1–§8 as the research question, keep §9–§12 as the output contract.
2. **`Workflow` tool** — fan one agent out per gate / per library row, verify findings adversarially (see §10), synthesise.
3. **Manual** — work the gates top-to-bottom yourself.

Whichever path: the **gates are mandatory checkpoints**, not
suggestions. They exist so that "we went all in" means *verified
breadth + verified depth*, not a pile of unverified blog summaries.

---

## 1 · Mission

Produce the **definitive, decision-ready map of how to extract PDFs
into clean Markdown using Vision-Language Models (VLMs) in 2026**, good
enough that this repo can (a) teach it, and (b) ship a production
extraction module that feeds the [Guillotine](https://github.com/theFellandes/Guillotine)
chunkers and the `rag_qa_api_pro` capstone.

The four pillars the user explicitly asked for — answer **all four**:

1. **Libraries** — what exists, what's alive, what's actually good.
2. **Methodologies** — the *approaches* (VLM-direct, OCR+layout, hybrid, agentic), not just tool names.
3. **Best practices** — how practitioners get reliable output at quality, cost, and scale.
4. **Image → Markdown** — *specifically*, how each library extracts **figures/images embedded in a PDF** and represents them in the Markdown output (the part most surveys skip).

**Bias toward VLM approaches** (the user is going "all in" on VLM), but
do not ignore strong non-VLM baselines — you must be able to say *when
a VLM is overkill*. **Gemini is the primary VLM for the test harness**
(see §11), but the survey must be provider-neutral.

### Non-goals (explicitly out of scope)

- General OCR for plain images unrelated to documents (receipts/photos belong to Lesson 37, not here).
- Audio/video multimodality.
- Re-teaching the chunkers — Guillotine owns chunking. This research stops at "clean Markdown + extracted assets handed to a chunker."
- Building the module *now*. This brief produces **research + a recommendation + a test plan**, not production code (unless a later gate is explicitly authorised).

---

## 2 · Repo context the researcher MUST honour

The recommendation has to fit *this* codebase. Before recommending
anything, internalise:

| Constraint | Where it lives | Implication for the recommendation |
|---|---|---|
| **Switchable provider + fallback chain** | [`shared/llm/`](../../../shared/llm/README.md) (`get_llm()`, `_ADAPTERS`, `with_fallbacks`) | A VLM extractor must plug into this adapter pattern — **no hard-coded SDK calls**. Gemini gets added as `shared/llm/google_adapter.py` exactly per the README's "Adding a new provider" recipe. |
| **`uv` only** | repo-wide | All install instructions use `uv add` / `uv run` — never `pip`. |
| **Adapter pattern (Mitrailleuse / Aletheia style)** | `shared/llm/`, user preference | One provider-specific file per backend, tiny factory. New extractors should mirror this shape. |
| **Polished, teaching-grade docs** | every `lessons/*/README.md` | Deliverables must be README-quality: tables, runnable snippets, anti-patterns, "when to use which." |
| **Existing coverage** | Lessons 20 / 36 / 37 | This research must go **deeper** than those READMEs (they're intros). Cross-link, don't duplicate. |
| **Downstream consumer = Guillotine** | [github.com/theFellandes/Guillotine](https://github.com/theFellandes/Guillotine) | Output format must be chunker-friendly: clean Markdown, preserved headings (for `MarkdownHeaderTextSplitter`-style splitting), tables as Markdown/HTML, and **figure references the chunker can carry as metadata**. Note Guillotine's multilingual focus (Turkish / Arabic / English) — extraction quality on those scripts matters. |
| **Capstone target** | `projects/rag_qa_api_pro/` | Extraction has to survive a real RAG ingestion path (batchable, fault-tolerant, observable). |

---

## 3 · Pillar 1 — Libraries (breadth gate)

Build a **verified inventory** of PDF→Markdown / document-understanding
tools. The seed list below is a **starting point, NOT a closed set** —
your job at G1 is to *confirm each is still maintained* and *find the
ones missing here*.

**Seed inventory (verify + expand — do not trust this list blindly):**

- **VLM-native / VLM-under-the-hood:** Docling + Granite-Docling-258M (IBM/DS4SD), Marker (VikParuchuri/Datalab), MinerU (OpenDataLab), Nougat (Meta), Surya, olmOCR (AllenAI), GOT-OCR2.0, Zerox (getomni-ai), MarkItDown (Microsoft), Mistral OCR, dots.ocr, SmolDocling.
- **Frontier-VLM-direct (no library, raw model):** Gemini 2.5 Pro/Flash, GPT-4o / GPT-4.1 vision, Claude Sonnet/Opus vision — page-image → Markdown via prompt.
- **Layout/OCR-classical (the baselines you must beat or justify):** Unstructured.io, PyMuPDF / `pymupdf4llm`, pdfplumber, PaddleOCR + PP-StructureV2, Camelot/Tabula (tables), Tesseract.
- **Hosted/commercial APIs:** LlamaParse (LlamaIndex), Reducto, Chunkr, Azure Document Intelligence, AWS Textract, Google Document AI, Adobe PDF Extract.

For **each** surviving library, the inventory row must capture the
**Master Rubric** dimensions in §6. A row with blanks is not done.

**Disambiguation note:** "Guillotine" in this repo refers to the
user's **own chunking project**, not an extractor. Do not list it as a
PDF parser. Treat it as the *consumer* of whatever this research
selects.

---

## 4 · Pillar 2 — Methodologies (approach taxonomy)

Tools come and go; **approaches** are the durable knowledge. Produce a
taxonomy of the *methods* for getting a PDF into Markdown, each with
"what it does / shines when / fails when / representative tool." At
minimum cover:

1. **Text-layer extraction** (born-digital PDFs; `pymupdf4llm`) — fast, free, breaks on scans/complex layout.
2. **Classical OCR + layout model** (Tesseract/Paddle + DocLayNet-style detector) — the pre-VLM SOTA.
3. **Specialised document VLM / OCR-free transformer** (Nougat, Surya, GOT-OCR, Granite-Docling) — trained end-to-end image→markup.
4. **Frontier VLM-direct, page-as-image** (Gemini/GPT-4o/Claude with a prompt) — most flexible, controllable rubric, cost/throughput risk.
5. **Hybrid pipelines** (layout detect → region-routing → VLM only on hard regions like tables/figures) — the production sweet spot; characterise it precisely.
6. **Agentic extraction** (a graph that renders → extracts → self-checks → re-renders failed pages) — map to a LangGraph shape (fan-out per Lesson 30, semaphore per Lesson 27).
7. **Two-pass / verification** (extract, then a second model validates structure/tables against the page image).

For each method, state the **failure modes** (reading order scramble,
hallucinated table cells, dropped footnotes, merged columns, RTL/script
issues, equation mangling) and which downstream symptom they cause in
RAG.

---

## 5 · Pillar 3 — Best practices

Synthesise the **operational playbook** — what separates a demo from a
pipeline. Cover at least:

- **Rendering:** DPI sweet spot (150 vs 300), page-image max-dimension per provider, when to tile.
- **Prompting the VLM** (for the direct approach): forcing valid Markdown, table format choice (Markdown vs HTML vs nothing), reading-order instructions, "don't hallucinate / return `[unreadable]`", per-page vs whole-doc context.
- **Cost & throughput:** token cost per page by model, batching, async fan-out + semaphore, caching, when a $0.0005 Flash call beats a local GPU.
- **Reliability:** retries, structured-output validation (tie to Lesson 04), the two-pass verify pattern, golden-set regression.
- **Tables, formulas, multi-column, footnotes, headers/footers** — the canonical hard cases and the known-good handling for each.
- **Local vs hosted** trade-off (privacy, GPU cost, latency, rate limits).
- **Anti-patterns** — write them in the Lesson-37 "smell → fix" table style.

---

## 6 · Master Comparison Rubric (the dimensions every library is scored on)

Every library in the §3 inventory gets scored on **all** of these.
This table *is* the spine of the deliverable.

| # | Dimension | What to record |
|---|---|---|
| 1 | **License & cost** | OSS license / paid API / per-page price |
| 2 | **Method** | which §4 methodology it is |
| 3 | **Local vs hosted** | runs offline? GPU needed? VRAM? |
| 4 | **Markdown quality** | does it emit real Markdown, or HTML/JSON you must convert? |
| 5 | **Reading order** | multi-column correctness |
| 6 | **Table fidelity** | tables → Markdown/HTML; merged cells; rotated tables |
| 7 | **Formula / math** | LaTeX output? quality? |
| 8 | **Figure / image handling** | **(Pillar 4 — see §7)** how embedded images are extracted + referenced |
| 9 | **Language / script** | non-Latin, **Turkish/Arabic (Guillotine-relevant)**, RTL |
| 10 | **Throughput** | pages/sec or pages/$; batch support |
| 11 | **Output structure** | reading-order JSON? bounding boxes? doc tree? |
| 12 | **LangChain/LangGraph fit** | existing loader? returns `Document`s? streamable? |
| 13 | **Maintenance health** | last release, stars, issue velocity, backing org |
| 14 | **Evidence grade** | high / moderate / low (per §10) — how well-sourced is this row |

---

## 7 · Pillar 4 — Image → Markdown (the deep dive the user specifically asked for)

This is the part most surveys hand-wave. Go deep. A PDF page contains
**embedded raster/vector figures, charts, diagrams, logos, stamps**.
The question: **what happens to those images when the document becomes
Markdown?** Answer, per library and per method:

1. **Extraction:** does it pull the figure out as a separate asset (PNG/JPEG) or rasterise the whole page? Where do the bytes go — file on disk, base64 inline, object store?
2. **Reference in Markdown:** `![alt](path)`? `<img>`? a placeholder token? a JSON sidecar with bounding boxes? Show the actual emitted Markdown.
3. **Captioning / alt-text:** does it (or can a VLM step) generate a description so the figure is *retrievable* by a text RAG? This is the bridge to Lesson 37's "VLM-summary indexing." Spell out the recipe.
4. **Figure classification:** chart vs table-as-image vs photo vs equation — does the tool tag them (e.g. Docling's picture classifier)?
5. **Bounding boxes & provenance:** are coordinates preserved so a chunk can point back to the source region? (Matters for Guillotine metadata and citation-grade RAG.)
6. **Round-trip integrity:** if you re-assemble the Markdown, do figures land in the right reading-order position relative to surrounding text?
7. **The recommended pattern:** end with a concrete, runnable recipe — "extract figure → VLM caption → embed `![caption](asset)` + index the caption text" — wired through `shared.llm.get_llm()` so it's provider-switchable.

Deliver a small **side-by-side**: same 2–3 figure-heavy PDF pages, show
how 3–4 top tools each represent the figures in their Markdown output.

---

## 8 · Benchmarks, datasets & evidence to consult

Don't rank tools on vibes. Find and cite the quantitative ground:

- **Benchmarks/leaderboards:** OmniDocBench, OCRBench / OCRBench v2, Marker's own benchmark suite, MinerU's reported numbers, Docling's eval, any 2025–2026 "PDF extraction shootout" leaderboards.
- **Datasets:** DocLayNet, PubLayNet, PubTabNet / FinTabNet (tables), the benchmark sets above.
- **Primary sources:** arXiv papers for Nougat, GOT-OCR2.0, olmOCR, Granite-Docling, Surya; official docs + GitHub READMEs (for feature claims) + release notes (for "is it alive").

**Recency rule:** this field moves monthly. Record the **access date**
for every source. Treat any benchmark older than ~12 months as
"verify it still holds" rather than fact.

---

## 9 · Deliverables (what running this prompt must produce)

Output into `docs/research/vlm-pdf-extraction/`, matching the
[`docs/research/` template](../../README.md) (framed question → search
log → evidence-graded findings → recommended pattern → full citations):

1. **`FINDINGS.md`** — the main synthesis:
   - Header: investigation date, sources screened → shortlisted → deep-read counts.
   - **TL;DR** (3–5 bullets, the actual recommendation).
   - Pillar 1: the **library comparison table** (full §6 rubric, every row evidence-graded).
   - Pillar 2: the **methodology taxonomy** table.
   - Pillar 3: the **best-practices playbook** + anti-patterns table.
   - Pillar 4: the **image→markdown deep dive** + the side-by-side.
   - **Decision tree:** "given {born-digital / scanned / table-heavy / figure-heavy / multilingual / budget / privacy}, use → X."
   - **Recommended stack for this repo** (primary + fallback), justified against §2 constraints.
   - **Evidence-grading table** (high / moderate / low) — required by repo convention.
   - Numbered citations with access dates.
2. **`TEST-PLAN.md`** — the Gemini harness spec (§11): golden set, metrics, how to run, pass thresholds.
3. **Update the folder `README.md`** (create it) — one-paragraph index + link table, matching `docs/research/README.md` style.
4. **Cross-link patch list** — the exact lines in Lessons 20 / 36 / 37 that should link here (propose; don't edit until approved).

Every claim that drives a recommendation carries a citation. No
recommendation rests on a single unverified source.

---

## 10 · Evidence & verification standards (how "extensive" is enforced)

- **Multi-source per claim:** a feature claim needs the official doc/README **and** corroboration (a benchmark, an independent test, or a second write-up). Marketing copy alone ≠ evidence.
- **Adversarial verification:** for every "X is best at Y" claim, actively look for a source that *refutes* it. If running as a Workflow, spawn a skeptic per top-N claim and keep only claims that survive. Record dissent.
- **Liveness check:** before recommending any OSS tool, confirm a release or meaningful commit within ~6 months. Flag abandoned-but-popular tools explicitly.
- **Grade everything:** every row/claim tagged **high** (primary source + independent corroboration + recent), **moderate** (one solid source), or **low** (single blog / unverified / stale). The TL;DR may only rest on **high/moderate**.
- **Show the search log:** databases/sites queried, counts screened → shortlisted → deep-read. Reproducibility over polish.

---

## 11 · Test harness spec — Gemini as the VLM under test

The user named **Gemini for testing**. The harness must validate the
recommendation empirically, not just on paper.

- **Wiring:** add Gemini via the documented path — `uv add langchain-google-genai`, create `shared/llm/google_adapter.py`, register in `_ADAPTERS` ([`shared/llm/README.md` §"Adding a new provider"](../../../shared/llm/README.md)). Extractor calls go through `get_llm("google", ...)` so it's switchable and fallback-able.
- **Golden set:** a small, hand-curated set of PDF pages under `data/sample_docs/` covering the hard cases — born-digital prose, a scanned page, a multi-column paper, a table-heavy page, a figure-heavy page, and a **Turkish or Arabic** page (Guillotine relevance).
- **What to compare:** Gemini-direct (page-image → Markdown prompt) **vs** the top OSS library from §3 **vs** a born-digital baseline (`pymupdf4llm`), on the same golden set.
- **Metrics:** reading-order correctness, table-cell accuracy, formula correctness, **figure-extraction + caption quality**, Markdown validity, cost/page, latency/page. Define each metric's scoring rule.
- **Judge option:** spec an LLM-as-judge (Gemini or Claude) scoring extraction-vs-page-image, tied to Lesson 35's evaluation discipline. State its rubric and its known biases.
- **Pass thresholds:** state the bar a tool must clear to be "recommended for `rag_qa_api_pro`."

> Note: building/running the harness is a **separate, later step**.
> G7 only requires the harness to be *fully specified and runnable on
> paper*, with the golden-set contents enumerated.

---

## 12 · The gates (mandatory checkpoints)

Work the gates in order. Check every exit box before advancing.

### G0 — Scope lock
**Task:** Restate the mission, confirm the §1 non-goals, and surface any assumption that, if wrong, changes the recommendation (e.g. "is local/offline a hard requirement?", "is there a per-page budget?", "Gemini-direct vs library-wrapping Gemini — both?").
**Exit:** [ ] Open scope questions listed and either answered from §2 or flagged for the user. [ ] No deep research started before this is done.

### G1 — Library breadth
**Task:** Confirm + expand the §3 inventory; kill dead tools; find missing ones.
**Exit:** [ ] ≥15 tools triaged. [ ] Each survivor has a liveness check (last release date). [ ] At least 3 tools found that were NOT in the §3 seed list (proves the search went beyond the prompt).

### G2 — Methodology taxonomy
**Task:** Produce the §4 approach taxonomy with failure modes.
**Exit:** [ ] All 7 methods characterised. [ ] Each mapped to ≥1 concrete tool and ≥1 RAG failure symptom.

### G3 — Rubric scoring (depth)
**Task:** Score every surviving library on the full §6 rubric.
**Exit:** [ ] No blank cells in the comparison table (use "unknown — unverified" explicitly, never silent blanks). [ ] Every row evidence-graded.

### G4 — Best-practices playbook
**Task:** Write §5, grounded in benchmarks (§8), not opinion.
**Exit:** [ ] Each practice cites a source or a benchmark number. [ ] Anti-patterns table written.

### G5 — Image → Markdown deep dive
**Task:** Answer all 7 questions in §7 + produce the figure side-by-side.
**Exit:** [ ] Actual emitted-Markdown samples shown for ≥3 tools. [ ] The runnable caption-and-index recipe written and wired through `get_llm()`.

### G6 — Synthesis & recommendation
**Task:** Decision tree + recommended repo stack + evidence-grading table.
**Exit:** [ ] Recommendation justified against every §2 constraint. [ ] TL;DR rests only on high/moderate evidence. [ ] Dissenting evidence from §10 acknowledged.

### G7 — Test plan & completeness critic
**Task:** Write `TEST-PLAN.md` (§11). Then run a **completeness critic**: what modality, claim, or hard-case did we skip? what's asserted but unverified?
**Exit:** [ ] Golden set enumerated. [ ] Metrics + thresholds defined. [ ] Critic's gaps either closed or logged as "known open questions" in `FINDINGS.md`.

---

## 13 · Appendix — claims to adversarially verify & recency landmines

Treat these as *suspect until proven on the access date* — they're the
field's most repeated, most stale assertions:

- "Docling is the best general-purpose PDF parser." (true in 2024 READMEs — re-check vs MinerU / Marker / Mistral OCR on current benchmarks.)
- "Just send the page image to a frontier VLM — it's solved." (cost, reading-order on dense multi-column, and table hallucination say otherwise — quantify.)
- "OCR is dead." (verify against scanned/handwritten/low-res cases where classical OCR or hybrid still wins.)
- "Markdown tables are enough." (merged/rotated cells often need HTML — check.)
- Any pricing or model-name figure — re-verify; multimodal prices and model IDs churn fast.
- Any benchmark leaderboard position older than ~12 months.

---

*End of prompt. Executing it produces the files in §9. Nothing in this
file is a finding — it is the question.*
