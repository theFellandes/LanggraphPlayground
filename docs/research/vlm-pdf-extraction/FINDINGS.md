# VLM & Document-Parser Landscape for PDF → Markdown Extraction (2026)

> Flagship research deliverable for `docs/research/`. Framed question → search log →
> evidence-graded findings → recommended pattern → numbered citations.
> Downstream consumers: **Guillotine** (the user's own multilingual TR/AR/EN chunker)
> and **`projects/rag_qa_api_pro/`** (the capstone RAG ingestion service).

| Field | Value |
|---|---|
| **Investigation date** | 2026-06-22 |
| **Owner** | theFellandes |
| **Question** | *Which document-extraction method + tool should feed clean, chunker-friendly Markdown (preserved headings, tables, figure refs) into Guillotine and survive the `rag_qa_api_pro` ingestion path — biased toward a VLM, plugged into `shared/llm`'s switchable adapter pattern, multilingual on Turkish/Arabic?* |
| **Sources screened → shortlisted → deep-read** | ~165 screened → 58 shortlisted → 31 deep-read (6 tools fully rubric-scored, 5 partially, 9 benchmarks, 7 methodologies, 18 best-practice notes) |
| **Recency rule** | Prefer sources dated 2025-09 onward; treat any source with `access_date = undefined` as **liveness-unconfirmed** and never let it carry a claim alone (see §10–§11). |
| **Repo constraints honoured** | switchable `get_llm("google", …)` adapter • `uv` only • Guillotine chunker-friendliness + TR/AR • `rag_qa_api_pro` production path |

---

## 1. TL;DR — the recommendation

These bullets rest **only on high/moderate-graded evidence**; VLM-direct and frontier-model claims are deliberately hedged (see §10–§11).

- **Primary extractor: `Docling` (MIT) with the standard pipeline + `MarkdownHeaderTextSplitter`-friendly output, and `langchain-docling` as the loader.** It is the only shortlisted tool that is (a) genuinely permissive (MIT core, Apache-2.0 Granite-Docling weights), (b) emits native Markdown with a real reading-order tree, and (c) ships a first-party LangChain loader — all three matter for an embeddable teaching/commercial service. *Do not* claim Docling is "best" on quality: it is **absent or mid-pack** on ParseBench (50.6%) and OmniDocBench did not publish its score; its formula path and dense-table fidelity are independently questioned. [1][2][14][27]
- **Hard-region / scanned / TR-AR fallback: a hosted VLM-native OCR API behind the adapter — `Mistral OCR 3` ($1–2/1k pages) as default, `Gemini` (`gemini-2.0-flash` via `shared/llm`) as the in-repo path.** A VLM is **overkill for born-digital PDFs** — route those to text-layer extraction — but is the only credible answer for scanned Arabic/Turkish, dense tables, and figure captioning. [10][29][30]
- **Method: a hybrid "route-by-difficulty" pipeline, not VLM-direct on every page.** Cheap classifier → text-layer for clean pages → VLM only for table/figure/scanned/RTL regions. This is the dominant production pattern in the 2026 best-practice corpus and the cheapest correct answer. [methodology §5; bestpractices §6] [33][34][35]
- **Tables as HTML, not Markdown pipes; figures as file-refs with VLM captions carried as chunk metadata.** Markdown has no colspan/rowspan; emit `<table>` and let Guillotine keep it intact. Figure pixels go to disk; a `gemini`-generated caption + bbox ride along as metadata. [bestpractices][37]
- **License-safety is a first-class filter.** `Marker`/`Surya` (RAIL), `MinerU` (unresolved AGPL/YOLO conflict), `PyMuPDF`/`Chunkr` (AGPL), `Nanonets` (Qwen non-commercial), `AWS Textract` (no TR/AR) are **disqualified or conditional** for an embeddable service. Docling, MarkItDown, Tesseract, PaddleOCR, dots.ocr, DeepSeek-OCR-2, olmOCR are the Apache/MIT-clean set. [27][28][31]

---

## 2. Search log

**Queried:** OmniDocBench / ParseBench / PulseBench-Tab / PureDocBench leaderboards; OCRTurk (ACL 2026) and KITAB-Bench / GlotOCR (Arabic); each tool's GitHub releases + LICENSE + HuggingFace card; vendor pricing pages; LangChain integration registries; olmOCR / Docling / MinerU arXiv papers; "VLM PDF Markdown RAG best practices 2026" surveys.

**Funnel:** ~165 screened (tool repos, benchmark papers, vendor blogs, comparison posts) → 58 shortlisted (passed a relevance + primary-source filter) → 31 deep-read (full rubric or methodology extraction).

**Recency rule applied:** sources ≥ 2025-09 preferred. A large block — the entire frontier-VLM tier, all 7 methodology entries, all 9 benchmark entries, all best-practice notes, and ~half the rubric sources — carries `access_date = undefined`. Those are flagged **liveness-unconfirmed** throughout and never solely support a recommendation.

---

## 3. Pillar 1 — Library comparison (14-dimension scorecard)

Scores condensed from the rubric data. **Grade** = overall evidence grade for that row. "TR/AR" = verified Turkish/Arabic accuracy. Throughput is best-reported (heterogeneous GPUs — see section 11 caveat). Cells marked *unver.* = claimed but not independently verified.

| Tool | Category | License (embeddable?) | Local/Hosted | MD-native | Reading-order | Table fidelity | Formula | Figure handling | TR/AR | Throughput (best) | Output | LangChain fit | Maintenance | Grade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Docling** | vlm-native ensemble | **MIT / Apache** yes | local (CPU ok) | yes native | tree, suppresses furniture | 97.9% clean; hallucinates dense | EqFormer opt.; std "inconsistent" | 3 modes (embed/ref/placeholder) | low — Granite-Docling AR experimental, TR unlisted | 7.9 pp/s RTX5090 | DoclingDocument + bbox | **first-party langchain-docling** | IBM+LF, v2.104 | high core / mod tables / **low TR-AR** |
| **MinerU** | hybrid VLM+OCR | **AGPL/YOLO conflict (#2863)** | local | yes (tables=HTML) | **best** RO 0.044 | TEDS 88.2 | **CDM 88.5 best** | crop PNG + captions | high-claim AR/TR via PP-OCRv6 | 2.1-4.5 pp/s | model/middle/content JSON | langchain-mineru | very active, 68k stars | high / **license unresolved** |
| **Marker** | pipeline (wraps Surya) | **GPL-3.0 + RAIL over 2M USD** | local (GPU) | yes native | RO ED 0.243 | TEDS 54 / 89% conflicting | **CDM 18.4 weak** | PNG refs; batch bug #617 | AR 72.7% unver. | 5.6 pp/s; 120 H100 unver. | block-tree JSON | **none** | active, 36k stars | high / mod / neg formula |
| **Surya** | single 650M VLM | **RAIL over 5M USD** | local | no JSON-first | dedicated RO model | HTML modes unver. | KaTeX | bbox only, no crop | **AR 72.7% vendor vs CER 4.95 worst (KITAB)** | 5.35 pp/s | flat block JSON | none | active, 21k stars | high / **AR contested** |
| **olmOCR** | 7B VLM | **Apache** yes | local (>=12GB) | yes (tables=HTML) | multi-col 83.7 | 82.3 olmOCR-bench | Math 83.0 | **text ref only, no pixels** | low — EN bench; AR community | 0.4-4 pp/s; 176 USD/M | MD+YAML, no bbox | **none** | AI2, ~3.5mo stale | high EN / **low multiling** |
| **Nougat** | doc-VLM (Donut) | **CC-BY-NC** | local (GPU) | mmd | implicit only | tabular LaTeX, weak | strength but hallucinates | **none (text only)** | **EN-only; non-Latin to repetition** | ~0.3 pp/s | flat .mmd | none | **stale (Aug 2023)** | high / **stale + DQ** |
| **GOT-OCR2.0** | 580M unified VLM | Apache (badge dispute) | local (~7GB) | mathpix-md | RO ED 0.141 | TEDS 72; **rotated 8.5%** | F1 0.749 | none (charts only) | **EN/ZH only; TR/AR untested** | unverified | string/mathpix | none | **dormant since Feb 2025** | moderate / stale-risk |
| **dots.ocr to dots.mocr** | 1.7B single VLM | **MIT** yes | local (>8GB) | yes (tables=HTML) | RO 0.040 strong | TEDS 88.6 | edit 0.329 | **pictures not parsed** | **AR 63.2% Acc@5 (GlotOCR); TR Latin-proxy** | 0.28-1.94 pp/s | JSON + MD | none (clean JSON) | active; rebrand ambig. | high quality / mod throughput |
| **PaddleOCR + PP-StructureV3** | classical pipeline | **Apache** yes | local (CPU/GPU) | yes (v3.x) | XY-Cut++ | OmniDoc EN ED 0.145 | yes | layout class | **109 langs incl. TR/AR (claim)** | 0.6-8.9 s/pg A100 | structured MD | community | 73k stars, CVPR 2026 | high |
| **PaddleOCR-VL-1.6** | 0.9B VLM | **Apache** yes | local | yes | — | **OmniDoc 96.33 #1** | 9.65 formula bench | — | multilingual claim | — | MD/JSON | via Paddle | active | high (bench) |
| **DeepSeek-OCR-2** | 3B VLM | **Apache** yes | local (~8GB) | yes | — | 91.09 OmniDoc v1.5 | — | — | unstated | — | MD/JSON/HTML | manual | active | high |
| **Nanonets-OCR2-3B** | 3B VLM | **Qwen non-commercial** | local (~8GB) | yes semantic tags | — | — | LaTeX | signatures/Mermaid unver. | unstated | — | tagged MD | manual | active | moderate / **DQ commercial** |
| **MarkItDown** | classical wrapper | **MIT** yes | local CPU | yes low fidelity | 0.844 simple / breaks multi-col | **0.273 poor** | none (Unicode) | **silently dropped** | low — passthrough | **~8.3 pp/s** | flat string | 3rd-party pkg | MS, 157k stars | high (incl. neg) |
| **PyMuPDF / pymupdf4llm** | text-layer | **AGPL-3.0** | local CPU | yes | multi-col scramble | MD tables | none | base64/ref | passthrough | **sub-second/pg** | MD/JSON | LangChain integ. | active, Artifex | high / **AGPL DQ** |
| **pdfplumber** | text-layer | **MIT** yes | local CPU | no objects | born-digital only | heuristic tables | none | none | passthrough | fast | dicts/DataFrame | none | single maintainer | high |
| **Tesseract** | classical OCR | **Apache** yes | local CPU | no | none | **no table understanding** | none | none | 100+ langs; **RTL inconsistent** | ~0.8-1 s/pg | hOCR/ALTO | none | active, 75k stars | high |
| **Camelot 2.0** | table extractor | **MIT** yes | local | MD tables (v2) | n/a | **73% (lattice)** | n/a | n/a | n/a | — | DataFrame/MD | none | revitalized Jun 2026 | high |
| **Tabula** | table extractor | MIT | local (JVM) | no | n/a | 67.9% | n/a | n/a | n/a | — | DataFrame/CSV | none | **stale (2021)** | moderate / stale |
| **Unstructured.io** | multi-strategy ETL | **Apache** yes | both | no typed elements | metadata coords | T-LAG **0.360 worst** | n/a | element class | varies | hi_res **51 s/pg** | typed Element JSON | LangChain | active, VC | high |
| **Mistral OCR 3** | hosted VLM-OCR | proprietary (API) | **hosted** | yes (tables=HTML) | RO 91.6% / ED 0.144 | TEDS ~70; 96.6% self | CDM 78.2 | bbox + images array | **TR listed; AR ~99% community; not in OCRTurk** | 2000 pg/min self | JSON+MD+bbox | manual | active, Dec 2025 | high / **TR-AR mod** |
| **LlamaParse** | hosted agentic | proprietary | hosted | yes | faith 89.7 | **GTRM 90.7 #1 (own bench)** | VLM, unbenchmarked | base64 + granular bbox | AR 91% single report | ~6 s/doc | typed JSON + bbox | LlamaIndex-native | **deprecated May 2026** | mod (COI) / **migration** |
| **Reducto** | hosted agentic | proprietary | hosted+VPC | yes | best bbox+RO | T-LAG 0.795 | yes | bbox provenance | **explicit TR + AR** | config-dep | MD/HTML/JSON/CSV | manual | active, YC | high |
| **Azure Doc Intelligence** | hosted | proprietary | hosted | yes outputContentFormat=markdown | reading-order | T-LAG 0.761; **chart 1.6%** | add-on | JSON + polygons | **TR + AR supported** | — | MD+JSON+bbox | manual | active | high |
| **Google Document AI** | hosted | proprietary | hosted | no JSON | reading-order | — | — | bbox | **TR + AR supported** | — | rich JSON | manual | active | high |
| **AWS Textract** | hosted | proprietary | hosted | no (3rd-party) | Layout feature | T-LAG 0.603 | n/a | bbox | **NO TR / NO AR — hard DQ** | — | JSON + bbox | manual | active | high (incl. DQ) |
| **Adobe PDF Extract** | hosted | proprietary, ent-only | hosted | yes MD endpoint | strong RO | cell detection | n/a | base64 | **TR/AR unverified** | — | JSON + bbox | manual | active | moderate |
| **Upstage Doc Parse** | hosted VLM | proprietary | hosted | yes HTML+MD | reading-order | TSR to HTML | equations | bbox | **TR/AR unconfirmed** | — | HTML/MD + bbox | langchain-upstage | active | moderate |
| **Zerox** | frontier-VLM wrapper | **MIT** (wrapper) | hosted (delegates) | yes per page | VLM holistic | MD pipes | none | inline only | inherits chosen VLM | ~0.1 pp/s (1 sample) | page JSON | manual | active | moderate (no indep bench) |
| **Gemini 2.5/3.x (direct)** | frontier-VLM | proprietary | hosted | yes when prompted | strong | **9.5/10 BMVC** | 9.75 formula | holistic | strong (ID drift, section 11) | API-bound | text | **shared.llm adapter** | active | high (2.5) / **3.x unconfirmed** |
| **GPT-4o/5.x (direct)** | frontier-VLM | proprietary | hosted | yes when prompted | redundancy issues | mod | 86.8 (4o) | holistic | multilingual | 128K (4o) | text | adapter | active | 4o high / **5.x IDs unconfirmed** |
| **Claude Sonnet/Opus (direct)** | frontier-VLM | proprietary | hosted | yes | strong tables | 7.0 BMVC (Sonnet) | mod | 2576px (Opus) | multilingual | API | text | adapter | active | **post-cutoff IDs unconfirmed (section 11)** |

### Liveness ledger (stale / at-risk)

| Tool | Signal | Verdict |
|---|---|---|
| **Nougat** | last release Aug 2023; non-Latin to instant repetition | **Stale — avoid.** Superseded by olmOCR/MinerU/Docling. [4] |
| **Tabula / tabula-java** | tabula-java no release since Aug 2021; maintainer bandwidth note | **At-risk.** Use Camelot 2.0 instead. |
| **GOT-OCR2.0** | original repo dormant since ~Feb 2025; HF-transformers maintenance claim **not separately evidenced** | **At-risk;** HF-maintenance unverified. [7] |
| **olmOCR** | last release 2026-03-12, ~3.5 months stale; graded active on a single date | **Watch.** Active org (AI2) but cadence thin. |
| **dots.ocr to dots.mocr** | rebrand 2026-03-19; canonical artifact ambiguous; Hebrew issue #225 unanswered | **Active but ambiguous.** |
| **LlamaParse** | llama-parse package deprecation ends 2026-05-01; migrate to llama-cloud>=1.0 | **Forced migration.** |
| **Claude Fable 5** | API suspended 2026-06-12 by US export-control directive — **extraordinary claim, only a marketing post + one blog, every source access_date=undefined** | **Do not plan around it (section 11).** |

### Three-plus tools found beyond the seed list

**PaddleOCR-VL-1.6** (Baidu, 0.9B, Apache, OmniDocBench #1 at 96.33), **DeepSeek-OCR-2** (3B, Apache, 91.09 OmniDoc v1.5), **Nanonets-OCR2-3B** (semantic tagging, Qwen non-commercial), plus **Granite-Docling-258M** (standalone IBM VLM), **Camelot 2.0** (neural table backend, Jun 2026), and **Upstage Document Parse**. [13][14][15][16]

---

## 4. Pillar 2 — Methodology taxonomy

| # | Method | What it does | Shines when | Fails when | Representative tool | RAG symptom |
|---|---|---|---|---|---|---|
| 1 | **Text-layer (born-digital)** | reads embedded glyph stream, no render/OCR | clean digital PDFs; speed; no GPU | any scanned page (silent blank); multi-col; equations | pymupdf4llm, pdfplumber | multi-col scramble mixes adjacent-column paragraphs into one chunk |
| 2 | **Classical OCR + layout** | render then layout detector (DocLayNet/YOLO) then Tesseract/PP-OCR then reassemble | Latin scanned, CPU/offline, high throughput | noisy scans; math; degraded RTL; borderless tables | Tesseract, PaddleOCR+PP-StructureV3 | hallucinated table cells poison numeric retrieval |
| 3 | **Specialised doc-VLM (OCR-free)** | page-image to markup end-to-end | math (Nougat); 90+ langs (Surya); compact (Granite-Docling) | out-of-domain repetition loops; degraded RTL (Surya AR 72.7%) | Nougat, Surya, GOT-OCR, Granite-Docling | repetition loops emit multi-KB garbage chunks that dominate kNN |
| 4 | **Frontier-VLM direct (page-as-image)** | render then frontier multimodal + prompt then MD/JSON | extreme visual complexity; arbitrary scripts; zero infra | high volume cost; privacy; **12.4% hallucination on dense text** | Gemini, GPT, Claude, Zerox | hallucinated cells inject false facts into the index — undetectable post-hoc |
| 5 | **Hybrid (layout to region routing to VLM on hard regions only)** | segment then classify difficulty then cheap path for text, VLM for tables/figures then merge by coords | mixed docs; latency/cost balance; >10 pp/s commodity HW | layout mis-segment cascades; borderless tables; floating elements | MinerU hybrid, Docling VLM pipeline, Unstructured | error-cascaded regions join table rows with unrelated prose |
| 6 | **Agentic (render to extract to self-check to re-render failures)** | fan-out workers + confidence + supervisor re-dispatch | heterogeneous batches; accuracy SLA; measurable per-page confidence | runaway retry cost; semaphore deadlock; cross-page tables lost | Reducto Agentic, LlamaParse | cross-page table split into two orphan half-table chunks |
| 7 | **Two-pass / verification (extract to 2nd model validates vs image)** | draft then verifier checks structure/numbers vs page image then targeted re-extract | high-stakes (finance/legal/medical); selective second pass | verifier also hallucinates; doubles cost; cross-page blind | Marker --use_llm, custom LLM-as-judge graph | "verified" wrong cells look more trustworthy than single-pass |

### LangGraph shapes (provided for hybrid, agentic, two-pass; missing for methods 1-4 — see section 11)

- **Hybrid (5):** `StateGraph` then `segment_page` node fans out via **Send API** to N parallel `extract_region` workers (one per detected region) then list-reducer then `merge_results` then conditional edge back to `reprocess_region` for low-confidence regions then `assemble_document`.
- **Agentic (6):** `ingest` then Send fan-out to `extract_page[i]` workers bounded by `asyncio.Semaphore(N)` then per-worker conditional self-loop (confidence < tau then `retry_page[i]` with alternate model/resolution, else then `results` reducer) then `assemble` then `quality_gate`.
- **Two-pass (7):** `extract` then `verify` (draft + page image then per-region confidence dict) then conditional Send to `re_extract_region[k]` then `merge_corrections` then `finalize`, with optional second `verify` guarded by `max_rounds`.

---

## 5. Pillar 3 — Best-practices playbook

**Rendering.** Render at **150-300 DPI** (150 for text, 300 for small fonts/dense tables/formulas). Do **not** let the provider rasterize — GPT-4o internally downsamples to ~90 DPI and produces "full of errors"; the same page at 210 DPI is clean. Gemini caps at 3072px, Claude rejects >2000px/side in >20-image requests. Tile oversized pages instead of downscaling. [bestpractices: rendering]

**Prompting.** Force JSON-schema output that emits metadata (`language`, `rotation`, `has_table`, `has_formula`) **before** linearized text; instruct "Do not hallucinate. If unreadable write [UNREADABLE]." Temp 0-0.1 normally, 0.7-0.8 only on retry to break repetition loops. Inject **document anchoring** (PDF text coords, <=6000 chars/page) for born-digital pages — olmOCR's key hallucination-reducer. Tables to **HTML** (colspan/rowspan); formulas to **LaTeX**. Explicit reading-order instruction for multi-column. [bestpractices: prompting][37]

**Cost/throughput.** Route by complexity: text-only to Gemini Flash / direct text (nearly free); tables/figures/scanned to premium VLM. Fan-out with `asyncio.BoundedSemaphore(8-16)`; **retry outside the semaphore**. Batch APIs for >10k pages (50% off). Prompt-cache static prefixes (Anthropic 90% off cache reads). [bestpractices: cost]

**Reliability.** Three-level retry: JSON-fail to regenerate; repeated to raise temp; N-fail to fall back to PyMuPDF plaintext. olmOCR's **~12% retry rate** is the well-tuned baseline — design for it. Validate with LLM-as-judge (r=0.93 vs human for tables) — but see the COI caveat in section 11. Keep a **golden-set regression suite** from your corpus; text accuracy alone masks structural failure (GPT-4o-mini: 75% text / 13% tree). [bestpractices: reliability]

### Anti-patterns (smell to fix)

| Smell | Fix |
|---|---|
| Send raw PDF to GPT-4o, let it render | Pre-render 150-300 DPI PNG yourself; provider downsamples to ~90 DPI |
| Judge quality by edit-similarity alone | Add TEDS / tree-similarity (75% text can hide 13% structure) |
| Markdown tables for merged-cell tables | Emit HTML <table> with colspan/rowspan |
| Retry inside the semaphore block | Release first; holding the slot starves other tasks |
| One fixed temperature for all passes | 0.1 baseline, 0.7-0.8 on retry to escape repetition loops |
| Same VLM strategy on every page | Route by detected complexity (word/image-count heuristic) |
| Force-decode schema on a fine-tuned VLM | olmOCR-class models self-adhere; force-decode is "unreliable" for them |
| Whole-document single request | Per-page (or 2-5 page) calls + anchoring beat diluted long-context attention |
| Treat extracted MD as RAG-ready | Validate structure with LLM-as-judge before indexing |
| No header/footer/footnote instruction | "Remove running headers/footers/page numbers; preserve footnotes as [FOOTNOTE: ...]" |
| GPU VLM in serverless/ephemeral container | Use hosted API for serverless; reserve local VLM for persistent batch infra |
| Benchmark on a generic set, extrapolate | Domain variance >55 pts — build a golden set from your corpus |

---

## 6. Pillar 4 — Image to Markdown deep dive

### 6.1 Per-tool answers to the 7 figure-handling questions

| Question | Docling | MinerU | Marker | olmOCR | dots.ocr | Mistral OCR 3 | LlamaParse |
|---|---|---|---|---|---|---|---|
| **1. Figures extracted as files?** | yes PNG (embed/ref/placeholder) | yes crop PNG to images/ | yes PNG images/ (batch-drop bug #617) | **no — text ref only, no pixels** | **no — "pictures not parsed"** | yes images array per page | yes separate image files |
| **2. Caption associated?** | enrichment (VLM desc) | spatial proximity + chart_caption | surrounding text | basic alt-text (v0.4) | — | inline placeholder | yes |
| **3. Silent drop vs explicit?** | explicit | explicit, "no figures discarded" | explicit (except batch bug) | **silent (pixels never saved)** | **explicit empty Picture box** | explicit | explicit |
| **4. Base64 vs file-ref?** | both configurable | ref (base64 opt.) | **ref only** | n/a | n/a | base64 in images array | **base64** ImageNode |
| **5. Bbox emitted?** | yes ProvenanceItem | yes [x0,y0,x1,y1] | yes JSON mode | **no bbox tree** | yes JSON bbox | yes per-image bbox | yes **granular per-word/line/cell** |
| **6. Chart to data?** | opt. enrichment (chart to table) | chart_* fields | no | no | SVG attempt "not robust" | structured annotation | agent (78% ParseBench) |
| **7. Figure ref carryable as chunk metadata?** | yes ref path + caption | yes bbox + caption fields | yes rel path | weak (no pixels) | bbox only | yes bbox + image id | yes file_id + bbox |

**Chart-to-data is the unsolved gap for OSS:** the only numbers in the corpus are cloud-VLM chart scores (**64-78%** ParseBench) vs **<6%** for traditional parsers; **no per-OSS-tool chart score exists** (section 11). [benchmarks: ParseBench]

### 6.2 Side-by-side: actual emitted Markdown for the same figure+table page

**Docling** (file-ref figure, HTML table inside Markdown):

```markdown
## Quarterly Results

![Figure 1](images/page_3_fig_1.png)

<table><tr><th>Quarter</th><th colspan="2">Revenue</th></tr>
<tr><td>Q1</td><td>EU</td><td>US</td></tr></table>
```

**MinerU** (crop ref + collapsible AI description, HTML table):

```markdown
![](images/3_0.png)
<details><summary>Image description</summary>Bar chart of revenue by region.</details>

<table><tr><td rowspan="2">Region</td>...</tr></table>
```

**olmOCR** (figure is a text reference only — no pixels written):

```markdown
---
language: en
---
![Figure 1](page_3_120_340_512_300.png)
<!-- filename encodes bbox; file not saved -->

<table>...</table>
```

**Mistral OCR 3** (inline placeholder + separate images[] array with bbox in JSON):

```markdown
![img-0.jpeg](img-0.jpeg)

| Quarter | Revenue |
<!-- or HTML when table_format=html -->
```

**Takeaway for Guillotine:** Docling and MinerU give you a real file path + caption + bbox you can lift into chunk metadata. olmOCR and dots.ocr leave you with a dangling reference or an empty picture box — bad for a figure-aware chunker.

### 6.3 Runnable caption-and-index recipe (wired through shared.llm.get_llm("google", ...))

This is the **only** way a VLM enters the pipeline per repo policy — through the adapter, never a hard-coded SDK call. First add Gemini exactly per `shared/llm/README.md`:

```bash
uv add langchain-google-genai          # uv ONLY — never pip
```

```python
# shared/llm/google_adapter.py  (new file — one provider per backend, Mitrailleuse style)
from langchain_google_genai import ChatGoogleGenerativeAI
from shared.settings import settings

DEFAULT_MODEL = "gemini-2.0-flash"

def build(model=None, **kwargs):
    return ChatGoogleGenerativeAI(
        model=model or DEFAULT_MODEL,
        google_api_key=settings.google_api_key,   # add google_api_key: str | None = None to settings.py
        **kwargs,
    )
```

```python
# shared/llm/base.py
from shared.llm import anthropic_adapter, openai_adapter, google_adapter
_ADAPTERS = {
    "anthropic": anthropic_adapter,
    "openai":    openai_adapter,
    "google":    google_adapter,        # registered; auto-joins the fallback chain
}
```

Now caption every extracted figure and emit a Guillotine-ready Markdown ref whose caption rides as chunk metadata:

```python
import base64, pathlib
from langchain_core.messages import HumanMessage
from shared.llm import get_llm

# VLM only for the captioning sub-task; with_fallback keeps it resilient.
vlm = get_llm("google", model="gemini-2.0-flash", temperature=0)

def caption_figure(png_path: str) -> str:
    img_b64 = base64.b64encode(pathlib.Path(png_path).read_bytes()).decode()
    msg = HumanMessage(content=[
        {"type": "text",
         "text": "Caption this figure in one sentence for a document index. "
                 "If it is a chart, also list the series and axis labels. Do not hallucinate."},
        {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
    ])
    return vlm.invoke([msg]).content.strip()

def figure_markdown(png_path: str, page: int, bbox: tuple) -> tuple[str, dict]:
    cap = caption_figure(png_path)
    md  = f"![{cap}]({png_path})"
    meta = {"figure_path": png_path, "figure_caption": cap,
            "page": page, "bbox": list(bbox)}        # carried onto the chunk
    return md, meta
```

Because the captioner is `get_llm("google", ...)`, it inherits the `Runnable.with_fallbacks` chain — if the Google key/quota fails, it falls through to the next configured provider with zero call-site change.

---

## 7. Decision tree

```
START
- Born-digital (real text layer)?
  - yes, simple 1-col, no tables ........... TEXT-LAYER (pdfplumber, MIT) -- VLM is OVERKILL here
  - yes, tables / multi-col / formulas ..... Docling standard pipeline (MIT)
- Scanned / image-only?
  - Latin, CPU-only, budget=0 .............. PaddleOCR+PP-StructureV3 (Apache) or Tesseract
  - degraded / mixed ....................... Docling (OCR backend) then VLM fallback on hard pages
- Table-heavy (merged cells, financial)?
  - born-digital, lattice .................. Camelot 2.0 (MIT) then HTML
  - scanned / complex ...................... hosted VLM-OCR (Mistral OCR 3 / Reducto), tables=HTML
- Figure/chart-heavy (need chart to data)?
  - ........................................ hosted VLM (Gemini via shared.llm) -- OSS chart-to-data is <6%
- Multilingual TR / AR?
  - privacy-OK, hosted-OK .................. Mistral OCR 3 / Azure / Google DocAI  (NOT AWS Textract)
  - must be local .......................... PaddleOCR/MinerU (claim TR/AR) + golden-set verify
                                             WARNING: NO verified per-tool OCRTurk/Arabic score (section 11)
- Budget-constrained, high volume?
  - ........................................ route-by-difficulty hybrid: text-layer (free)
                                             + Gemini Flash (~0.0004 USD/pg) on hard pages only
- Privacy / air-gapped?
  - ........................................ local only: Docling / olmOCR / DeepSeek-OCR-2 / dots.ocr
                                             (all Apache/MIT) -- no hosted API
```

---

## 8. Recommended stack for THIS repo

### Primary + fallback

| Layer | Choice | Why (against every constraint) |
|---|---|---|
| **Extractor (primary)** | **Docling** standard pipeline, langchain-docling loader | MIT/Apache = embeddable in a teaching/commercial service (unlike Marker/Surya/MinerU/PyMuPDF/Chunkr/Nanonets). Native Markdown + DoclingDocument tree to MarkdownHeaderTextSplitter keeps headings; tables export as HTML (Guillotine-safe). First-party LangChain loader drops straight into a LangGraph node. [1][14][27] |
| **Hard-region / scanned / TR-AR fallback** | **Mistral OCR 3** (default hosted) or `get_llm("google", "gemini-2.0-flash")` (in-repo, adapter-native) | A VLM is reserved for scanned/TR-AR/dense-table/figure pages only — overkill on born-digital. Mistral lists Turkish + Arabic ~99% (community); Gemini is the path that flows through the existing shared/llm adapter + fallback chain with no SDK call. [10][29] |
| **Captioning** | `get_llm("google", ...)` per section 6.3 | Adapter pattern honoured; figure captions become chunk metadata for Guillotine. |

### Adapter-pattern compliance

The VLM enters **only** via `get_llm("google", ...)`, added by the exact `shared/llm/README.md` recipe (`uv add langchain-google-genai` then `google_adapter.py` with `build(model=None, **kwargs)` returning `ChatGoogleGenerativeAI(model=model or "gemini-2.0-flash", google_api_key=settings.google_api_key, **kwargs)` then `google_api_key` in `settings.py` then `"google": google_adapter` in `_ADAPTERS`). One provider file, tiny factory — Mitrailleuse/Aletheia style. No hard-coded genai/SDK calls anywhere. uv only.

### Guillotine chunker-friendliness + TR/AR

Output is clean Markdown with preserved headings (MarkdownHeaderTextSplitter), tables as HTML/Markdown (kept intact, not shredded mid-table), and figure refs (path + caption + bbox) carryable as chunk metadata. **TR/AR honesty:** no tool has a verified per-tool OCRTurk/Arabic score (section 11) — Guillotine should run a small golden set of Turkish (the diacritics c-cedilla, g-breve, dotless-i, dotted-I, o-umlaut, s-cedilla, u-umlaut, plus the dotless-i / dotted-I casing trap) and Arabic RTL pages through Docling-local vs Mistral/Azure-hosted and pick per-corpus.

### rag_qa_api_pro production path

Today `graph.py` ingests **`*.md` only** via `DirectoryLoader+TextLoader` (lines 69-80), chunked by `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)`. To add PDFs:

1. **Extraction is an OFFLINE precompute step, NOT an in-lifespan node.** Run Docling (+VLM fallback) to write `.md` into `SOURCE_DIR` before deploy. This keeps the slow/expensive VLM pass **out of `ensure_index`** — a multi-minute VLM extraction under the **Redis SETNX 120s TTL lock** (`acquire(..., ttl_seconds=120)` in `locks.py`) would risk lock expiry then double-extraction. Never extract inside that lock.
2. **Chunking must change for VLM-Markdown.** Replace the flat 500/80 splitter with `MarkdownHeaderTextSplitter` then `RecursiveCharacterTextSplitter` so HTML tables and LaTeX are not split mid-structure; keep table/figure blocks atomic and attach figure metadata to the chunk.
3. **Batchable, fault-tolerant, observable:** precompute with the section-5 three-level retry + golden-set regression; the existing `AsyncPostgresSaver` checkpointer and Send-API fan-out (semaphore-bounded) handle concurrency — keep N small to limit checkpoint write amplification across replicas.

### When a VLM is overkill (stated explicitly)

Born-digital, single-column, table-free PDFs: use **text-layer** (pdfplumber/Docling-parse). A frontier VLM here adds cost, latency, and a 12.4% hallucination risk for zero quality gain. Bias to VLM **only** for scanned, RTL, dense-table, or figure/chart pages.

---

## 9. Evidence-grading table (key claims)

| # | Claim | Grade | Basis |
|---|---|---|---|
| 1 | Docling is MIT/Apache, native Markdown, first-party LangChain loader | **high** | LICENSE + langchain-docling PyPI + IBM announcement [1][14] |
| 2 | Docling is NOT a clear quality leader (ParseBench 50.6%; absent from OmniDoc published) | **high** | ParseBench arXiv + leaderboard [27] |
| 3 | MinerU leads OmniDocBench reading-order/formula but license (AGPL/YOLO #2863) is unresolved | **high** (perf) / **moderate** (license) | arXiv 2509.22186 + GitHub #2863 [3] |
| 4 | Marker/Surya RAIL, PyMuPDF/Chunkr AGPL, Nanonets non-commercial then conditional/DQ for embedding | **high** | LICENSE files [27][28][31] |
| 5 | AWS Textract has NO Turkish/Arabic — hard DQ for TR/AR | **high** | AWS language-support docs |
| 6 | Markdown cannot express merged cells; emit HTML for complex tables | **high** | arXiv 2603.18652 + olmOCR [37] |
| 7 | Hybrid route-by-difficulty is the dominant cost-correct production pattern | **moderate** | Firecrawl/Unstructured/Applied-AI corpus (vendor-leaning) |
| 8 | Frontier VLM-direct hallucinates ~12.4% on dense text | **moderate** | Reducto internal measurement (single vendor) |
| 9 | PaddleOCR-VL-1.6 #1 on OmniDocBench (96.33) | **moderate** | OmniDoc leaderboard + CodeSOTA (vendor self-report risk) |
| 10 | Mistral OCR 3 ~1-2 USD/1k pages, TR listed, AR ~99% | **moderate** | vendor + community (not in OCRTurk) [10][29] |
| 11 | No tool has a verified Turkish OCRTurk score | **high** (as a gap) | OCRTurk cited everywhere, zero per-tool score extracted |
| 12 | Surya Arabic = 72.7% (vendor) contradicts KITAB-Bench CER 4.95 (worst) | **moderate** (conflict) | Surya blog vs KITAB-Bench arXiv 2502.14949 |
| 13 | olmOCR does not save figure pixels; multilingual unvalidated | **high** (figure) / **low** (multiling) | olmOCR arXiv + GitHub README |
| 14 | "Claude Fable 5 API suspended by export controls" | **low** | marketing post + one blog, all access_date=undefined |
| 15 | Post-cutoff model IDs/prices (GPT-5.x, Gemini 3.x, Claude Opus 4.8/Fable 5) | **low** | every supporting source access_date=undefined |

---

## 10. Adversarial verdicts honoured

- **Do not claim "Docling is best."** It is the safest embeddable choice, not the quality leader; it is mid-pack/absent on the freshest benchmarks and its dense-table/formula paths are independently questioned. [2][27]
- **Do not claim VLM-direct "solves" extraction.** It carries a measured ~12.4% dense-text hallucination rate and is overkill on born-digital pages; PureDocBench shows "document parsing is NOT solved" (best ~74-78%, formula <67%). [benchmarks]
- **Do not trust vendor-run benchmarks uncritically.** ParseBench (LlamaIndex sells LlamaParse), PulseBench-Tab (Pulse AI sells Pulse Ultra), olmOCR-bench (AI2) all have winning-tool COI and unvalidated novel metrics (T-LAG, LLM-judge r=0.93). OmniDocBench is the only multi-source-corroborated leaderboard — and Docling/LlamaParse are largely absent from it. [27]
- **TR/AR caveats stand.** No verified Turkish score anywhere; Arabic numbers are fragmentary and contradictory; RTL reading-order **correctness** (not glyph recognition) is asserted but never independently verified.

---

## 11. Known open questions (completeness-critic gaps)

1. **Model-ID / price drift (high).** Are GPT-5.4/5.5, Gemini 3.1 Pro / 3.5 Flash, Claude Opus 4.8 / Sonnet 4.6 / Fable 5 real, and is "Fable 5 suspended 2026-06-12 by export-control directive" corroborated by any primary government source? Every supporting source is access_date=undefined. Treat all post-cutoff IDs as **unverified**.
2. **Access dates missing wholesale (high).** The entire frontier-VLM block, all 7 methodologies, all 9 benchmarks, all best-practices, and ~half the rubric sources lack access dates — liveness for those rows is unsupported.
3. **Turkish (Guillotine) (high).** No verified per-tool OCRTurk score for any tool. The Turkish diacritics and the dotless-i / dotted-I casing trap are untested everywhere. Which tool is actually best on Turkish is **currently undefendable** — must be measured on a local golden set.
4. **Arabic (Guillotine) (high).** Surya 72.7% (vendor) vs KITAB CER 4.95 (worst) unresolved; dots.ocr 63.2% Acc@5; Mistral ~99% community-only; olmOCR Arabic unverified; per-tool KITAB scores for Docling/Marker/MinerU not extracted; RTL reading-order correctness never independently verified. AWS Textract = hard DQ.
5. **rag_qa_api_pro integration (high — partially answered in section 8).** Recommended: offline precompute **outside** the Redis SETNX 120s lock; MarkdownHeaderTextSplitter instead of flat 500/80; figure metadata onto chunks. Still open: exact latency/cost budget for re-extraction, and Send-API semaphore N vs AsyncPostgresSaver write amplification on the multi-replica deployment.
6. **LangGraph shapes for methods 1-4 (moderate).** Only hybrid/agentic/two-pass have shapes; text-layer / classical-OCR / doc-VLM / frontier-direct do not. No concrete semaphore-N tied to the project checkpointer.
7. **Rubric coverage holes (moderate).** Only 6 tools fully scored; the hosted enterprise tier and several new VLMs (PaddleOCR-VL standalone, DeepSeek-OCR-2, Nanonets-OCR2, Granite-Docling standalone) lack full per-dimension scoring; Zerox record truncated.
8. **Figure/chart deep-dive (moderate).** No per-OSS-tool chart-to-data score (only cloud 64-78% vs <6% traditional); figure-caption-association accuracy and handwriting/signature detection (Nanonets claim) unverified.
9. **Benchmark COI + reproduction (moderate).** Three freshest decision-relevant benchmarks are vendor-run with unvalidated metrics; only OmniDocBench is multi-source-corroborated.
10. **License blockers for the production path (moderate).** MinerU AGPL/YOLO conflict unresolved; Marker/Surya/PyMuPDF/Chunkr/Nanonets copyleft/non-commercial — sections 3/8 mark the Apache/MIT-clean embeddable set (Docling, MarkItDown, Tesseract, PaddleOCR, dots.ocr, DeepSeek-OCR-2, olmOCR).
11. **Cost/throughput on commodity HW (moderate).** Throughput numbers span RTX 5090/A100/H100/H200/L4/L40S/M3 Max with no normalized pages/sec/dollar; no figure for a single consumer GPU on Windows 11 (the user's likely dev box); no local-vs-hosted break-even tied to the actual re-index cadence.
12. **Liveness depth (low).** Some "active" grades rest on one release date (olmOCR ~3.5mo stale; GOT-OCR HF-maintenance claim unevidenced; dots.ocr to dots.mocr canonical-artifact ambiguity).

---

## 12. Numbered citations

*Access date for all primary tool repos, vendor pricing, and HuggingFace cards: **2026-06-22**. Sources marked **(undefined)** had no access date in the source data — treat as liveness-unconfirmed (section 11).*

1. Docling — GitHub repo & releases (v2.104.0). https://github.com/docling-project/docling
2. Docling Technical Report — arXiv 2501.17887 / 2408.09869. https://arxiv.org/html/2501.17887v1
3. MinerU2.5 — arXiv 2509.22186 + License discussion #2863. https://github.com/opendatalab/MinerU/discussions/2863
4. Nougat — arXiv 2308.13418 + Arabic-Nougat 2411.17835 (base BLEU 0.0037 AR). https://github.com/facebookresearch/nougat
5. Surya OCR 2 — GitHub + Datalab blog. https://github.com/datalab-to/surya
6. olmOCR / olmOCR-2 — arXiv 2502.18443, 2510.19817. https://github.com/allenai/olmocr
7. GOT-OCR2.0 — arXiv 2409.01704 + HF transformers doc. https://github.com/Ucas-HaoranWei/GOT-OCR2.0
8. Zerox — GitHub + docs. https://github.com/getomni-ai/zerox
9. MarkItDown — GitHub + OpenDataLoader benchmark (PDF 0.589). https://github.com/microsoft/markitdown
10. Mistral OCR 3 — mistral.ai/news/mistral-ocr-3 + CodeSOTA verified review. https://mistral.ai/news/mistral-ocr-3/
11. dots.ocr / dots.mocr — GitHub + blog.md + GlotOCR 2604.12978. https://github.com/rednote-hilab/dots.ocr
12. Granite-Docling-258M — IBM announcement + SmolDocling 2503.11576. https://huggingface.co/ibm-granite/granite-docling-258M
13. PaddleOCR-VL-1.6 — arXiv 2606.03264 + GitHub. https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6
14. langchain-docling — PyPI + GitHub. https://pypi.org/project/langchain-docling/
15. DeepSeek-OCR-2 — HF + GitHub. https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
16. Nanonets-OCR2-3B — HF (Qwen non-commercial license). https://huggingface.co/nanonets/Nanonets-OCR2-3B
17. PaddleOCR + PP-StructureV3 — GitHub + PaddleOCR 3.0 report 2507.05595. https://github.com/PaddlePaddle/PaddleOCR
18. Tesseract OCR — GitHub releases v5.5.2. https://github.com/tesseract-ocr/tesseract
19. Camelot 2.0 — GitHub releases + TableFormer benchmark 2511.16134. https://github.com/camelot-dev/camelot
20. Tabula — tabula-java (last 2021) + TableFormer benchmark. https://github.com/tabulapdf/tabula-java
21. Unstructured.io — PyPI + pricing + Procycons (51 s/pg hi_res). https://github.com/Unstructured-IO/unstructured
22. PyMuPDF / pymupdf4llm — GitHub + AGPL LICENSE. https://github.com/pymupdf/pymupdf4llm
23. pdfplumber — GitHub. https://github.com/jsvine/pdfplumber
24. Azure Document Intelligence — pricing + OCR language support (TR/AR). https://learn.microsoft.com/azure/ai-services/document-intelligence/
25. Google Document AI — pricing + language support (TR/AR). https://cloud.google.com/document-ai/
26. AWS Textract — pricing + FAQ (no TR/AR). https://aws.amazon.com/textract/
27. ParseBench — arXiv 2604.08538 + LlamaIndex blog (COI). https://arxiv.org/abs/2604.08538
28. Reducto — pricing + supported-languages (TR + AR). https://docs.reducto.ai/parsing/supported-languages
29. Mistral pricing — mistral.ai/pricing. https://mistral.ai/pricing/
30. Gemini API pricing/document-processing — **(undefined)**. https://ai.google.dev/gemini-api/docs/pricing
31. Marker — GitHub + GPL LICENSE + Datalab pricing. https://github.com/datalab-to/marker
32. OmniDocBench — CVPR 2025, arXiv 2412.07626 + Real5 2603.04205. https://github.com/opendatalab/OmniDocBench
33. Firecrawl — Best PDF Parsers for RAG 2026 (hybrid/region routing) — **(undefined)**. https://www.firecrawl.dev/blog/best-pdf-parsers
34. Unstructured — 4 PDF parsing strategies for RAG — **(undefined)**. https://unstructured.io/blog/mastering-pdf-transformation-strategies-with-unstructured-part-2
35. Applied AI — The State of PDF Parsing — **(undefined)**. https://www.applied-ai.com/briefings/pdf-parsing-benchmark/
36. olmOCR — Unlocking Trillions of Tokens (anchoring, 176 USD/M) — arXiv 2502.18443. https://arxiv.org/html/2502.18443v1
37. Benchmarking PDF Parsers on Table Extraction (HTML > MD; LLM-judge r=0.93) — arXiv 2603.18652. https://arxiv.org/html/2603.18652v1
38. PulseBench-Tab — arXiv 2606.07534 (Pulse AI COI, T-LAG). https://arxiv.org/abs/2606.07534
39. PureDocBench — arXiv 2605.07492 ("parsing NOT solved"). https://github.com/zhihengli-casia/PureDocBench
40. KITAB-Bench (Arabic OCR) — arXiv 2502.14949. https://arxiv.org/html/2502.14949
41. OCRTurk (Turkish OCR benchmark) — ACL 2026 / arXiv 2602.03693. https://aclanthology.org/2026.sigturk-1.16.pdf
42. GlotOCR Bench — arXiv 2604.12978 (dots.ocr Arabic 63.2% Acc@5). https://arxiv.org/html/2604.12978v1
43. Mathematical Formula Extraction Benchmark — ICPR 2026, arXiv 2512.09874. https://arxiv.org/html/2512.09874v1
44. OCRBench v2 — NeurIPS 2025, arXiv 2501.00321. https://arxiv.org/abs/2501.00321
45. Upstage Document Parse — product + pricing + langchain-upstage. https://www.upstage.ai/products/document-parse
46. Adobe PDF Extract API — overview + pricing. https://developer.adobe.com/document-services/docs/overview/pdf-extract-api/
47. Chunkr (Lumina AI, AGPL-3.0) — GitHub + pricing. https://github.com/lumina-ai-inc/chunkr
48. LlamaParse — v2 blog + pricing + deprecation notice (ends 2026-05-01). https://github.com/run-llama/llama_cloud_services
49. Claude Fable 5 — Anthropic announcement + Roboflow eval — **(undefined; extraordinary claim, section 11)**. https://www.anthropic.com/news/claude-fable-5-mythos-5
50. shared/llm/README.md — "Adding a new provider (Google Gemini)" recipe (in-repo). path: shared/llm/README.md
