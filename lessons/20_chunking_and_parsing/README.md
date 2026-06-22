# Lesson 20 · Chunking & Parsing Strategies

## What you'll learn

- The two halves of "document ingestion": **parsers** turn bytes into `Document`s, **chunkers** turn `Document`s into smaller `Document`s.
- The four chunkers you'll actually use (and when to pick which): **Character**, **Recursive**, **Token-aware**, **Markdown-header**.
- The three parser tiers (text → structured → AI-powered) and where each shines.
- Why "chunk size" is a model-budget decision, not just a length decision.

## Why it matters

Bad retrieval almost always traces back to one of two things: **bad
chunks** (you split mid-sentence; the meaningful neighbour is in
another chunk) or **bad parsing** (you fed `pypdf` a scanned PDF and
got garbage). Lessons 06 / 07 / `rag_qa_api` all assume good chunks.
This lesson is the upstream layer.

If you've seen the [**Guillotine**](https://github.com/theFellandes/Guillotine)
project, the same five-axis taxonomy applies — that's a much deeper
treatment for the production case.

## Key concepts

### The pipeline

```
raw bytes ──parser──► Document(s) ──chunker──► smaller Document(s) ──embedder──► vectors
            (tier 1-3)             (4 strategies)
```

### The four chunkers — picking one

| Chunker | Splits on | Best for | Worst for |
|---|---|---|---|
| **CharacterTextSplitter** | one literal separator (e.g. `"\n\n"`) | known structure (e.g. logs with `---`) | natural prose — too brittle |
| **RecursiveCharacterTextSplitter** | tries `["\n\n", "\n", " ", ""]` in order | **default for prose** — what lesson 06 uses | binary data, code (use `from_language`) |
| **TokenTextSplitter** (tiktoken) | model tokens, not characters | **fitting a model's context exactly**, multilingual text | when you don't have tiktoken (e.g. non-OpenAI tokenizers) |
| **MarkdownHeaderTextSplitter** | `#`, `##`, `###` headings | structured docs, wikis, handbooks — keeps section context | non-Markdown content |

**Rule of thumb:** start with Recursive. Move to Token-based the
moment your prompt budgets get tight. Add Markdown-Header when your
docs have meaningful structure you'd otherwise lose.

### Chunk size & overlap

- **`chunk_size`** is the *target* — splitters usually undershoot, never overshoot.
- **`chunk_overlap`** is how much the next chunk repeats from the end of the previous one. 10–20% of `chunk_size` is typical. Overlap = context for the model when the answer straddles a boundary.
- Smaller chunks → more precise retrieval, more chunks to embed, higher search latency, more redundancy. Larger chunks → fewer hits but more context per hit.

### The three parser tiers

| Tier | Tools | What it handles |
|---|---|---|
| **1 · Text-native** | `TextLoader`, `JSONLoader`, `CSVLoader`, `BSHTMLLoader` | Plain text, JSON, CSV, well-formed HTML. Fast, no model. |
| **2 · Structured docs** | `PyPDFLoader`, `Docx2txtLoader`, `UnstructuredFileLoader` | Born-digital PDFs, Word, PowerPoint. Captures pages + basic structure. |
| **3 · AI-powered / layout-aware** | [**Docling**](https://github.com/DS4SD/docling), [**Unstructured**](https://github.com/Unstructured-IO/unstructured), [**LlamaParse**](https://github.com/run-llama/llama_parse), Azure Document Intelligence | Scanned PDFs, complex layouts, tables-as-tables, hand-written content. Slower, sometimes paid. |

Pick the lowest tier that produces correct output. If you can answer
your eval questions with `PyPDFLoader`, don't pay for LlamaParse.

## Walk through `example.py`

The script does the absolute minimum to compare strategies on the
**same input file** (`data/sample_docs/langgraph_intro.md`):

1. **`parse_text`** — `TextLoader` → one `Document` covering the whole file.
2. **`parse_pdf`** — `PyPDFLoader` → one `Document` per page (skipped cleanly if no PDF on disk).
3. **Four chunkers** run on the parsed doc, each printing `len(chunks)` and the first chunk's first line so you can compare granularity at a glance.

Look at the metadata after `MarkdownHeaderTextSplitter` — each chunk
carries `h1`/`h2`/`h3` headers, so when you build a prompt you can
prepend the section context "for free."

## Run it

```bash
uv run python -m lessons.20_chunking_and_parsing.example
```

Drop a small `.pdf` into `data/sample_docs/` and re-run to see
PyPDFLoader's output too.

## Debug it

Put `breakpoint()` after each chunker call and compare:

```text
ipdb> [len(c.page_content) for c in chunks]
ipdb> chunks[0].metadata
```

Look for **size variance** — wildly different chunk sizes from
"size=300" usually mean your separator hierarchy is wrong for the
text. Recursive's `keep_separator=True` (the default) is the most
common cause of small surprises.

## Beyond these four — the rest of the landscape

| Strategy | When | Where to find it |
|---|---|---|
| **Code-aware** (per language) | source code | `RecursiveCharacterTextSplitter.from_language(Language.PYTHON, ...)` |
| **HTML / DOM** | scraped web pages | `HTMLHeaderTextSplitter`, `HTMLSemanticPreservingSplitter` |
| **Semantic** | very long prose where headers/structure don't exist | `SemanticChunker` (langchain-experimental) — slices on embedding-distance jumps |
| **Sentence-aware** | small, multilingual, retrieval-critical | Guillotine's `SentenceChunker` (Turkish/Arabic/English regex), or `NLTKTextSplitter` |
| **Layout-aware** (regions, tables) | scientific papers, financial reports | Docling / Unstructured / LlamaParse (then run a normal chunker on their output) |
| **Hierarchical / "parent-doc"** | retrieval at small granularity, generation with parent context | `ParentDocumentRetriever` (lesson 07) on top of a small chunker |

> **See also:** [VLM-based PDF→Markdown extraction research](../../docs/research/vlm-pdf-extraction/FINDINGS.md)
> goes deeper than the Layout-aware row above on producing *chunker-friendly*
> Markdown (preserved `#`/`##`/`###` headings for `MarkdownHeaderTextSplitter`,
> tables as Markdown/HTML, figure refs as chunk metadata) and on multilingual
> fidelity (Turkish / Arabic / English).

## Try it yourself

- Add a 5th chunker: `RecursiveCharacterTextSplitter.from_language(Language.PYTHON, chunk_size=500)`. Feed it `shared/llm/base.py`. Notice it splits on `def` / `class` boundaries.
- Generate a quick comparison table: run all four chunkers on `data/sample_docs/company_handbook.md`, then a small RAG query through each — which retriever gives the best answer to "what's the refund policy?"
- Swap `pypdf` for [`pdfplumber`](https://github.com/jsvine/pdfplumber) and notice the difference on a PDF with tables.

## Next →

Up to you — head back to [Tier 3 · LangGraph core](../08_langgraph_basics/README.md), or jump into a [capstone](../../projects/).
