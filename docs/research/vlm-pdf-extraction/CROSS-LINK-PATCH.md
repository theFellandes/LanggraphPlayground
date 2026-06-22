# Cross-link patch — proposed (do NOT auto-apply)

Three **proposed** edits that point lessons 20 / 36 / 37 at the deeper
VLM-PDF-extraction research. These are intentionally *cross-links, not
duplications* — each lesson is an intro; [`FINDINGS.md`](FINDINGS.md)
goes deeper (multilingual TR/AR/EN script fidelity, Markdown/heading/table
round-trip quality, batchable + fault-tolerant + observable RAG ingestion,
Gemini-adapter wiring). A human should apply each by hand so the link
text matches the surrounding prose. The relative path from each lesson
README is `../../docs/research/vlm-pdf-extraction/FINDINGS.md`.

Note for the applier: the path resolves from `lessons/<n>/README.md` →
repo root → `docs/research/vlm-pdf-extraction/FINDINGS.md`. Verify it
renders before committing.

---

## 1. `lessons/20_chunking_and_parsing/README.md`

**Where:** in the **"Beyond these four — the rest of the landscape"**
table, the **Layout-aware** row is the natural anchor — that row already
defers PDF parsing to Docling / Unstructured / LlamaParse, which is
exactly the decision this research deepens (when a *VLM* producing
Markdown beats those, and how the output stays chunker-friendly).

**Surrounding line (quote, for locating):**

```markdown
| **Layout-aware** (regions, tables) | scientific papers, financial reports | Docling / Unstructured / LlamaParse (then run a normal chunker on their output) |
```

**Proposed insertion** — add a sentence immediately **after** that table
(before the `## Try it yourself` heading), so it reads as a "go deeper"
pointer rather than altering the table:

```markdown
> **See also:** [VLM-based PDF→Markdown extraction research](../../docs/research/vlm-pdf-extraction/FINDINGS.md)
> — goes deeper than this table on producing *chunker-friendly* Markdown
> (preserved `#`/`##`/`###` headings for `MarkdownHeaderTextSplitter`,
> tables as Markdown/HTML, figure refs as chunk metadata) and on
> multilingual fidelity (Turkish / Arabic / English).
```

---

## 2. `lessons/36_library_landscape/README.md`

**Where:** the **"Web ingestion (the RAG-input problem)"** section is the
closest existing home — it already frames "give me clean text/markdown
from a source" as a distinct problem class. The VLM-PDF research is the
PDF-shaped sibling of that web-ingestion problem.

**Surrounding line (quote, for locating)** — the section heading:

```markdown
## Web ingestion (the RAG-input problem)
```

**Proposed insertion** — add a short pointer line immediately **under**
that heading, before the `### Firecrawl` subsection:

```markdown
> **PDF-input sibling:** for the PDF-shaped version of this problem
> (rendering pages to a VLM to get clean Markdown), see the deeper
> [VLM-based PDF→Markdown extraction research](../../docs/research/vlm-pdf-extraction/FINDINGS.md).
```

*Alternative anchor* if the applier prefers to keep it near the cross-link
list: the **"Pairs with"** list (the lines under `## Pairs with`) — append
a bullet:

```markdown
- **[VLM PDF→Markdown research](../../docs/research/vlm-pdf-extraction/FINDINGS.md)** — Gemini-adapter PDF extraction for chunker-friendly Markdown
```

---

## 3. `lessons/37_multimodal/README.md`

**Where:** **Part 2 · VLM-based PDF understanding — the OCR replacement**,
specifically the **"When to use which"** bullet list. That list is where
the lesson stops at an intro depth ("Docling for bulk, direct VLM for
targeted") — the research extends it with the production criteria (output
must survive a real RAG ingestion path, multilingual script quality, the
`get_llm("google", ...)` wiring).

**Surrounding line (quote, for locating)** — the last bullet of that list:

```markdown
- **Scanned legal documents** → AWS Textract or Azure Document Intelligence (specialised; better at forms)
```

**Proposed insertion** — add a pointer line immediately **after** that
bullet, closing the "When to use which" subsection:

```markdown
> **See also:** [VLM-based PDF→Markdown extraction research](../../docs/research/vlm-pdf-extraction/FINDINGS.md)
> — goes deeper than this section on Gemini-as-VLM extraction wired through
> the `shared/llm/` switchable provider layer (`get_llm("google", ...)` +
> `with_fallbacks`), chunker-friendly Markdown for Guillotine, and
> multilingual (Turkish / Arabic / English) fidelity.
```

*Alternative anchor:* the **"Pairs with"** list at the bottom already links
lesson 20; append a sibling bullet there if a footer link is preferred:

```markdown
- **[VLM PDF→Markdown research](../../docs/research/vlm-pdf-extraction/FINDINGS.md)** — the production-depth version of Part 2
```

---

## Why deeper, not duplicate

| Lesson covers (intro) | `FINDINGS.md` adds (depth) |
|---|---|
| Docling / direct-VLM exist | Which one for *chunker-friendly Markdown*, graded by evidence |
| "tables preserved" as a claim | Table-as-Markdown/HTML round-trip through `MarkdownHeaderTextSplitter`, figure refs as metadata |
| Multimodal models are multilingual | TR / AR / EN script-level extraction fidelity, tested |
| `langchain-google-genai` one-liner | Gemini wired into the `shared/llm/` adapter pattern (`get_llm("google", ...)`, `_ADAPTERS`, `with_fallbacks`), `uv add` only |
| "fan-out over 200 invoices" | Batchable + fault-tolerant + observable ingestion for `rag_qa_api_pro` |
