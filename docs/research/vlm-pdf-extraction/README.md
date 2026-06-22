# `docs/research/vlm-pdf-extraction/`

Long-form research synthesis on using **vision-language models (VLMs)
via Gemini** to turn PDFs into clean, chunker-friendly Markdown — the
"why-this-way" behind the Gemini provider added to `shared/llm/` and the
extraction path feeding [Guillotine](https://github.com/theFellandes/Guillotine)
and the `rag_qa_api_pro` capstone. Follows the `docs/research/`
convention: framed question → search log → findings graded by evidence
quality → recommended pattern → full numbered citations → evidence-grading
table. The focus is **chunker-friendly output** (preserved headings,
tables as Markdown/HTML, figure references as metadata) and **multilingual
fidelity** (Turkish / Arabic / English), because that is what the
downstream consumers actually need.

| File | Topic | Feeds into |
|---|---|---|
| [`RESEARCH-PROMPT.md`](RESEARCH-PROMPT.md) | The brief — framed question, scope, repo constraints (switchable provider layer, Gemini-per-README, `uv`-only), success criteria | [Lesson 20 · Chunking & parsing](../../../lessons/20_chunking_and_parsing/README.md) · [Lesson 36 · Library landscape](../../../lessons/36_library_landscape/README.md) · [Lesson 37 · Multimodal](../../../lessons/37_multimodal/README.md) |
| [`FINDINGS.md`](FINDINGS.md) | The synthesis — VLM-vs-OCR-vs-layout-parser trade-offs, Markdown/heading/table fidelity, multilingual (TR/AR/EN) script quality, batchable + fault-tolerant + observable RAG ingestion, recommended pattern graded by evidence | [Lesson 20](../../../lessons/20_chunking_and_parsing/README.md) · [Lesson 36](../../../lessons/36_library_landscape/README.md) · [Lesson 37](../../../lessons/37_multimodal/README.md) · [Guillotine](https://github.com/theFellandes/Guillotine) (chunking consumer) · [`projects/rag_qa_api_pro/`](../../../projects/rag_qa_api_pro/) (capstone consumer) |
| [`METHODS-ARCHITECTURE.md`](METHODS-ARCHITECTURE.md) | The **how-it-works** companion — all 7 extraction methods step-by-step (architecture + ASCII pipeline + per-method image handling + failure modes), the **figure lifecycle** (extract → caption → store → reference → index), and mermaid decision / hybrid / agentic / two-pass diagrams. Each method links a runnable demo | [Lesson 20](../../../lessons/20_chunking_and_parsing/README.md) · [Lesson 37](../../../lessons/37_multimodal/README.md) · [`demos/`](../../../projects/vlm_extraction_harness/demos/README.md) (8 runnable demos) |
| [`TEST-PLAN.md`](TEST-PLAN.md) | The Gemini harness spec — how to validate the `google` adapter end-to-end via `get_llm("google", ...)` + `with_fallbacks`, MarkdownHeaderTextSplitter round-trip checks, TR/AR/EN fixtures, batch/fault-tolerance assertions | [Guillotine](https://github.com/theFellandes/Guillotine) · [`projects/rag_qa_api_pro/`](../../../projects/rag_qa_api_pro/) |
| [`VLM-COMPARISON.md`](VLM-COMPARISON.md) | **Bring your own VLM** — trade-offs vs Gemini/Claude, how to wire a `local` provider (any OpenAI-compatible server: vLLM/Ollama/LM Studio/TGI), and a runnable head-to-head comparison ([`compare_vlms.py`](../../../projects/vlm_extraction_harness/compare_vlms.py) + [`metrics.py`](../../../projects/vlm_extraction_harness/metrics.py)) | [`projects/vlm_extraction_harness/`](../../../projects/vlm_extraction_harness/README.md) · [Guillotine](https://github.com/theFellandes/Guillotine) |

## Why this lives here

Mirroring the parent [`docs/research/README.md`](../README.md): **lessons
are *how-to*, research docs are *why-this-way*.** Lessons 20 / 36 / 37
introduce chunking, the library landscape, and VLM-based PDF
understanding at an intro depth — this folder goes **deeper** on the one
question those intros leave open ("when you reach for a VLM to produce
Markdown for a real RAG ingestion path, what actually works, on which
scripts, with what failure modes?"). Keeping the synthesis in version
control means the recommended Gemini-adapter pattern is **citable** back
to primary sources, and when the field moves on the doc tells the next
reader what was true on the search date and what to re-verify.
