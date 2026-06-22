# `docs/research/`

Long-form research syntheses that fed into the lessons. Each one
follows the `scientific-paper-researcher` skill's deliverable
template: framed question, search log, findings graded by evidence
quality, recommended pattern, full citations.

| File | Topic | Feeds into |
|---|---|---|
| [`date-parsing-with-llms.md`](date-parsing-with-llms.md) | **Round 1** — failure modes, prompting, structured output, constrained decoding, rule-based libraries | [Lesson 21](../../lessons/21_date_parsing/README.md) |
| [`langchain-date-field-bug.md`](langchain-date-field-bug.md) | Reproduced LangChain `with_structured_output` + `datetime.date` bug; root cause in `_rm_titles`; three workarounds | [Lesson 21](../../lessons/21_date_parsing/README.md) + [`shared/llm/schema_sanitizer.py`](../../shared/llm/schema_sanitizer.py) |
| [`llm-date-solutions-deep-dive.md`](llm-date-solutions-deep-dive.md) | **Round 2** — 10-approach taxonomy (guardrails / prompts / sanitizers / VLM / NER / agentic / validation libraries / specialised parsers), academic literature review (NAACL 2024, DateLogicQA, Test of Time, TemporalBench), 4 production recipes | [Lesson 21](../../lessons/21_date_parsing/README.md) + [Lesson 22 · Architecture](../../lessons/22_architecture/README.md) |
| [`vlm-pdf-extraction/`](vlm-pdf-extraction/README.md) | **VLM PDF → Markdown** — 41-tool inventory on a 14-dim rubric, 7-method taxonomy, best-practice playbook, image→Markdown deep dive, adversarial verdicts, + a Gemini test-harness spec. Recommends Docling (MIT) primary + hosted/Gemini VLM fallback via the `get_llm("google", …)` adapter | [Lesson 20](../../lessons/20_chunking_and_parsing/README.md) · [Lesson 36](../../lessons/36_library_landscape/README.md) · [Lesson 37](../../lessons/37_multimodal/README.md) + [Guillotine](https://github.com/theFellandes/Guillotine) / `rag_qa_api_pro` |

## Why these live in the repo

Lessons are *how-to*. Research docs are *why-this-way*. Keeping the
synthesis in version control means:

- the lesson's recommended pattern is **citable** back to primary sources
- when the field moves on, the research doc tells the next reader
  what was true on the search date and what should be re-verified
- forks / contributors can update either independently

## Adding a new research doc

1. Use the structure from `date-parsing-with-llms.md` as a template.
2. **Header** with search date, databases, records-screened →
   shortlisted → deep-read counts.
3. Mark every claim with a numbered citation.
4. End with an **evidence-grading table** (high / moderate / low).
5. Link from the lesson that uses it, and from this README.
