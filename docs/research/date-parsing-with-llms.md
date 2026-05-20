# Date parsing with LLMs — research synthesis

**Search date:** 2026-05-20
**Databases:** Web search (Google), PubMed, arXiv, vendor docs (OpenAI, Anthropic, LangChain, Pydantic), GitHub repos (Outlines, lm-format-enforcer, dateparser, dateutil, Recognizers-Text, Duckling), practitioner blogs
**Question:** How do production teams reliably parse dates (especially birth dates) from text using LLMs? What goes wrong, what works?
**Records screened:** ~50 → shortlisted: 14 → deep-read: 9

---

## TL;DR

1. **LLMs alone are not reliable enough for date extraction.** A peer-reviewed clinical study found GPT-4 produced **19 wrong dates** (and 3 hallucinations) on the MMSE-extraction sub-task; LlaMA-2-70b produced **23 wrong dates + 27 hallucinations** on the same data [1]. Treat the LLM as a *first pass*, not as the authority.
2. **The canonical fix is a four-layer pipeline** — prompt that asks for ISO 8601, schema that validates structurally, deterministic date library that re-parses what the LLM returned, and Pydantic field validators that enforce domain rules (birth date ≥ 1900, ≤ today). Every layer catches something the others miss.
3. **Don't let the LLM compute relative dates.** "Last Tuesday", "5 days ago", "born 35 years ago" — anchor with `RELATIVE_BASE = today` in `dateparser` or compute the date yourself. Models are bad at calendar arithmetic and confidently wrong.

---

## The 10 failure modes (all documented)

| # | Failure | Example | Evidence |
|---|---|---|---|
| 1 | Format ambiguity | `04/05/87` → April 5 or May 4? | All libraries ship `DATE_ORDER` settings precisely because of this [3, 5] |
| 2 | Two-digit year | `04/05/87` → 1987 or 2087? | dateparser defaults to past for 2-digit years [5] |
| 3 | Relative dates | "5 days ago" | dateparser supports native; LLMs do the arithmetic poorly [3, 5] |
| 4 | Natural language | "5 mayıs 1987" | dateparser handles 200+ locales [3] |
| 5 | Non-Gregorian calendars | `1366/2/16` (Jalali) | dateparser ships `JalaliCalendar`, `HijriCalendar` modules [5] |
| 6 | Invalid dates | `Feb 31, 1990` | LLMs emit them; Pydantic `date` rejects them [8] |
| 7 | OCR / typo digits | `19B7` | Documented as "token merging failures" in vision-language literature [2] |
| 8 | Hallucination | LLM invents a date | GPT-4 = 3 cases, LlaMA-2 = 27 cases in one clinical study [1] |
| 9 | Partial dates | "born in '87", "March 1990" | dateparser's `PREFER_DAY_OF_MONTH` is the standard knob [5] |
| 10 | "X years old" → year | "She is 35" → DOB? | LLM arithmetic; always anchor with explicit `RELATIVE_BASE` [5] |

---

## What's settled (high-confidence findings)

### A. ISO 8601 is the unanimous output target

Every secondary source — the OpenAI structured-outputs docs, Anthropic tool-use guidance, James Tang's "best practices" article, MLJAR / TianPan / Databricks blog posts — says the same thing:

> Output dates as `YYYY-MM-DD` (or `YYYY-MM-DDTHH:MM:SSZ` with time). State this in the prompt, the schema, *and* the field description. Same format on input and output. [4, 7, 9, 10]

Rationale: zero regional ambiguity, sortable as strings, machine + human readable, standardised. The anti-pattern list is equally crisp:

- ❌ Localised strings (`27/11/2023`)
- ❌ Relative language in the *output* (`tomorrow`)
- ❌ Unix timestamps (seconds vs milliseconds ambiguity is a real bug source)
- ❌ Native `datetime` objects that don't survive JSON serialisation

### B. Structural enforcement beats hopeful prompting

The OpenAI structured-outputs docs (GPT-4o ≥ 2024-08-06) and Anthropic's structured-outputs / tool-use docs both expose **deterministic constrained decoding** for JSON schema. When available, schema *violation* becomes impossible — the decoder cannot emit a non-matching token [4, 9, 10]. This is the strongest single lever.

For self-hosted / open-weight models, the equivalents are: **Outlines** (FSM-based), **lm-format-enforcer** (token filter), **XGrammar**, **vLLM structured outputs**, **llama.cpp + GBNF** [6]. None currently enforce JSON Schema's `format: "date"` keyword end-to-end — the practical workaround is a regex like `\d{4}-\d{2}-\d{2}` inside the schema's string field [6].

### C. Rule-based libraries are still the gold standard for *normalisation*

Even after the LLM hands you a string, parse it once more with a deterministic library so `Feb 31` becomes a `ValueError` instead of a corrupt database row.

| Library | Native lang | Multilingual | Non-Greg calendars | Relative dates | Notes |
|---|---|---|---|---|---|
| `python-dateutil` | Python | English-only | ❌ | ❌ | Fast, default for English. dateparser uses it internally [3] |
| `dateparser` | Python | 200+ locales | ✅ Jalali + Hijri | ✅ rich | ~8× slower than dateutil; the most capable Python library [3, 5] |
| **HeidelTime** | Java | 13+ langs | partial | ✅ | Academic gold standard; rule-based, very high precision [11, 12] |
| **SUTime** (Stanford) | Java | English | ❌ | ✅ | Lower precision than HeidelTime in head-to-head [11] |
| **Facebook Duckling** | Haskell | many | partial | ✅ | Powerful but needs a running server; deployment friction [13] |
| **MS Recognizers-Text** | .NET / JS / Python | 10+ langs | partial | ✅ | Powers LUIS / Bot Framework; offline; the Duckling alternative people pick when they don't want a Haskell server [13] |

**Rule of thumb:** `dateparser` for breadth, `dateutil` for speed-critical English-only, MS Recognizers-Text when you need a polyglot library with a non-Python team.

---

## What's contested (lower confidence)

- **Whether to fine-tune for date extraction.** No primary evidence that fine-tuning beats prompt + schema + library for *generic* date fields. Likely worth it only for unusual domain formats (legal citations, archaic / historical notation).
- **Whether constrained decoding's JSON-mode hurts quality on adjacent fields.** vLLM / lm-format-enforcer maintainers note subtle quality degradation on long schemas. Mixed reports. A/B test on your own data.
- **`with_structured_output(method="function_calling")` + Pydantic `date` field.** A live LangChain issue (#29604) reports a 400 error with `ChatMistralAI` + `json_schema` mode + `datetime.date` [4]. **Workaround**: type the field as `str` and run a Pydantic validator that parses it. This bites people in production.

---

## What's an open question

- **Multi-document birth-date disambiguation** (two pages disagree → which wins?) — no standard pattern.
- **Cross-calendar normalisation at scale** (Persian-only user supplies 1366/2/16 — what's the Gregorian equivalent, and do you record both?). Libraries exist; production wrappers don't.
- **OCR-degraded dates** (handwritten / scanned forms) — currently the domain of vision-language models; fast-moving area, re-check quarterly.

---

## The recommended pipeline (defence-in-depth)

```
              ┌───────────────────────────────────────────────────────┐
              │  LAYER 1 · PROMPT                                     │
              │   • "Today is YYYY-MM-DD"  (anchors relative refs)    │
              │   • "Return null if not present"  (anti-hallucination)│
              │   • 3-5 few-shot examples (DD/MM, MM/DD, name month)  │
              │   • "Use ISO 8601 (YYYY-MM-DD) in the date field"     │
              └─────────────────────┬─────────────────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────────────────┐
              │  LAYER 2 · STRUCTURAL (provider-side constrained dec) │
              │   • OpenAI structured outputs (strict=True)           │
              │   • Anthropic tool use       (strict=True)            │
              │   • Outlines / lm-format-enforcer for self-hosted     │
              │   • Schema: { birth_date: str | null,                 │
              │              raw_text: str,                           │
              │              confidence: 0-1 }                        │
              └─────────────────────┬─────────────────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────────────────┐
              │  LAYER 3 · DETERMINISTIC RE-PARSE                     │
              │   • dateparser.parse(str, settings={                  │
              │       DATE_ORDER: locale,                             │
              │       RELATIVE_BASE: today,                           │
              │       PREFER_DAY_OF_MONTH: 'first',                   │
              │       STRICT_PARSING: True })                         │
              │   • catches Feb 31, surfaces ambiguity                │
              └─────────────────────┬─────────────────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────────────────┐
              │  LAYER 4 · DOMAIN VALIDATION (Pydantic field_validator)│
              │   • date ≥ 1900-01-01                                  │
              │   • date ≤ today                                       │
              │   • plausible age (0-120 for living human)             │
              │   • On failure → InstructorRetryException with the    │
              │     specific error → LLM gets a second chance         │
              └─────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
                            (date, source_text, confidence)
                            → log audit trail
                            → human review if confidence < 0.7
```

Each layer catches what the previous misses:

| Layer | Catches |
|---|---|
| 1 · Prompt (today + ISO + few-shot + "don't invent") | Format ambiguity, two-digit year, relative dates, anti-hallucination |
| 2 · Structured output (provider-side constrained decoding) | JSON-shape errors, wrong field types, markdown-wrapped JSON |
| 3 · `dateparser.parse(STRICT_PARSING=True)` | Feb-31, OCR garbage, weird locale strings that passed the schema regex |
| 4 · Field validator (after 1900, not future, plausible) | Hallucinated valid-looking dates that aren't in the source |

Wrap the whole thing in **Instructor** (or LangChain's `with_structured_output` + retry middleware) and a Pydantic `ValidationError` becomes the prompt for retry attempt #2 with the specific error message [8, 14]. That's the validation-feedback loop that delivers the rest of the reliability.

---

## Evidence grading

| Claim | Grade | Why |
|---|---|---|
| LLMs make 15-25% date errors on clinical extraction tasks at GPT-4 quality | **Moderate** | 1 peer-reviewed study; replicated across 2 model families; single domain [1] |
| ISO 8601 is the right output format | **High** | Unanimous across 8+ practitioner sources; aligned with W3C / RFC 3339 |
| dateparser handles multilingual / non-Gregorian; dateutil doesn't | **High** | Stated in primary docs of both libraries [3, 5] |
| HeidelTime > SUTime in precision | **High** | Multiple peer-reviewed comparisons [11, 12] |
| Constrained decoding eliminates JSON-shape errors | **High** | Vendor docs (OpenAI, Anthropic) + benchmark paper [4, 6, 9] |
| Pydantic validation + retry > one-shot prompting | **Moderate** | Strong practitioner consensus; Instructor is widely adopted [8, 14]; no large head-to-head benchmark |
| Fine-tuning helps for generic date extraction | **Low / unclear** | No evidence that beats prompt + schema + library for ordinary fields |

---

## Cheapest experiment if you want to verify on your data

Build a 30-item eval set covering: 5× DD/MM, 5× MM/DD, 5× DD-Month-YYYY, 5× ISO, 5× relative ("5 years ago"), 3× non-Gregorian, 2× invalid (Feb 31), 5× no date present.

Run the four-layer pipeline. Track:

- **Strict accuracy** — exact ISO match
- **Hallucination rate** — date emitted when none present (should be 0)
- **Validation-catch rate** — fraction of LLM errors caught by layers 3 + 4
- **Retry-success rate** — fraction of layer-4 failures fixed on retry-with-error

A good production target: ≥ 98% strict accuracy on the easy classes, ≥ 95% on relative, 0 hallucinations.

---

## References

| # | Source |
|---|---|
| [1] | [Evaluating LLMs in Extracting Cognitive Exam Dates and Scores — PMC11634005](https://pmc.ncbi.nlm.nih.gov/articles/PMC11634005/) — primary peer-reviewed evidence of GPT-4 vs LlaMA-2 date-extraction failure rates |
| [2] | [LLM-driven transferable key information extraction — Nature Scientific Reports](https://www.nature.com/articles/s41598-025-15627-z.pdf) — failure-mode taxonomy incl. token-merging on dates |
| [3] | [dateparser vs dateutil — Zyte blog](https://www.zyte.com/blog/parse-natural-language-dates-with-dateparser/) — feature comparison |
| [4] | [LangChain structured output docs](https://docs.langchain.com/oss/python/langchain/structured-output) + [issue #29604 (Mistral + date field)](https://github.com/langchain-ai/langchain/issues/29604) |
| [5] | [dateparser official docs](https://dateparser.readthedocs.io/) — Jalali/Hijri calendar APIs and settings |
| [6] | [Constrained Decoding guide — aidancooper.co.uk](https://www.aidancooper.co.uk/constrained-decoding/) + [lm-format-enforcer repo](https://github.com/noamgat/lm-format-enforcer) + [JSONSchemaBench paper (arXiv 2501.10868)](https://arxiv.org/pdf/2501.10868) |
| [7] | [Best Practices for Handling Dates in Structured Output in LLM — James Tang, Medium](https://medium.com/@jamestang/best-practices-for-handling-dates-in-structured-output-in-llm-2efc159e1854) — ISO 8601 rationale + anti-patterns |
| [8] | [Pydantic for validating LLM outputs — MachineLearningMastery](https://machinelearningmastery.com/the-complete-guide-to-using-pydantic-for-validating-llm-outputs/) — field validators + retry pattern |
| [9] | [Anthropic Claude structured outputs + tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — `strict: true`, `input_examples`, temperature 0.2 guidance |
| [10] | [OpenAI structured outputs — introducing-structured-outputs-in-the-api](https://openai.com/index/introducing-structured-outputs-in-the-api/) — deterministic constrained decoding (GPT-4o ≥ 2024-08-06) |
| [11] | [HeidelTime vs SUTime — Stanford NLP TempEval](https://nlp.stanford.edu/pubs/lrec2012-sutime.pdf) + [HeidelTime for German](https://arxiv.org/pdf/2204.08848) |
| [12] | [Multilingual recognition of temporal expressions](https://nlp.fi.muni.cz/raslan/2020/paper2.pdf) + [TEI2GO arXiv 2403.16804](https://arxiv.org/html/2403.16804v1) |
| [13] | [Microsoft Recognizers-Text repo](https://github.com/microsoft/Recognizers-Text) + [Facebook Duckling repo](https://github.com/facebook/duckling) |
| [14] | [Instructor library — retry mechanisms](https://python.useinstructor.com/learning/validation/retry_mechanisms/) — the validation-feedback-loop pattern |
