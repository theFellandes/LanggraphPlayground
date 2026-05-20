# Lesson 21 · Date parsing with LLMs

> **This lesson is the INPUT side** — getting a clean ISO-8601 date *out
> of* messy text. The OUTPUT side ("what's today?", date arithmetic,
> localized rendering like *"23 Mayıs 2026"*) is **[Lesson 23 ·
> Date computation & localized output](../23_date_localization/README.md)**.

## What you'll learn

- Why LLMs alone are not reliable enough for date extraction (with **numbers** from a peer-reviewed clinical study).
- The **four-layer pipeline** — prompt → structured output → deterministic re-parse → domain validation — and what each layer catches that the others miss.
- The Python idioms: anchored prompt, Pydantic `field_validator`, `dateparser` with `STRICT_PARSING`, the ValidationError-as-retry-signal pattern.
- Where the industry libraries fit: `dateparser`, `dateutil`, **HeidelTime**, **SUTime**, **Duckling**, **Microsoft Recognizers-Text**.

## Why it matters

A clinical-NLP evaluation (Cui et al., *npj Digital Medicine* 2024) found
that **GPT-4 emitted 19 wrong dates** on a MMSE-extraction sub-task —
and **LlaMA-2-70b emitted 23 wrong dates + 27 hallucinations** on the
same notes. Production date fields (birth dates, expiry dates, event
dates) are load-bearing identity attributes; any error survives forever
in the database.

> See **[`docs/research/date-parsing-with-llms.md`](../../docs/research/date-parsing-with-llms.md)** for the full research synthesis with citations.

### How hard is birth-date extraction, really? — the nuance

A fair self-check: the headline failure numbers above are for *clinical
date extraction generally* (MMSE assessment dates, in unstructured prose
with multiple candidate dates), not for *birth date extraction
specifically*. Direct evidence on DOB extraction is more nuanced:

| Input shape | Difficulty | Evidence |
|---|---|---|
| Clean explicit DOB ("Born April 5, 1987" / "DOB: 1987-04-05") | **Easy** | NEJM AI 2024 (LLM-Anonymizer): *"Date of birth was extracted in nearly all letters for almost all models"*; KYC industry: 97–99% on clean ID docs |
| Ambiguous format ("04/05/87") | **Medium** | Routine industry experience; all libraries ship `DATE_ORDER` settings precisely because this is consistently confused |
| Relative ("born 35 years ago") | **Hard** — arithmetic fails | Test of Time (arXiv 2406.09170): *"approximate correct calculations but often stumble in final steps"* |
| Scanned ID / handwritten form | **Hard** — multimodal OCR hallucination | *Seeing is Believing?* (arXiv 2506.20168) — direct evidence on the failure mode |
| Non-Gregorian calendars (Jalali, Hijri, Japanese era) | **Hard** — normalisation fails | Lesson 23 + the cross-calendar gap noted in NAACL 2024 paper |

**So the 4-layer pipeline is insurance, not proof DOB is always hard.**
For clean explicit text it never trips. For the failure modes in rows
2-5 above — which is what real-world DOB-extraction pipelines actually
hit — it earns its keep.

The full nuanced picture (including all citations above) lives in the
**["Is birth-date extraction specifically hard?"](../../docs/research/llm-date-solutions-deep-dive.md#is-birth-date-extraction-specifically-hard--the-honest-nuance)**
section of the Round-2 deep-dive doc.

## Key concepts

### The 10 failure modes

| # | Failure | Example |
|---|---|---|
| 1 | Format ambiguity     | `04/05/87` → April 5 or May 4? |
| 2 | Two-digit year       | `87` → 1987 or 2087? |
| 3 | Relative dates       | "born last Tuesday", "5 days ago" |
| 4 | Natural language     | "5 mayıs 1987" |
| 5 | Non-Gregorian        | Persian `1366/2/16`, Hijri, Japanese era |
| 6 | Invalid dates        | LLM happily emits `Feb 31, 1990` |
| 7 | OCR / typo digits    | `19B7` |
| 8 | **Hallucination**    | LLM invents a date that wasn't in the text |
| 9 | Partial dates        | "born in '87", "March 1990" |
| 10 | "X years old"       | "She is 35" → which year is "now"? |

### The four-layer pipeline

```
[ LAYER 1 · PROMPT ]   "Today is YYYY-MM-DD" + ISO request + few-shot + "don't invent"
        │              catches: format ambiguity, two-digit year, relative dates, hallucination
        ▼
[ LAYER 2 · STRUCTURED OUTPUT ]   with_structured_output(BirthDateExtraction)
        │                         catches: JSON-shape errors, wrong field types
        ▼
[ LAYER 3 · dateparser.parse(STRICT) ]   re-parse with RELATIVE_BASE=today
        │                                catches: Feb-31, weird strings the schema didn't reject
        ▼
[ LAYER 4 · field_validator ]   birth_date ≥ 1900, ≤ today, age < 120
        │                       catches: hallucinated *valid-looking* dates outside the domain
        ▼
                                              (date, source_text, confidence)
```

Each layer is **defence in depth**. The cheap layers (1, 4) run on every
call; the more expensive ones (2, 3) are still cheap compared to a wrong
date in your database.

### Why `birth_date: str` and not `birth_date: date`

The default schema (`BirthDateExtraction`) deliberately types `birth_date` as `str | None`, not `datetime.date | None`. Two reasons:

1. **LangChain + some providers don't survive `date`-typed structured-output schemas.** A reproduced bug ([LangChain issue #29604](https://github.com/langchain-ai/langchain/issues/29604), **closed as "not planned"**) — Pydantic emits `"format": "date"` for a `date` field, and Mistral / OpenAI strict mode return HTTP 400 with *"Received unsupported keyword 'format' in schema."* Typing it as `str` and parsing it in a validator sidesteps the problem.
2. **Audit trail.** We want the LLM's raw output preserved so we can debug. The validator both **parses** and **normalises** to ISO; the round-trip lives in one place.

**Full investigation** with reproducer, root-cause code reference, and three solutions: **[`docs/research/langchain-date-field-bug.md`](../../docs/research/langchain-date-field-bug.md)**.

### The native-`date` alternative (Solution B)

If your downstream code really wants a `datetime.date` object, this repo ships a **schema sanitizer** at [`shared/llm/schema_sanitizer.py`](../../shared/llm/schema_sanitizer.py) that strips `"format": "date"` (and a few other strict-mode-incompatible keys) before LangChain sends the schema to the provider. The lesson's `BirthDateNative` schema demonstrates it:

```python
from datetime import date
from pydantic import BaseModel
from shared.llm import with_structured_output_safe

class BirthDateNative(BaseModel):
    birth_date: date | None       # native type — works again

llm = with_structured_output_safe(BirthDateNative)
result = llm.invoke("Born April 5, 1987.")
type(result.birth_date)           # → datetime.date
```

Run the same eval against Solution B with `--native`.

### Why one string field, not decomposed `day` / `month` / `year` ints

The instinct after seeing the date-field bug is reasonable: *"OK, I'll
side-step the whole format thing and just use plain integers"*. So the
schema becomes:

```python
# DON'T do this — the anti-pattern
@tool
def verify_customer_identity(
    birth_day:   Annotated[int, "Day (1-31). Convert spoken Turkish: 'on bir' → 11."],
    birth_month: Annotated[int, "Month (1-12). Convert: 'kasım' → 11, 'mart' → 3."],
    birth_year:  Annotated[int, "Year (4-digit). Convert: 'doksan yedi' → 1997."],
    father_name: Annotated[str, "Father's name exactly as spoken. Don't guess."],
): ...
```

It looks cleaner — clean integer types, no `format: "date"` quirk, no
parsing logic. **It fails worse than the string field.** Two reasons.

**1 · Hallucination surface scales with the number of fields.**
Every numeric field the model produces is an independent opportunity
to emit the wrong digit. A real example from a voice-input pipeline:
the user says *"hamdi yirmi beş on iki yetmiş dokuz"* (`hamdi / 25 /
12 / 79`). The model returns:

```python
{"birth_day": 25, "birth_month": 10, "birth_year": 1979, "father_name": "hamdi"}
#                              ^^ should be 12 — single-component hallucination
```

`birth_day` and `birth_year` are correct; `birth_month` drifted from
`12` to `10`. The model fills the four fields in sequence and any one
of them can drift independently — three numeric chances to fail instead
of one. **A coherent ISO date string (`"1979-12-25"`) is one token
sequence the model emits as a unit**, with the structure constraining
the components against each other.

**2 · Asking the LLM to do in-prompt Turkish-number conversion is the
wrong layer.** "On iki → 12" is a deterministic computation. Anything
deterministic belongs in code, not in a prompt the model has to
re-derive on every call. Lesson 23's whole point applies here too:
*the LLM picks the right tool; a tool does the arithmetic*.

**The right shape** has two acceptable forms:

```python
# Good — one ISO field; LLM does ONE conversion, your validator double-checks
@tool
def verify_customer_identity(
    birth_date:  Annotated[str, "Date of birth in ISO 8601 (YYYY-MM-DD). Convert from any spoken form."],
    father_name: Annotated[str, "Father's name exactly as spoken."],
): ...
# Same lesson-21 four-layer pipeline applies on birth_date.

# Best for voice / Turkish input — let the LLM identify the span, let a TOOL parse it
@tool
def verify_customer_identity(
    spoken_birth_date: Annotated[str, "Customer's DOB exactly as they said it (e.g., 'yirmi beş on iki yetmiş dokuz'). Do NOT convert."],
    father_name:       Annotated[str, "Father's name exactly as spoken."],
):
    # Server-side: deterministic parsing — dateparser handles Turkish locale + month
    # names, and a small Turkish-number helper handles 'yirmi beş' / 'doksan yedi' / etc.
    parsed = parse_turkish_spoken_date(spoken_birth_date)   # → date(1979, 12, 25)
    if parsed is None:
        return {"error": "could_not_parse", "hint": "ask the customer to repeat"}
    return verify(parsed, father_name)
```

The **best** shape pushes the conversion *out* of the LLM entirely.
The LLM's only job is to identify the relevant span and pass it
through unchanged — which is one of the things LLMs are *most*
reliable at, because there's no transformation to drift on.

| Schema shape | Hallucination surface | Fix when it breaks |
|---|---|---|
| 3 separate `int` fields + in-prompt Turkish→number map | **Worst** — 3 independent components, each can drift; numeric digits especially | Don't use; redesign |
| One `str` ISO field + `field_validator` | **Acceptable** — 1 string emission; coherent unit | Lesson-21 four-layer pipeline catches it |
| One `str` raw-span field + server-side parsing tool | **Best** — 0 in-prompt conversion; LLM does only the easy NER bit | Validator + tool error → ask the user to repeat |

This generalises beyond dates — see lesson 22's anti-patterns for the
broader "field decomposition multiplies hallucination" rule.

## Walk through `example.py`

The script has three entry points:

| Function | Needs API key? | What it shows |
|---|---|---|
| `main()` (default)         | yes | Full pipeline · **Solution A** (str + dateparser validator) on the 13-case eval set |
| `main_native()` (`--native`)| yes | Full pipeline · **Solution B** (`date` + schema sanitizer) on the same eval set |
| `demo_validation()` (`--validation-only`) | no  | Layers 3 + 4 only — fast, deterministic, no API call |

The 13-case `EVAL_SET` deliberately exercises:

- ISO clean, English long-form, DD/MM, MM/DD, two-digit year — *should pass*
- relative ("she is 35"), partial ("in 1987"), no date present, hallucination bait ("famous author") — *should return null*
- Feb 31, pre-1900, future date, age > 120 — *should be **rejected** by layer 3 or 4*

A real production eval would be 30+ cases per locale.

## Run it

**Without an API key** (validation layers only):

```bash
uv run python -m lessons.21_date_parsing.example --validation-only
```

You should see each raw string either accepted (with the normalised
ISO date) or rejected (with the layer-3 / layer-4 error message).

**With an API key** — two solutions you can A/B test:

```bash
# Solution A: birth_date: str + dateparser validator (universal)
uv run python -m lessons.21_date_parsing.example

# Solution B: birth_date: date + with_structured_output_safe (native typing)
uv run python -m lessons.21_date_parsing.example --native
```

Each line shows: ✓ / ✗, the case label, expected value, what the
pipeline returned. Aim for ≥ 12 / 13 with a frontier model.

## Debug it

Put `breakpoint()` inside `_layer3_reparse_and_layer4_sanity` and run
the validation demo:

```text
ipdb> pp v
ipdb> pp dateparser.parse(v, settings={"STRICT_PARSING": True})
ipdb> n
```

That's the cleanest spot to see what the LLM *actually* returned vs
what your schema *thought* it would return.

## Industry library comparison

| Library | Use it when |
|---|---|
| `dateparser` | Default for multilingual + relative + non-Gregorian (Jalali, Hijri). ~8× slower than dateutil. |
| `dateutil`   | English-only, speed-critical. dateparser uses it internally. |
| **HeidelTime** | Java; academic gold-standard rule-based system. Use via subprocess when you need its precision. |
| **SUTime** (Stanford) | Java; lower precision than HeidelTime but ships with Stanford CoreNLP. |
| **Facebook Duckling** | Haskell; needs a server. Powerful but deployment-heavy. |
| **Microsoft Recognizers-Text** | .NET / JS / Python; offline; multi-language. The Duckling alternative if you don't want a server. Powers LUIS / Bot Framework. |

This lesson uses **`dateparser`** because it covers the most ground per
line of code and ships as a Python package.

## Try it yourself

- **Add a Turkish test case** like `"5 mayıs 1987 doğumlu"` — see if your chosen LLM normalises it correctly.
- **Switch the prompt examples** to mostly DD/MM/YYYY and see whether the model handles US-format inputs less reliably.
- **Wrap with `instructor`** so a layer-4 `ValidationError` triggers an automatic retry with the error fed back as a system message.
- **Add a 5th layer** that calls Microsoft Recognizers-Text via subprocess and compares its answer to the LLM's — disagreement → human review.

## References & raw sources

Every claim in this lesson and in the two companion research docs is
traceable to a primary source. Grouped by source type so you can pick
your trust prior. The mix is deliberate: peer-reviewed papers for the
"how bad is the problem" claim, vendor docs for what the APIs actually
do today, GitHub for the bug status, engineering blogs for the
practitioner consensus.

### Peer-reviewed papers (PubMed Central / Nature / arXiv)

- **[Cui et al., "Evaluating Large Language Models in Extracting Cognitive Exam Dates and Scores" — PMC11634005](https://pmc.ncbi.nlm.nih.gov/articles/PMC11634005/)** · the clinical evaluation behind the "GPT-4 = 19 wrong dates / LlaMA-2-70b = 23 wrong dates + 27 hallucinations" numbers in the TL;DR. Same authors also at [PMC10888985](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10888985/) and [PMC11713360](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11713360/) (extended versions).
- **["Large language model driven transferable key information extraction" — Nature Scientific Reports 41598-025-15627-z](https://www.nature.com/articles/s41598-025-15627-z.pdf)** · failure-mode taxonomy including token-merging on date fields.
- **[Chang & Manning, "SUTIME: A Library for Recognizing and Normalizing Time Expressions" — Stanford LREC 2012](https://nlp.stanford.edu/pubs/lrec2012-sutime.pdf)** · the canonical SUTime paper.
- **[Hammami & Gelbukh, "I still have Time(s): Extending HeidelTime for German Texts" — arXiv 2204.08848](https://arxiv.org/pdf/2204.08848)** · HeidelTime extension paper, includes head-to-head precision numbers vs SUTime.
- **[Macedo et al., "TEI2GO: A Multilingual Approach for Fast Temporal Expression Identification" — arXiv 2403.16804](https://arxiv.org/html/2403.16804v1)** · 2024 multilingual temporal-expression recogniser.
- **["Multilingual Recognition of Temporal Expressions" — Muni.cz RASLAN 2020](https://nlp.fi.muni.cz/raslan/2020/paper2.pdf)** · multi-language temporal-expression survey.
- **[Geng et al., "JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models" — arXiv 2501.10868](https://arxiv.org/pdf/2501.10868)** · the benchmark for structured-output reliability under various constraint engines.

### Vendor / official documentation

- **[OpenAI · Introducing Structured Outputs in the API](https://openai.com/index/introducing-structured-outputs-in-the-api/)** + **[OpenAI · Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs)** · the deterministic-constrained-decoding feature (GPT-4o ≥ 2024-08-06); also the source for which JSON-Schema keywords strict mode rejects.
- **[Anthropic Claude · Tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)** + **[Anthropic · Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)** · `strict: true`, `input_examples`, temperature 0.2 guidance.
- **[LangChain · Structured output (v1.3 docs)](https://docs.langchain.com/oss/python/langchain/structured-output)** · the current `create_agent` + `response_format` surface.
- **[Mistral · JSON Schema mode](https://docs.mistral.ai/capabilities/structured-output/json_schema/)** · provider-side restrictions on accepted schema keywords.
- **[Pydantic · Datetimes docs](https://docs.pydantic.dev/latest/api/standard_library_types/)** · `date`, `datetime`, `PastDate`, `FutureDate`, custom `field_validator` semantics.
- **[dateparser · Official docs (readthedocs)](https://dateparser.readthedocs.io/)** · `JalaliCalendar`, `HijriCalendar`, `DATE_ORDER`, `PREFER_DAY_OF_MONTH`, `RELATIVE_BASE`, `STRICT_PARSING` — every setting used in this lesson's validator.
- **[python-dateutil docs](https://dateutil.readthedocs.io/)** · the English-only fallback. dateparser uses it internally.

### GitHub — issues and source code

- **[LangChain issue #29604 — *Pydantic model with datetime.date value using json_schema raises a 400 bad request*](https://github.com/langchain-ai/langchain/issues/29604)** · the bug at the heart of this lesson's "Solution A vs B" framing. **Closed as "not planned"** — read this if you ever doubt the workaround is necessary.
- **[LangChain issue #32687 — *structured output issue using llm.with_structured_output(pydantic class)*](https://github.com/langchain-ai/langchain/issues/32687)** · related (the LLM returns a string, not a dict, failing Pydantic validation).
- **[langchain_core source — `_rm_titles` in function_calling.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/utils/function_calling.py#L89)** · the exact ~30 lines responsible for stripping `title` but NOT `format`.
- **[noamgat/lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)** · the constrained-decoding library that enforces JSON Schema / regex on model outputs. Used as a context-free-grammar peer to Outlines / XGrammar.
- **[microsoft/Recognizers-Text](https://github.com/microsoft/Recognizers-Text)** · multilingual numbers / dates / time entity recogniser. The .NET-first Duckling alternative; powers LUIS and Bot Framework.
- **[facebook/duckling](https://github.com/facebook/duckling)** · Haskell-based natural-language date/time parser. Powerful, deployment-heavy.
- **[Stanford NLP · SUTime download](https://nlp.stanford.edu/software/sutime.html)** + **[Stanford Temporal Tagger project page](https://nlp.stanford.edu/projects/time.shtml)**.

### Engineering blogs / Medium articles

- **[James Tang · "Best Practices for Handling Dates in Structured Output in LLM" — Medium, 2024](https://medium.com/@jamestang/best-practices-for-handling-dates-in-structured-output-in-llm-2efc159e1854)** · the source of the ISO-8601 dogma plus the anti-pattern list (no localised strings / no relative output / no Unix timestamps / no native datetime objects). High signal-to-noise.
- **[aviadr1 · "How to Fix OpenAI Structured Outputs Breaking Your Pydantic Models" — Medium, 2025](https://medium.com/@aviadr1/how-to-fix-openai-structured-outputs-breaking-your-pydantic-models-bdcd896d43bd)** · the broader catalogue of Pydantic-emitted JSON-Schema patterns that OpenAI strict mode rejects (beyond `format`: `additionalProperties: {}`, recursive `$ref`, `anyOf` unions, numeric constraints on nullable). Author's full sanitizer code is in a [referenced gist](https://gist.github.com/aviadr1/2d1186625d67fba9c8f421d273bf7a53).
- **[Aidan Cooper · "A Guide to Structured Outputs Using Constrained Decoding"](https://www.aidancooper.co.uk/constrained-decoding/)** · ground-up explainer for Outlines / SGLang / GBNF / guidance / DSPy Assertions. The constrained-decoding landscape map.
- **[Zyte · "Parse natural language dates with Dateparser"](https://www.zyte.com/blog/parse-natural-language-dates-with-dateparser/)** · the head-to-head dateparser-vs-dateutil comparison that anchors the library table above.
- **[MachineLearningMastery · "The Complete Guide to Using Pydantic for Validating LLM Outputs"](https://machinelearningmastery.com/the-complete-guide-to-using-pydantic-for-validating-llm-outputs/)** · `field_validator` patterns + retry-on-error feedback loop.
- **[Instructor library · Retry mechanisms docs](https://python.useinstructor.com/learning/validation/retry_mechanisms/)** · canonical implementation of the validation-feedback-loop pattern (Pydantic `ValidationError` → re-prompt the LLM with the error). The library this lesson would graduate to in production.
- **[GoatReview · "Format ChatGPT results with PydanticOutputParser"](https://goatreview.com/format-chatgpt-results-with-pydantic-langchain-2/)** · LangChain-specific structured-output patterns; supplementary.

### Companion docs in this repo

- **[`docs/research/date-parsing-with-llms.md`](../../docs/research/date-parsing-with-llms.md)** · **Round 1 synthesis** — failure modes, prompt patterns, structural fixes, library comparison.
- **[`docs/research/langchain-date-field-bug.md`](../../docs/research/langchain-date-field-bug.md)** · the LangChain bug investigation — reproducer, root-cause code reference (`_rm_titles`), three solutions, decision tree.
- **[`docs/research/llm-date-solutions-deep-dive.md`](../../docs/research/llm-date-solutions-deep-dive.md)** · **Round 2 deep dive** — 10-approach taxonomy (NER, VLM, agentic tool use, validation libraries, specialised parsers), academic literature (NAACL 2024, DateLogicQA, Test of Time, TemporalBench), four production "recipes".

### A note on source mix

| Source type | What it's good for | Trust |
|---|---|---|
| **Peer-reviewed papers** (PMC, Nature, arXiv) | "How bad is the problem on real data?" Quantitative failure rates. | High |
| **Official vendor docs** (OpenAI, Anthropic, LangChain, Pydantic) | "What does the API actually accept *today*?" | High but volatile — re-check quarterly |
| **GitHub issues + source** | "Is this bug fixed? What's the actual code path?" | Highest for current state |
| **Engineering blogs / Medium** | "What pattern do practitioners reach for?" Community consensus. | Medium — verify with code |

The research used **all four** because none alone is enough. Vendor docs tell you the API surface; papers tell you it doesn't always behave as documented in the wild; GitHub tells you what's broken right now; blogs tell you how everyone else routes around it.

## Pairs with

- **[Lesson 19 · Guardrails](../19_guardrails/README.md)** — the same pattern of layered defences applied to other failure modes (PII, cost, schema).
- **[Lesson 20 · Chunking & parsing](../20_chunking_and_parsing/README.md)** — the upstream sibling: getting clean text into your LLM.
- **[`shared/llm/schema_sanitizer.py`](../../shared/llm/schema_sanitizer.py)** — the reusable fix powering Solution B.

## Next →

That's the curriculum side-quests. Capstones in [`projects/`](../../projects/).
