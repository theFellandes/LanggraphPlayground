# LLM Date Problems — Deep Dive (Round 2)

**Investigation date:** 2026-05-20
**Companion to:** [`date-parsing-with-llms.md`](date-parsing-with-llms.md) (Round 1) and [`langchain-date-field-bug.md`](langchain-date-field-bug.md) (bug analysis)
**Question (re-asked):** *How do other people actually solve LLM date problems? What's the full landscape of approaches — guardrails, prompts, libraries, VLMs, fine-tuning? Is there a "Docling for dates"?*

---

## TL;DR

1. **There are 10 distinct solution approaches**, not 4. Round 1 covered prompts + Pydantic + dateparser + sanitizer. This round adds: NER models, VLM-based extraction, agentic tool use, two-stage parser pipelines (LangChain `DatetimeOutputParser`), validation-feedback libraries (Instructor / Marvin / Mirascope), and specialized natural-language parsers (`pendulum`, `arrow`, `chrono-python`, `parsedatetime`, Rust `dtparse`).
2. **There is no "Docling for dates" specifically** — but Docling's `ExtractionVlmPipeline` and IBM's **Granite-Docling-258M** are the closest thing: VLM-direct document understanding that skips OCR errors entirely. For dedicated date-NER, there's `dslim/bert-base-NER` and SpaCy's `DATE` label — these are the strongest *traditional* extractors, still beat naive LLM prompting in benchmarks.
3. **The academic literature now agrees** (NAACL 2024, DateLogicQA, Test of Time): LLMs *"approximate correct calculations but often stumble in final steps"* — the off-by-one and confidently-wrong failure mode is structural, not a prompting issue. Defence-in-depth is mandatory for production.

---

## The 10 solution approaches — full taxonomy

Each row tells you: **what the approach does**, **when it shines**, **when to avoid it**, and **a concrete tool** to start with.

| # | Approach | What it does | Shines when | Avoid when | Tool |
|---|---|---|---|---|---|
| 1 | **Anchored prompt + ISO 8601 + few-shot** | Tell the LLM "today is X, output `YYYY-MM-DD`, here are 3 examples, return null if absent" | Always. Cheapest lever. Foundation for everything else | Standalone — never enough on its own | Plain string in your prompt |
| 2 | **Pydantic schema with `str` + regex `pattern`** | Type the field as `str`, attach `pattern=r"^\d{4}-\d{2}-\d{2}$"`, validate downstream | You want a universal solution that survives any provider's strict mode | You need a real `datetime.date` in downstream code | `pydantic.Field(pattern=...)` |
| 3 | **Pydantic `date` + schema sanitizer** | Keep native `date` type; strip the offending `"format": "date"` key before sending to provider | You want native Python types AND structured output | You're worried about an extra middleware step | [`shared/llm/schema_sanitizer.py`](../../shared/llm/schema_sanitizer.py) in this repo |
| 4 | **Provider-side constrained decoding** | OpenAI strict mode / Outlines / lm-format-enforcer / XGrammar make the *decoder* unable to emit invalid JSON tokens | You're on a frontier provider, or a self-hosted model where you control the inference engine | You're on a provider that doesn't expose constrained decoding | OpenAI `strict=True`, `outlines`, `lm-format-enforcer` |
| 5 | **Two-stage extract-then-parse** | LLM returns plain text; a separate parser converts to date | You want maximum loose coupling between "LLM speaks" and "we extract a typed date" | You want one round-trip | LangChain `DatetimeOutputParser` (in `langchain_classic.output_parsers.datetime`) |
| 6 | **Hybrid LLM + deterministic library** | LLM produces a string; `dateparser` re-parses it strictly; Pydantic validator runs domain sanity | The defence-in-depth default. **This is lesson 21's recommended pattern** | Single-locale, English-only, low-volume — overkill | `dateparser` + Pydantic `field_validator` |
| 7 | **Agentic tool use** | Give the LLM a `today()` tool. Force it to *retrieve* the anchor date instead of guessing. For relative dates, give it a `compute_date(reference, offset)` tool too | Multi-turn agents that handle "schedule me 3 weeks from today" | Single-shot extraction — overkill, adds a round-trip | `@tool` + `create_agent` (lesson 10) |
| 8 | **VLM-direct extraction** | Vision-language model reads the page image and emits the date — skip OCR entirely | Scanned PDFs, handwritten forms, multi-column invoices where OCR cascades errors | Plain text input — pointless extra cost | **Docling `ExtractionVlmPipeline`**, **IBM Granite-Docling-258M**, GPT-4o Vision |
| 9 | **Validation-feedback loop** | When Pydantic raises `ValidationError`, re-prompt the LLM with the specific error message and let it try again (1-3 attempts) | Production reliability + cost is acceptable | Latency-critical paths | **Instructor**, **Marvin** (PrefectHQ), **Mirascope** |
| 10 | **Fine-tuned domain NER model** | A BERT-based NER model trained to find `DATE` entities. Pre-LLM era, but still SOTA for *just-extract-the-date* tasks | High-volume, low-margin, single-language pipelines | You need the model to also normalize, reason, or pick the *right* date out of several | `dslim/bert-base-NER`, SpaCy `en_core_web_lg`, custom fine-tune |

The point of the taxonomy: **most production teams stack 3-5 of these**, not one. The decision tree at the bottom maps which combinations make sense.

---

## Code anchors — one canonical example per approach

Tiny, focused, runnable. Imports first, smallest call second. Sized
to be copy-pasted into a `python -c '...'` shell.

### 1 · Anchored prompt + ISO 8601 + few-shot

```python
from datetime import date
prompt = f"""Today is {date.today().isoformat()}.

Extract the date of birth. Output ISO 8601 (YYYY-MM-DD) only.
If not present, return "null". Do NOT invent.

Examples:
  "Born April 5, 1987" → 1987-04-05
  "DOB 04/05/87 (MM/DD/YY)" → 1987-04-05
  "She is 35 years old" → null

Text: {{text}}"""
```

### 2 · Pydantic `str` + regex `pattern` (universal)

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    birth_date: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="ISO 8601 (YYYY-MM-DD), null if absent.",
    )
```

### 3 · Pydantic `date` + schema sanitizer (native typing)

```python
from datetime import date
from pydantic import BaseModel
from shared.llm import with_structured_output_safe   # ships in this repo

class Person(BaseModel):
    birth_date: date | None

llm = with_structured_output_safe(Person)
person = llm.invoke("Born April 5, 1987.")
type(person.birth_date)        # → <class 'datetime.date'>
```

### 4 · Provider-side constrained decoding (OpenAI strict / Outlines)

```python
# OpenAI strict mode — schema violation becomes literally impossible
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o").with_structured_output(Person, strict=True)

# Self-hosted alternative — Outlines + a regex FSM
import outlines
generator = outlines.generate.regex(
    model, r"\d{4}-\d{2}-\d{2}"
)
generator("The birth date is")   # → "1987-04-05" (decoder cannot drift)
```

### 5 · Two-stage extract-then-parse (`DatetimeOutputParser`)

```python
from langchain_classic.output_parsers.datetime import DatetimeOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = DatetimeOutputParser(format="%Y-%m-%d")
prompt = ChatPromptTemplate.from_template(
    "Extract the date.\n{format_instructions}\nText: {text}"
).partial(format_instructions=parser.get_format_instructions())
chain = prompt | llm | parser
chain.invoke({"text": "Born April 5, 1987"})   # → datetime.datetime(1987, 4, 5, 0, 0)
```

### 6 · Hybrid LLM + deterministic library (the lesson-21 default)

```python
from datetime import date
from pydantic import BaseModel, Field, field_validator
import dateparser
TODAY = date.today()

class Person(BaseModel):
    birth_date: str | None = Field(default=None)

    @field_validator("birth_date")
    @classmethod
    def _parse(cls, v):
        if v is None: return None
        d = dateparser.parse(v, settings={
            "STRICT_PARSING": True, "RELATIVE_BASE": TODAY,
        })
        if d is None: raise ValueError(f"unparseable: {v!r}")
        if d.year < 1900 or d.date() > TODAY:
            raise ValueError(f"implausible: {d.date()}")
        return d.date().isoformat()
```

### 7 · Agentic tool use (give the LLM a `today()` tool)

```python
from datetime import date
from langchain_core.tools import tool
from langchain.agents import create_agent

@tool
def today_iso() -> str:
    """Return today's date in ISO 8601."""
    return date.today().isoformat()

agent = create_agent(model=llm, tools=[today_iso])
agent.invoke({"messages": [{"role": "user",
    "content": "What date was 5 days ago?"}]})
# Model calls today_iso() instead of guessing — no off-by-one errors.
```

### 8 · VLM-direct extraction (Docling for images / scans)

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()        # uses the default VLM pipeline
result = converter.convert("invoice.pdf")
# For form-field-style extraction with templates:
# from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline
# pipeline = ExtractionVlmPipeline(template={"invoice_date": "...", ...})

# A separate, lighter VLM (Granite-Docling-258M) — same idea, ~258M params.
```

### 9 · Validation-feedback loop (Instructor / Marvin / Mirascope)

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, field_validator

client = instructor.from_openai(OpenAI())

class Person(BaseModel):
    birth_date: str
    @field_validator("birth_date")
    @classmethod
    def _iso(cls, v):
        if not v[:4].isdigit(): raise ValueError("must start with year")
        return v

# On ValidationError, instructor re-prompts with the error message — up to N tries.
person = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=Person,
    max_retries=3,
    messages=[{"role": "user", "content": "Born April 5, 1987."}],
)
```

### 10 · Fine-tuned NER (Hugging Face / SpaCy)

```python
# Option A — Hugging Face BERT-NER
from transformers import pipeline
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
ner("Born April 5, 1987 in Istanbul.")   # → [{'entity_group': 'MISC', 'word': 'April 5, 1987', ...}]

# Option B — SpaCy (has built-in DATE label)
import spacy
nlp = spacy.load("en_core_web_lg")
[(e.text, e.label_) for e in nlp("Born April 5, 1987").ents]
# → [('April 5, 1987', 'DATE')]
```

---

## What the academic literature actually says (2024-2026)

The Round 1 doc cited one peer-reviewed paper (PMC11634005, clinical). Round 2 widens to four LLM-temporal-reasoning papers — all converging on the same diagnosis:

| Paper | Key finding | Implication |
|---|---|---|
| **"Are Large Language Models Temporally Grounded?"** (NAACL 2024) [1] | LLMs (GPT-3.5, GPT-4) "frequently failed to recognise when information became outdated"; "confidently provided incorrect temporal information, showing no uncertainty markers" | LLMs *will* confidently emit wrong dates. They don't know they don't know. Validate everything. |
| **"Test of Time"** (arXiv 2406.09170) [2] | LLMs "approximate correct calculations but often stumble in final steps, highlighting gaps in their ability to execute complex arithmetic with precision" | Off-by-one errors on relative dates (5 days ago / next Tuesday / 3 months later) are structural. Use deterministic arithmetic. |
| **DateLogicQA** (arXiv 2412.13377) [3] | 190 curated questions across past/present/future × commonsense/factual/conceptual/numerical reasoning. Tests temporal bias. | Provides the eval set to actually measure your pipeline's date-reasoning accuracy. Use it. |
| **TemporalBench** (arXiv 2602.13272) [4] | Benchmark for LLM agents on event-informed time-series tasks | Agents struggle with multi-step temporal reasoning even more than single-shot extraction. |
| **Cui et al. — clinical date extraction** (PMC11634005) [5] | GPT-4: 19 wrong dates + 3 hallucinations; LlaMA-2-70b: 23 wrong dates + 27 hallucinations on the same notes | The Round 1 anchor — production failure rates on real clinical data are 15-25%. |

**Cross-paper consensus:** LLMs' weakness on dates is **not** a prompting problem you can engineer your way out of. It's a structural limitation that scales poorly with model size. Defence in depth is mandatory.

---

## Is *birth-date* extraction specifically hard? — the honest nuance

The papers above are about date / temporal reasoning *in general*. They
do not measure birth-date extraction as a named task. So when somebody
asks "is birth-date parsing actually hard for LLMs, and is there
research on it?" the honest answer is split:

### Where birth dates are **easy** for LLMs

When the input is *clean text* with an explicit DOB, modern LLMs do well:

| Source | Finding |
|---|---|
| **LLM-Anonymizer** (NEJM AI, 2024) [8] | Clinical-document deidentification study; **"Date of birth was extracted in nearly all letters for almost all models"** — DOB was among the most-reliably-extracted demographic fields. |
| **A Study of LLMs for Patient Information Extraction** (arXiv 2509.04753) [9] | Multi-task patient demographic extraction (including DOB). DOB scored among the higher-accuracy fields. |
| **CaseReportBench** (PMC12477612) [10] | Dense information extraction benchmark for clinical case reports; demographics including DOB are routinely-extracted fields. |
| **KYC industry reports** (Veryfi, Parsli, AlgoDocs, IN-D, 2025–2026) [11] | 97–99% accuracy on DOB extraction from clean ID documents (passport / driver's licence / national ID). |

When the input is *clean, explicit, and structurally regular* — "DOB:
1987-04-05" or "Born April 5, 1987" — LLMs handle DOB about as well
as they handle any other named entity. **This is the case where the
field's reputation for being "hard" is overstated.**

### Where birth dates are **hard** for LLMs

The published evidence on failure shifts to specific operating conditions:

| Source | Failure mode quantified |
|---|---|
| **"Seeing is Believing? Mitigating OCR Hallucinations in Multimodal LLMs"** (arXiv 2506.20168) [12] | When the document is a *scan or photo* (driver's licence, passport, handwritten form), multimodal LLMs hallucinate plausible-but-wrong DOBs. Documented OCR-LLM intersection failure mode. |
| **Test of Time** (arXiv 2406.09170) [2] | When the DOB is *relative* ("born 35 years ago", "three days after my brother in 1987"), LLMs do arithmetic poorly. Off-by-one errors are routine. |
| **Are LLMs Temporally Grounded?** (NAACL 2024) [1] | When the DOB involves *cross-calendar reasoning* (Jalali ↔ Gregorian, Japanese era ↔ Gregorian), models confidently emit wrong dates. |
| **Cui et al.** (PMC11634005) [5] | Adjacent task — date extraction from clinical notes when the document is *unstructured prose* with ambiguous date formats and multiple plausible candidates. 15–25% wrong-date rates at GPT-4 quality. |
| **Schema decomposition** (practitioner-observed) | When the tool schema splits the DOB into *separate* `birth_day` / `birth_month` / `birth_year` integer fields and the prompt asks the LLM to do in-prompt locale conversion (e.g. Turkish *"on bir"* → `11`, *"kasım"* → `11`), the model fills the components independently. Each numeric field is an independent chance to drift. Real-world failure: user says *"yirmi beş on iki yetmiş dokuz"* (25 / 12 / 79), the model emits `{birth_day: 25, birth_month: 10, birth_year: 1979}` — a single-component hallucination that the other two correct fields make look plausible. **Fix:** one coherent ISO string field (so the components constrain each other) plus deterministic server-side parsing; never ask the LLM to do digit-by-digit conversion. Documented in [lesson 21 · "Why one string field, not decomposed ints"](../../lessons/21_date_parsing/README.md#why-one-string-field-not-decomposed-day--month--year-ints). |

### Why both can be true

DOB extraction is the conjunction of two sub-problems:

```
DOB extraction = locate the date span  ×  normalise to ISO 8601
                 (NER — generally easy)   (parsing + reasoning — varies)
```

- **Locating** a DOB in clean text is similar to standard NER (DATE entity recognition). Pre-LLM systems (SpaCy, BERT-NER) already did this at >90% F1 on CoNLL-2003 a decade ago, and modern LLMs match or exceed that on clean input [LLM-Anonymizer evidence].
- **Normalising** to a canonical form is where it gets hard: format ambiguity (04/05/87), relative references, non-Gregorian calendars, OCR errors, partial dates, hallucinated valid-looking dates. This is the part the general temporal-reasoning papers (Test of Time, DateLogicQA, NAACL 2024) document.

**So the practitioner-level claim "LLMs have trouble parsing birth
dates" is true** — but it's true for *specific* failure modes, not as a
blanket statement. The four-layer pipeline in [lesson 21](../../lessons/21_date_parsing/README.md)
specifically targets the normalisation half, which is where the difficulty
actually lives.

### Implication for the four production "recipes"

This nuance reshapes which recipe to pick:

| Input shape | Difficulty | Right recipe |
|---|---|---|
| Plain text, explicit ("DOB: April 5, 1987") | Low | Recipe A (minimal) is enough |
| Plain text, ambiguous ("04/05/87 (US format)") | Medium | Recipe A + locale-aware dateparser settings |
| Plain text, relative ("born 35 years ago") | High — *arithmetic* fails | Recipe A + the agentic tool from [lesson 23](../../lessons/23_date_localization/README.md) |
| Scanned ID document / handwritten form | High — *OCR + multimodal hallucination* | Recipe C (Docling/Granite-Docling VLM-direct) + Recipe A on the output |
| Multi-calendar (Jalali/Hijri/Japanese era) | High — *normalisation* fails | dateparser with `JalaliCalendar` / `HijriCalendar` + lesson-23 localization helpers |

Treat the four-layer pipeline as **insurance**, not as proof that DOB is
always hard. For the easy-input case it never trips; for the hard cases
it earns its keep.

---

## The library landscape — full taxonomy now

Round 1 covered `dateparser`, `dateutil`, HeidelTime, SUTime, Duckling, Microsoft Recognizers-Text. Round 2 adds the **specialised natural-language parsers** (often overlooked) and the **VLM-direct alternatives**.

### Python natural-language date parsers (the long tail)

| Library | Strength | Weakness |
|---|---|---|
| **`dateparser`** | 200+ locales, Jalali / Hijri / Buddhist calendars, relative dates, ambiguity settings | ~8× slower than dateutil; some edge cases on ambiguous DD/MM vs MM/DD |
| **`dateutil`** | Fast, robust English, the bedrock — dateparser uses it internally | English-only, no relative-date support |
| **`pendulum`** | Drop-in `datetime` replacement; clean timezone semantics; modern API | Less natural-language coverage than dateparser |
| **`arrow`** | Popular, fluent API; built-in humanize | "Behaviour can be erratic and unpredictable" (per practitioners); avoid for production parsing |
| **`chrono-python`** (port of chrono.js) | Light natural-language parsing | Less coverage than dateparser; smaller community |
| **`parsedatetime`** | Specifically built for natural-language ("next Tuesday at 3pm") | Older codebase; English-focused |
| **`dtparse`** (Rust + PyO3) | **10–15× faster than `datetime.strptime`** for bulk parsing | Limited natural-language; mostly format-string parsing |
| **HeidelTime** (Java) | Academic gold standard, very high precision, multilingual | JVM dependency; subprocess overhead |
| **SUTime** (Java, Stanford) | English temporal tagger; tight CoreNLP integration | Lower precision than HeidelTime |
| **Facebook Duckling** (Haskell) | Excellent multi-language | Server deployment; deployment friction |
| **Microsoft Recognizers-Text** (.NET / JS / Python) | Multi-language, offline, no server | Less Pythonic than dateparser |

**Practical defaults:**

- One-off Python apps: `dateparser` for breadth, `dateutil` for speed
- Mixed-locale production: `dateparser` (settings per request) or Recognizers-Text
- Bulk high-throughput pipelines: `dtparse` (Rust) for format-known input
- Multilingual research-grade: HeidelTime (subprocess)

### Vision-language models — the "Docling for dates" angle

The user asked specifically about a Docling-like specialized library for dates. The honest answer: **none of the popular projects ship a "date-specific" library**, but the **VLM-direct route** is the closest thing in spirit.

| Tool | What it does | Why it matters for dates |
|---|---|---|
| **Docling — `ExtractionVlmPipeline`** [6] | Template-driven extraction of form fields / invoice fields directly from page images | Skips the OCR → broken digits → wrong date cascade. The VLM sees `04/05/87` in its visual context and reads it correctly. |
| **IBM Granite-Docling-258M** | Compact (258M params) VLM specifically for document understanding; preserves layout, tables, equations | Same idea, much cheaper than GPT-4o vision for high-volume document workloads |
| **GPT-4o / Claude 4.x Vision** | General-purpose VLMs — accept image + extract structured data | Works zero-shot but expensive at scale |
| **LayoutLMv3 / Donut** | Layout-aware document understanding transformers | Pre-VLM-era; still strong on form-field extraction. |
| **olmOCR** (arXiv 2502.18443) [7] | Open-weights VLM-driven OCR-replacement aimed at trillion-token corpora | Good when you need OCR+understanding in one pass |

**The verdict on "Docling for dates":** if your input is **images / scanned PDFs**, VLMs (Docling + Granite, GPT-4o Vision) are unambiguously better than OCR+LLM. If your input is **already plain text**, VLMs are pointless overhead.

### The NER route (deeply under-used)

A fine-tuned BERT-NER model can extract DATE entities with >90% F1 on CoNLL-2003 benchmarks. Cost per inference: tiny compared to an LLM. Latency: ~10ms on CPU. Reliability: deterministic outputs.

| Tool | Notes |
|---|---|
| **`dslim/bert-base-NER`** on Hugging Face | Fine-tuned on CoNLL-2003; recognises `MISC` (includes dates indirectly) |
| **`dbmdz/bert-large-cased-finetuned-conll03-english`** | Larger variant; higher F1 |
| **SpaCy `en_core_web_lg`** | Built-in `DATE` entity label; one-liner `nlp(text).ents` |
| **`presidio-analyzer`** (Microsoft) | PII detection layer that includes `DATE_TIME` recogniser; can be wrapped as a guardrail |
| **Custom fine-tune** | If you have 1000+ in-domain examples (medical notes, legal contracts), fine-tuning a BERT-NER on `DATE` typically beats prompting an LLM on the same task |

**When NER beats LLM:** high-volume, single-task, single-language, format-stable pipelines. Think: insurance claims, lab reports, recurring form data.

**When LLM beats NER:** multi-format inputs, multi-language, contextual disambiguation ("which date is the patient's *birth* date among these three?"), free-text reasoning.

### Validation-feedback libraries — the "if it fails, ask again with the error" pattern

Round 1 mentioned Instructor briefly. Round 2 catalogues the three serious contenders:

| Library | Author | What's distinct |
|---|---|---|
| **Instructor** | 567-labs (Jason Liu) | The dominant choice. ~3M monthly downloads, 11k GitHub stars. Patches the OpenAI / Anthropic / etc. client. Retry-with-error loop is the default. Works with every provider Pydantic AI does. |
| **Marvin** | PrefectHQ | Newer, Pydantic-AI-based. "AI Model" subclasses `BaseModel` and parses any string into it. Cleanest syntax of the three. Multi-provider via Pydantic AI. |
| **Mirascope** | Mirascope team | "No magic" philosophy. Class-per-provider (not a client patch). Most Pythonic. Smaller community than Instructor. |

**All three implement the same core pattern:**

```
1. LLM produces output
2. Pydantic validates
3. If ValidationError → re-prompt LLM with the specific error
4. Retry up to N times (configurable)
5. Surface a final exception if still failing
```

This is the *operational* fix that turns the 4-layer pipeline into a reliable production component. For dates specifically: when layer-4 raises `"layer-4 reject: 1850-01-01 is before 1900"`, that exact message goes back to the LLM, which usually corrects to a sensible year on retry.

---

## How everyone is actually solving this (the field guide)

Aggregating the practitioner sources (Medium, blog posts, conference talks, GitHub discussions), four production "recipes" dominate:

### Recipe A — "The minimal" (most teams start here)

```
Anchored prompt with "today is YYYY-MM-DD"
        +
Pydantic schema with birth_date: str + regex pattern
        +
Manual dateparser parse in a custom validator
```

Use when: you have <1000 documents/day, the input is text, the locale is known. **This is lesson 21's Solution A.**

### Recipe B — "The robust" (when minimal isn't enough)

```
Recipe A
        +
Provider-side strict structured output (OpenAI strict / Anthropic structured)
        +
Instructor (or equivalent) for retry-with-error
        +
LangSmith / equivalent for trace replay on failures
```

Use when: customer-facing application, errors are visible to end users, you can afford 1-3 model calls per extraction.

### Recipe C — "The document-AI" (forms, scans, PDFs)

```
Docling ExtractionVlmPipeline (or Granite-Docling for cheap)
        +
A post-VLM dateparser pass (still needed for normalisation)
        +
Pydantic validators for domain sanity (birth date ≥ 1900, etc.)
```

Use when: input is images / scanned PDFs / form documents. The VLM is the OCR + extraction combined; downstream validation still applies.

### Recipe D — "The high-volume hybrid" (millions of records)

```
SpaCy or BERT-NER first pass to extract date spans
        +
dtparse (Rust) for fast normalisation of well-formed input
        +
LLM fallback only when the NER returns 0 or >1 candidates
        +
Pydantic validators on the final date
```

Use when: you're processing millions of records and the LLM-call cost is the bottleneck. The NER handles 80% deterministically; the LLM only fires on the hard 20%.

---

## The Guardrails AI gap (and how to fill it)

**Guardrails AI** ships a hub of validators (`regex_match`, `valid_json`, `pii`, `toxicity`, `competitor_check`, etc.). It does **NOT** ship a `date_format` validator as of May 2026.

The community workaround: compose `regex_match` with an ISO 8601 pattern.

```python
from guardrails import Guard
from guardrails.validators import RegexMatch

guard = Guard().use(
    RegexMatch(
        regex=r"^\d{4}-\d{2}-\d{2}$",
        on_fail="reask",          # ask the LLM to fix it
    )
)
```

This is **strictly worse** than the dateparser-based layer-3 + Pydantic layer-4 combination in this repo, because:

- regex doesn't catch `Feb 31, 2020-01-01` (passes regex, isn't a real date)
- regex doesn't catch implausible birth dates (1850-01-01, year 2099)
- regex on input doesn't help with format normalisation (DD/MM → YYYY-MM-DD)

So: **for dates specifically, the four-layer pipeline beats Guardrails AI**. Use Guardrails AI for other concerns (toxicity, PII, competitor-mention) and keep the date layers separate.

---

## Decision tree — picking the right combination

```
What's your input?
│
├── Image / scanned PDF
│       └── VLM (Docling ExtractionVlmPipeline or GPT-4o Vision)
│           + Pydantic validator + dateparser normalisation
│
├── Plain text, low volume (<1k/day)
│       └── Recipe A (anchored prompt + str + dateparser)
│
├── Plain text, high volume (>100k/day)
│       └── Recipe D (NER first, LLM fallback only on ambiguity)
│
└── Plain text, customer-facing, high reliability
        └── Recipe B (Recipe A + Instructor retry + strict structured outputs)
```

Within each branch, additional questions:

- **Multilingual?** → `dateparser` (locales) or HeidelTime (precision)
- **Persian / Hijri / Japanese era calendars?** → `dateparser` with `JalaliCalendar` / `HijriCalendar`
- **Native `datetime.date` required downstream?** → Solution B (sanitizer) in this repo
- **Multi-turn agent doing relative-date arithmetic?** → Add a `today()` tool, don't let the LLM compute it
- **Need explainability ("why this date?")** → Two-stage extract-then-parse (LangChain `DatetimeOutputParser`) so the string is preserved alongside the parsed value

---

## What's still open (genuine research gaps in May 2026)

1. **No specialised "LLM-era date library."** Closest thing is Docling-VLM for images; for text, the field hasn't produced an analogue. *Opportunity.*
2. **Date-NER fine-tunes aren't widely shared.** Anyone training a BERT-NER on multilingual medical/legal/finance date spans would ship something genuinely useful.
3. **Standardised eval sets are immature.** DateLogicQA (190 Qs) is the best, still small. The clinical study has ~100 notes. We don't have a HumanEval for date extraction.
4. **Multi-document birth-date reconciliation.** Two source pages say different things — no library handles this; you write the business logic yourself.
5. **OCR-degraded dates.** VLMs handle this best today but the eval landscape is sparse.

---

## References (Round 1 + Round 2 combined)

### Peer-reviewed papers

- [1] [Are Large Language Models Temporally Grounded? — NAACL 2024](https://aclanthology.org/2024.naacl-long.391.pdf) · [arXiv 2311.08398](https://arxiv.org/pdf/2311.08398)
- [2] [Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning — arXiv 2406.09170](https://arxiv.org/html/2406.09170v1)
- [3] [DateLogicQA: Benchmarking Temporal Biases in Large Language Models — arXiv 2412.13377](https://arxiv.org/pdf/2412.13377)
- [4] [TemporalBench: A Benchmark for Evaluating LLM-Based Agents on Contextual and Event-Informed Time Series Tasks — arXiv 2602.13272](https://arxiv.org/html/2602.13272v1)
- [5] [Cui et al., *Evaluating LLMs in Extracting Cognitive Exam Dates and Scores* — PMC11634005](https://pmc.ncbi.nlm.nih.gov/articles/PMC11634005/)
- [6] [Docling Vision Language Models — DeepWiki](https://deepwiki.com/docling-project/docling/4.3-vision-language-models)
- [7] [olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models — arXiv 2502.18443](https://arxiv.org/pdf/2502.18443)
- [8] [*The LLM-Anonymizer: Deidentifying Medical Documents with Local, Privacy-Preserving LLMs* — NEJM AI, 2024](https://ai.nejm.org/doi/full/10.1056/AIdbp2400537) — peer-reviewed clinical study; **direct evidence on DOB extraction**: "Date of birth was extracted in nearly all letters for almost all models"
- [9] [*A Study of LLMs for Patient Information Extraction* — arXiv 2509.04753](https://arxiv.org/pdf/2509.04753) — multi-task patient demographic extraction including DOB
- [10] [*CaseReportBench: An LLM Benchmark for Dense Information Extraction in Clinical Case Reports* — PMC12477612](https://pmc.ncbi.nlm.nih.gov/articles/PMC12477612/) — clinical extraction benchmark with patient demographics
- [11] KYC industry reports on DOB extraction accuracy (97–99% on clean ID docs): [Veryfi](https://www.veryfi.com/document-processing-kyc-verification/), [Parsli 2026 guide](https://parsli.co/guides/kyc-document-extraction-automation), [AlgoDocs](https://algodocs.com/how-to-use-ai-to-improve-kyc-data-extraction/), [IN-D / UiPath Marketplace](https://marketplace.uipath.com/listings/in-d-kyc-id-document-classification-extraction-and-validation)
- [12] [*Seeing is Believing? Mitigating OCR Hallucinations in Multimodal LLMs* — arXiv 2506.20168](https://arxiv.org/html/2506.20168v2) — direct evidence on DOB-from-scans failure mode
- [Chang & Manning, *SUTIME* — Stanford LREC 2012](https://nlp.stanford.edu/pubs/lrec2012-sutime.pdf)
- [Hammami & Gelbukh, *I still have Time(s)* (HeidelTime for German) — arXiv 2204.08848](https://arxiv.org/pdf/2204.08848)
- [Macedo et al., *TEI2GO* — arXiv 2403.16804](https://arxiv.org/html/2403.16804v1)
- [Multilingual Recognition of Temporal Expressions — RASLAN 2020](https://nlp.fi.muni.cz/raslan/2020/paper2.pdf)
- [Geng et al., *JSONSchemaBench* — arXiv 2501.10868](https://arxiv.org/pdf/2501.10868)

### Vendor docs

- [OpenAI · Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs)
- [Anthropic Claude · Tool use + Structured outputs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [LangChain · Structured output (v1.3)](https://docs.langchain.com/oss/python/langchain/structured-output) + [`DatetimeOutputParser` API](https://reference.langchain.com/python/langchain-classic/output_parsers/datetime/DatetimeOutputParser)
- [Mistral · JSON schema mode](https://docs.mistral.ai/capabilities/structured-output/json_schema/)
- [Pydantic · Datetimes](https://docs.pydantic.dev/latest/api/standard_library_types/)
- [dateparser · readthedocs](https://dateparser.readthedocs.io/)
- [Pendulum FAQ](https://pendulum.eustace.io/faq/)

### GitHub — issues, source, repos

- [LangChain issue #29604 — date field bug](https://github.com/langchain-ai/langchain/issues/29604) (**closed: not planned**)
- [LangChain issue #22740 — PydanticOutputParser unions don't convert dates](https://github.com/langchain-ai/langchain/issues/22740)
- [LangChain issue #32687 — structured-output validation](https://github.com/langchain-ai/langchain/issues/32687)
- [`langchain_core/utils/function_calling.py` — `_rm_titles` source](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/utils/function_calling.py#L89)
- [567-labs/instructor](https://github.com/567-labs/instructor)
- [PrefectHQ/marvin](https://github.com/PrefectHQ/marvin)
- [Mirascope](https://mirascope.com/) · class-per-provider Pydantic-AI library
- [guardrails-ai/guardrails](https://github.com/guardrails-ai/guardrails) + [regex_match validator](https://guardrailsai.com/hub/validator/guardrails/regex_match)
- [noamgat/lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)
- [microsoft/Recognizers-Text](https://github.com/microsoft/Recognizers-Text)
- [facebook/duckling](https://github.com/facebook/duckling)
- [wanasit/chrono-python](https://github.com/wanasit/chrono-python)
- [gukoff/dtparse](https://github.com/gukoff/dtparse) (Rust + PyO3)
- [dslim/bert-base-NER on Hugging Face](https://huggingface.co/dslim/bert-base-NER)
- [dbmdz/bert-large-cased-finetuned-conll03-english](https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english)

### Engineering blogs / Medium articles

- [James Tang · *Best Practices for Handling Dates in Structured Output in LLM*](https://medium.com/@jamestang/best-practices-for-handling-dates-in-structured-output-in-llm-2efc159e1854)
- [aviadr1 · *How to Fix OpenAI Structured Outputs Breaking Your Pydantic Models*](https://medium.com/@aviadr1/how-to-fix-openai-structured-outputs-breaking-your-pydantic-models-bdcd896d43bd)
- [Aidan Cooper · *A Guide to Structured Outputs Using Constrained Decoding*](https://www.aidancooper.co.uk/constrained-decoding/)
- [Zyte · *Parse natural language dates with Dateparser*](https://www.zyte.com/blog/parse-natural-language-dates-with-dateparser/)
- [Riccardo Tartaglia · *Teaching Your LLM to Tell Time*](https://medium.com/@riccardo.tartaglia/teaching-your-llm-to-tell-time-a-practical-guide-to-llm-tool-integration-a52436f68a58) · agentic `today()` tool pattern
- [Paul Simmering · *The best library for structured LLM output*](https://simmering.dev/blog/structured_output/) · Instructor vs Marvin vs Outlines comparison
- [LearnByBuilding · *Marvin vs Guardrails vs Instructor*](https://learnbybuilding.ai/vs/marvin-ai-vs-guardrails-vs-instructor)
- [MachineLearningMastery · *Pydantic for Validating LLM Outputs*](https://machinelearningmastery.com/the-complete-guide-to-using-pydantic-for-validating-llm-outputs/)
- [Instructor · Retry mechanisms docs](https://python.useinstructor.com/learning/validation/retry_mechanisms/)
