# Lesson 23 · Date computation & localized output

The **output** side of the date problem. Lesson 21 was about extracting a
date *from* text. This lesson is about two things that come after:

1. **Date computation** — "what's today?", "3 weeks from now?", "next
   Tuesday?". LLMs are confidently wrong at this; you need deterministic
   tools.
2. **Localized output** — your user asks in Turkish, you reply
   *"23 Mayıs 2026"*, not *"May 23, 2026"*. Same date, different locale,
   different calendar in some cases.

## What you'll learn

- Why LLMs **cannot be trusted to compute dates** (training cutoff + bad
  arithmetic on relative dates) and the **`today_iso()` / `compute_date()`
  tool pattern** that fixes it
- How to render a single ISO date across **10+ locales** with `babel`:
  *"May 23, 2026"* / *"23 Mayıs 2026"* / *"23. Mai 2026"* / *"23 mai 2026"* /
  *"23 مايو 2026"* / *"2026年5月23日"*
- The full CLDR format-name spectrum (`short` / `medium` / `long` /
  `full`) and how to write **custom CLDR patterns** like
  `"EEEE, d MMMM y"` → *"Cumartesi, 23 Mayıs 2026"*
- **Calendar conversions** — Persian Jalali via `jdatetime`, Islamic Hijri
  via `hijridate`, Japanese era via `babel`
- The architectural rule: **store ISO 8601, localize at the edge**

## Why it matters

Two different concerns, often conflated:

| Concern | Layer | Tool | Lesson |
|---|---|---|---|
| Extract `2026-05-23` from messy free text | Validation | Pydantic + `dateparser` | 21 |
| Compute `today + 3 weeks` reliably | Orchestration (tools) | `@tool today_iso()` + `@tool add_days()` | **23** |
| Show `2026-05-23` as *"23 Mayıs 2026"* to a Turkish user | Interface / presentation | `babel.format_date(d, locale="tr_TR")` | **23** |

Conflating these is the most common date-handling bug in LLM apps — e.g.
asking the model to do the locale-format-conversion itself (it gets it
wrong) or storing dates in localized form (then you can't sort them).

## Key concepts

### 1 · LLMs cannot tell you "today"

LLMs have a training cutoff and no clock. Asking "what's today's date?"
gives one of three failure modes:

| Failure | What happens |
|---|---|
| **Guessing** | LLM returns its training cutoff or a recent date that "feels right" |
| **Hallucinating** | LLM emits a plausible-but-wrong date with no uncertainty marker |
| **Asking you** | (Sometimes useful) Gemini explicitly asks the app to supply the date |

The peer-reviewed paper *Are Large Language Models Temporally Grounded?*
(NAACL 2024) confirms this is structural — models "confidently provided
incorrect temporal information, showing no uncertainty markers".

**The fix** has three shapes, in order of preference:

1. **Anchor in the system prompt:** `"Today is {date.today().isoformat()}."` — cheap, no tool round-trip.
2. **Provide as a tool:** `@tool today_iso()` — the model fetches it deterministically when needed. Best for agents that *might* need it but might not.
3. **Pre-process the query:** scan the user's query for relative references and resolve them server-side before the LLM ever sees them.

### 2 · LLMs cannot do date arithmetic

The *Test of Time* benchmark (arXiv 2406.09170) showed LLMs "approximate
correct calculations but often stumble in final steps". Off-by-one errors
on "3 weeks from today", "next Tuesday", "5 months ago" are routine.

**The fix:** deterministic arithmetic tools. The lesson ships:

| Tool | What it does |
|---|---|
| `today_iso()` | `→ "2026-05-23"` |
| `add_days(iso, n)` | `add_days("2026-05-23", 21)` → `"2026-06-13"` |
| `add_months(iso, n)` | `add_months("2026-05-23", 3)` → `"2026-08-23"` (clamps if needed) |
| `parse_relative_date(text)` | `parse_relative_date("3 weeks from now")` → ISO (delegates to `dateparser` with `RELATIVE_BASE=today`) |
| `date_diff_days(a, b)` | `date_diff_days("2026-12-25", "2026-05-23")` → `216` |

The LLM **never does arithmetic** — it picks the right tool and reads back
the deterministic result. This is the same pattern as `@tool add(a, b)`
for math (lesson 05), applied to dates.

### 3 · Localized output with `babel`

[`babel`](https://babel.pocoo.org/) is the canonical Python library for
locale-aware date formatting. It wraps the Unicode CLDR (Common Locale
Data Repository) — the same database that ships with iOS, Android, glibc.

```python
from datetime import date
from babel.dates import format_date

d = date(2026, 5, 23)

format_date(d, format="long", locale="en_US")   # 'May 23, 2026'
format_date(d, format="long", locale="tr_TR")   # '23 Mayıs 2026'
format_date(d, format="long", locale="de_DE")   # '23. Mai 2026'
format_date(d, format="long", locale="fr_FR")   # '23 mai 2026'
format_date(d, format="long", locale="ar_SA")   # '23 مايو 2026'
format_date(d, format="long", locale="ja_JP")   # '2026年5月23日'
```

Four named formats give different levels of verbosity:

| `format=` | en_US | tr_TR |
|---|---|---|
| `"short"`  | `5/23/26` | `23.05.2026` |
| `"medium"` | `May 23, 2026` | `23 May 2026` |
| `"long"`   | `May 23, 2026` | `23 Mayıs 2026` |
| `"full"`   | `Saturday, May 23, 2026` | `23 Mayıs 2026 Cumartesi` |

Need something the named formats don't cover? Pass a **CLDR pattern
string** directly:

```python
format_date(d, format="EEEE, d MMMM y", locale="tr_TR")
# 'Cumartesi, 23 Mayıs 2026'
```

`EEEE` = full weekday, `d` = day, `MMMM` = full month, `y` = year. The
full pattern grammar lives in the [CLDR LDML spec](https://www.unicode.org/reports/tr35/tr35-dates.html#Date_Format_Patterns).

### 4 · Calendar conversion

When your user's locale uses a non-Gregorian calendar, you have two
options.

**Option A — let `babel` do it.** For locales like `ja_JP_TRADITIONAL`
(Japanese era), babel automatically renders Imperial-era markers. For
Persian (`fa_IR`) the default is "Persian script but Gregorian numbers";
to get full Jalali you need Option B.

**Option B — convert explicitly with `jdatetime` / `hijridate`:**

```python
import jdatetime
g = date(2026, 5, 23)
jalali = jdatetime.date.fromgregorian(date=g)
jalali.strftime("%Y/%m/%d")                # '1405/03/02'

from hijridate import Gregorian
h = Gregorian(2026, 5, 23).to_hijri()
f"{h.year}-{h.month:02d}-{h.day:02d}"      # '1447-12-06'  (Dhu al-Hijjah)
```

Note: `hijri-converter` was the older package; it's deprecated in favor
of `hijridate` (same author, same API surface). The lesson uses
`hijridate`.

### 5 · The architectural rule

> **Internally: ISO 8601 always.**
> **At the edge: localize per user.**

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   USER                                                       │
│    │   "Randevum ne zaman?"          (locale: tr_TR)         │
│    ▼                                                          │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  AGENT (works only in ISO 8601)                       │    │
│  │   today_iso() → "2026-05-23"                          │    │
│  │   add_days("2026-05-23", 14) → "2026-06-06"           │    │
│  │   appointment.birth_date = "2026-06-06"               │    │
│  └────────────────────────┬─────────────────────────────┘    │
│                            │                                  │
│                            ▼                                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  PRESENTATION LAYER (knows user's locale)             │    │
│  │   localize_date(date(2026, 6, 6), "tr_TR", "long")    │    │
│  │   → "6 Haziran 2026"                                  │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│   REPLY  "Randevunuz 6 Haziran 2026 günü."                   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

Three rules fall out:

1. **Storage is always ISO.** Database columns, JSON fields, Pydantic schemas — all ISO 8601. Localization happens nowhere in storage.
2. **The LLM works in ISO.** Tool inputs, tool outputs, prompt anchors — all ISO. The model never sees a localized date in input or output.
3. **Localization happens once, at the edge.** A `localize_date(iso, locale)` call in the presentation layer (or `render_localized` tool invoked at the very end). One function, one place.

## Walk through `example.py`

Three runnable demos:

| Function | Needs API key? | What it shows |
|---|---|---|
| `demo_localization()` (`--localization`) | no | The user's main question — same ISO date across 10 locales + 4 format levels + custom CLDR patterns |
| `demo_calendars()` (`--calendars`) | no | Jalali / Hijri / Japanese era conversions |
| `demo_date_tools()` (`--tools`) | yes | Agent with `today_iso() / add_days / parse_relative_date / render_localized` tools, asked questions in en_US, tr_TR, de_DE |

The `localize_date()` helper is reusable — `from lessons.23_date_localization.example import localize_date` and it works anywhere.

## Run it

```bash
# All three (the localization + calendars demos need NO API key)
uv run python -m lessons.23_date_localization.example

# Just the locale grid
uv run python -m lessons.23_date_localization.example --localization

# Just the calendars (Jalali / Hijri / Japanese era)
uv run python -m lessons.23_date_localization.example --calendars

# Just the agent + tools (needs ANTHROPIC_API_KEY or OPENAI_API_KEY)
uv run python -m lessons.23_date_localization.example --tools
```

Sample output (locale grid):

```
en_US     [American English                ]  May 23, 2026
en_GB     [British English                 ]  23 May 2026
tr_TR     [Turkish                         ]  23 Mayıs 2026     ← what you asked for
de_DE     [German                          ]  23. Mai 2026
fr_FR     [French                          ]  23 mai 2026
ar_SA     [Arabic (Saudi)                  ]  23 مايو 2026
fa_IR     [Persian (Iran)                  ]  23 مهٔ 2026
ja_JP     [Japanese                        ]  2026年5月23日
zh_CN     [Chinese (Simplified)            ]  2026年5月23日
```

## Debug it

Put `breakpoint()` after a `format_date(...)` call and inspect what
came back. The most common surprise is that `babel` returns *bytes-like*
behaviour on edge cases (very old locales, unsupported scripts) —
catching that early saves a JSON-encoding bug later.

For the agent demo, set a breakpoint in any tool to see what arguments
the LLM is calling with. You'll notice the model often *re*-asks for
`today_iso()` between tools — that's the agent loop working as intended,
keeping anchor dates fresh.

## Anti-patterns

| Smell | Why it's bad | Fix |
|---|---|---|
| Asking the LLM "what's today?" without a tool | Confidently wrong (training cutoff) | `@tool today_iso()` |
| Asking the LLM to do `"5 weeks from today"` arithmetic in prose | Off-by-one routine | `@tool add_days()` |
| Storing dates in localized form (`"23 Mayıs 2026"`) | Can't sort, can't compare, can't migrate | ISO 8601 storage; localize only at render time |
| Using `datetime.strftime("%B %d, %Y")` for non-English | `%B` returns English month name unless you set `locale.setlocale(...)` (global state, fragile) | `babel.format_date(d, locale="tr_TR")` |
| Manually translating month names with a dictionary | Reinvents CLDR badly, breaks on plural forms / declensions (Russian, Polish) | Use `babel` (or ICU bindings) |
| Mixing Gregorian and Jalali in the same field | Year 1405 vs year 2026 — silent data corruption | Two separate fields if needed; never one |
| Letting babel pick the calendar when you wanted Gregorian | `fa_IR` formatting may use Persian script but Gregorian numbers, or actual Jalali depending on version | Be explicit: pick `babel` for Gregorian-in-locale, `jdatetime` for true Jalali |

## Try it yourself

- **Add a Turkish appointment-reminder flow.** Take a date stored as
  `"2026-06-06"` and produce *"Randevunuz 6 Haziran 2026 Cumartesi günü
  saat 14:30'da."* — that's `localize_date` + a Turkish time string.
- **Wire `render_localized` into the customer-support capstone.** When
  the user types in Turkish, set `locale="tr_TR"` in the agent state and
  let the agent call it before every user-facing response.
- **Calendar-aware birth-date storage.** Store both the ISO Gregorian
  date AND the calendar name (`"jalali"`, `"hijri"`, `"gregorian"`). At
  render time, choose the right path.
- **Stretch:** build a `parse_localized_input(text, locale)` for the
  *input* side — accept "23 Mayıs 2026" from a Turkish user and parse it
  back to ISO. `dateparser` already does this (it supports 200+ locales);
  this lesson's helper closes the loop.

## References

### Vendor / official docs

- [`babel` 2.17 — Date and Time formatting](https://babel.pocoo.org/en/latest/dates.html) · the canonical CLDR-backed Python library
- [Unicode CLDR LDML — Date Format Patterns](https://www.unicode.org/reports/tr35/tr35-dates.html#Date_Format_Patterns) · the pattern grammar `babel` implements
- [`jdatetime` on PyPI](https://pypi.org/project/jdatetime/) · Persian Jalali datetime bindings
- [`hijridate` (replaces `hijri-converter`)](https://app.readthedocs.org/projects/hijri-converter/) · Islamic Hijri / Umm al-Qura calendar
- [`persiantools` on PyPI](https://pypi.org/project/persiantools/) · alternative Jalali library with date arithmetic

### Peer-reviewed papers (why LLMs need these tools)

- [*Are Large Language Models Temporally Grounded?* — NAACL 2024](https://aclanthology.org/2024.naacl-long.391.pdf) · "confidently provided incorrect temporal information, showing no uncertainty markers"
- [*Test of Time: Benchmark for LLM Temporal Reasoning* — arXiv 2406.09170](https://arxiv.org/html/2406.09170v1) · "approximate correct calculations but often stumble in final steps"
- [*DateLogicQA* — arXiv 2412.13377](https://arxiv.org/pdf/2412.13377) · 190-question benchmark across date formats and reasoning types

### Engineering blogs

- [Riccardo Tartaglia · *Teaching Your LLM to Tell Time*](https://medium.com/@riccardo.tartaglia/teaching-your-llm-to-tell-time-a-practical-guide-to-llm-tool-integration-a52436f68a58) · the agentic `today()` pattern in depth
- [James Tang · *Best Practices for Handling Dates in Structured Output*](https://medium.com/@jamestang/best-practices-for-handling-dates-in-structured-output-in-llm-2efc159e1854) · the ISO-everywhere-internally rule

### Pairs with

- **[Lesson 21 · Date parsing](../21_date_parsing/README.md)** — the input side: extracting ISO dates from messy text.
- **[Lesson 22 · LLM application architecture](../22_architecture/README.md)** — this lesson is concrete reinforcement of the **Interface Layer** (presentation, locale-aware) and the **Orchestration Layer** (tool calls instead of LLM arithmetic).
- **[`docs/research/llm-date-solutions-deep-dive.md`](../../docs/research/llm-date-solutions-deep-dive.md)** — approach #7 (agentic tool use) in the 10-approach taxonomy is what this lesson operationalises.

## Next →

That's the curriculum, end-to-end on dates. From here, build something:
the [capstones in `projects/`](../../projects/) are the natural next step.
