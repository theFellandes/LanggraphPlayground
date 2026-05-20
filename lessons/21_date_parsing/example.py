"""Lesson 21 · Date parsing with LLMs — the four-layer pipeline.

Implements the defence-in-depth pattern from
`docs/research/date-parsing-with-llms.md`:

  LAYER 1 · anchored prompt with few-shot ISO 8601 examples
  LAYER 2 · provider-side structured output (with_structured_output)
  LAYER 3 · dateparser.parse(STRICT_PARSING=True, RELATIVE_BASE=today)
  LAYER 4 · Pydantic field_validator enforces birth-date sanity

Two SCHEMAS demonstrating both fixes for the LangChain date-field bug
(see docs/research/langchain-date-field-bug.md):

  Solution A — birth_date: str  + dateparser-driven validator
                (works on every provider; downstream sees a string)
  Solution B — birth_date: date + with_structured_output_safe() sanitizer
                (keeps Pydantic's native date type; needs the sanitizer)

Three entry points:

  main()             — full pipeline, Solution A; needs an API key
  main_native()      — full pipeline, Solution B; needs an API key
  demo_validation()  — layers 3 + 4 only, no API key needed

Run:
    uv run python -m lessons.21_date_parsing.example
    uv run python -m lessons.21_date_parsing.example --native
    uv run python -m lessons.21_date_parsing.example --validation-only
"""

from __future__ import annotations

import sys
from datetime import date
from typing import Literal

import dateparser
from pydantic import BaseModel, Field, ValidationError, field_validator

from shared import get_llm
from shared.llm import with_structured_output_safe
from shared.pretty import console, section

TODAY = date.today()
MIN_BIRTH_YEAR = 1900
MAX_AGE_YEARS = 120

# ───────────────────────────────────────────────────────────────────────────
# LAYER 4 (with LAYER 3 wrapped in it) — Pydantic schema + validators
# ───────────────────────────────────────────────────────────────────────────
class BirthDateExtraction(BaseModel):
    """Schema the LLM populates; validators re-parse + sanity-check.

    `birth_date` stays as `str` not `datetime.date` so the LLM's raw output is
    preserved for audit and so we control the parse step ourselves
    (LangChain + Mistral has a known issue with date-typed schema fields).
    """

    birth_date: str | None = Field(
        default=None,
        description=(
            "The person's date of birth in ISO 8601 (YYYY-MM-DD). "
            "Null if not present in the source text. Do NOT invent a date."
        ),
    )
    raw_text: str = Field(
        default="",
        description="The exact substring the date came from. Empty if no date.",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Your confidence in the extraction, 0 to 1.",
    )

    @field_validator("birth_date")
    @classmethod
    def _layer3_reparse_and_layer4_sanity(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None

        # LAYER 3 · deterministic re-parse — catches Feb 31, weird locale strings.
        parsed = dateparser.parse(
            v,
            settings={
                "STRICT_PARSING": True,        # reject ambiguous strings
                "DATE_ORDER": "YMD",            # we asked the LLM for ISO
                "RELATIVE_BASE": _to_datetime(TODAY),
                "PREFER_DAY_OF_MONTH": "first",
            },
        )
        if parsed is None:
            raise ValueError(
                f"layer-3 reject: {v!r} is not a valid date string"
            )
        d = parsed.date()

        # LAYER 4 · domain sanity for birth dates.
        if d.year < MIN_BIRTH_YEAR:
            raise ValueError(
                f"layer-4 reject: {d.isoformat()} is before {MIN_BIRTH_YEAR}"
            )
        if d > TODAY:
            raise ValueError(
                f"layer-4 reject: {d.isoformat()} is in the future"
            )
        age = (TODAY - d).days // 365
        if age > MAX_AGE_YEARS:
            raise ValueError(
                f"layer-4 reject: implied age {age} > {MAX_AGE_YEARS}"
            )

        return d.isoformat()


def _to_datetime(d: date):
    """`dateparser` wants datetime for RELATIVE_BASE."""
    from datetime import datetime
    return datetime(d.year, d.month, d.day)


# ───────────────────────────────────────────────────────────────────────────
# LAYER 1 — the anchored prompt with few-shot ISO 8601 examples
# ───────────────────────────────────────────────────────────────────────────
PROMPT = """Today is {today}.

Extract the person's date of birth from the text below.

Rules:
- Use ISO 8601 (YYYY-MM-DD) in the `birth_date` field.
- If the date is relative ("5 years ago"), compute it against today.
- If no birth date is mentioned, set `birth_date` to null. Do NOT invent.
- Copy the exact source substring into `raw_text`.

Examples:
  Text:   "Born April 5, 1987 in Istanbul."
  Output: {{"birth_date": "1987-04-05", "raw_text": "April 5, 1987", "confidence": 1.0}}

  Text:   "DOB: 04/05/87 (US format, MM/DD/YY)"
  Output: {{"birth_date": "1987-04-05", "raw_text": "04/05/87", "confidence": 0.9}}

  Text:   "She is 35 years old."
  Output: {{"birth_date": null, "raw_text": "", "confidence": 0.0}}

Text:
{text}"""


# ───────────────────────────────────────────────────────────────────────────
# LAYER 1 + 2 + 3 + 4 — the full pipeline
# ───────────────────────────────────────────────────────────────────────────
def extract_birth_date(text: str) -> BirthDateExtraction | str:
    """Run the full four-layer pipeline — SOLUTION A (str + validator).

    `birth_date` is typed as `str` in the schema, so Pydantic never emits
    `"format": "date"` — works on every provider.

    Returns:
        BirthDateExtraction on success, or a string describing the failure
        (which layer rejected, what message). Real apps would retry-with-error
        via Instructor / LangChain middleware here.
    """
    llm = get_llm().with_structured_output(BirthDateExtraction)
    try:
        return llm.invoke(PROMPT.format(today=TODAY.isoformat(), text=text))
    except ValidationError as e:
        # Layer 3 or 4 caught something. The error message tells you which.
        return f"validation: {e.errors()[0]['msg']}"
    except Exception as e:
        return f"runtime: {type(e).__name__}: {e}"


# ───────────────────────────────────────────────────────────────────────────
# SOLUTION B · keep the native datetime.date type via the schema sanitizer
# ───────────────────────────────────────────────────────────────────────────
class BirthDateNative(BaseModel):
    """Same idea as BirthDateExtraction but with a native `date` field.

    This schema looks like the "obvious" Pydantic version — and *would* be
    rejected by Mistral / OpenAI strict mode (HTTP 400, "unsupported keyword
    'format'") if we called `.with_structured_output()` directly.

    We use `with_structured_output_safe()` instead — it strips the offending
    `"format": "date"` key before sending the schema to the provider.
    See docs/research/langchain-date-field-bug.md for the full story.
    """

    birth_date: date | None = Field(
        default=None,
        description="The person's date of birth (YYYY-MM-DD). Null if absent.",
    )
    raw_text: str = Field(default="", description="Exact source substring.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("birth_date")
    @classmethod
    def _layer4_only(cls, v: date | None) -> date | None:
        # Pydantic already did layer 3 for us (string -> date). We only need
        # the domain sanity check.
        if v is None:
            return None
        if v.year < MIN_BIRTH_YEAR:
            raise ValueError(f"layer-4 reject: {v.isoformat()} is before {MIN_BIRTH_YEAR}")
        if v > TODAY:
            raise ValueError(f"layer-4 reject: {v.isoformat()} is in the future")
        if (TODAY - v).days // 365 > MAX_AGE_YEARS:
            raise ValueError(f"layer-4 reject: implied age > {MAX_AGE_YEARS}")
        return v


def extract_birth_date_native(text: str) -> BirthDateNative | str:
    """Run the pipeline with a native `datetime.date` field — SOLUTION B."""
    llm = with_structured_output_safe(BirthDateNative)
    try:
        return llm.invoke(PROMPT.format(today=TODAY.isoformat(), text=text))
    except ValidationError as e:
        return f"validation: {e.errors()[0]['msg']}"
    except Exception as e:
        return f"runtime: {type(e).__name__}: {e}"


# ───────────────────────────────────────────────────────────────────────────
# Eval set covering the 10 failure modes
# ───────────────────────────────────────────────────────────────────────────
Case = tuple[str, str, str | None]
#               (label,   text,                          expected ISO or None)

EVAL_SET: list[Case] = [
    ("ISO clean",           "Born 1987-04-05 in Istanbul.",              "1987-04-05"),
    ("English long form",   "Date of birth: April 5, 1987.",             "1987-04-05"),
    ("DD/MM/YYYY",          "DOB: 05/04/1987 (UK format).",              "1987-04-05"),
    ("MM/DD/YYYY",          "DOB: 04/05/1987 (US format).",              "1987-04-05"),
    ("Two-digit year",      "DOB 04/05/87 (MM/DD/YY).",                  "1987-04-05"),
    ("Relative",            "She is 35 years old.",                       None),       # we don't invent
    ("Partial — year only", "Born sometime in 1987.",                    "1987-01-01"),
    ("None present",        "The patient lives in Istanbul.",            None),
    ("Hallucination bait",  "He is a famous author.",                     None),
    # The next four should be REJECTED by layer 3 or 4, not silently accepted.
    ("Invalid Feb 31",      "DOB Feb 31, 1990.",                          "REJECT"),
    ("Pre-1900",            "Born in 1850.",                              "REJECT"),
    ("Future date",         "DOB 2099-01-01.",                            "REJECT"),
    ("Implausible age",     "Born 1700-06-15.",                           "REJECT"),
]


# ───────────────────────────────────────────────────────────────────────────
# demo_validation — exercise layers 3 + 4 only, no API key needed
# ───────────────────────────────────────────────────────────────────────────
def demo_validation() -> None:
    """Skip the LLM. Show layers 3 + 4 on raw strings."""
    section("Lesson 21 · validation-only demo (no API key required)")
    console.print(f"[dim]Today is {TODAY.isoformat()}. MIN year = {MIN_BIRTH_YEAR}, "
                  f"MAX age = {MAX_AGE_YEARS}.[/]\n")

    raws = [
        "1987-04-05",           # ✓ clean ISO
        "April 5, 1987",        # ✓ long-form English
        "05/04/1987",           # ⚠ ambiguous; dateparser uses DATE_ORDER='YMD'
        "Feb 31, 1990",         # ✗ invalid → layer-3 reject
        "1850-01-01",           # ✗ pre-1900 → layer-4 reject
        "2099-01-01",           # ✗ future → layer-4 reject
        "completely-nonsense",  # ✗ unparseable → layer-3 reject
        "",                     # → None (treated as "no date")
    ]

    for raw in raws:
        try:
            out = BirthDateExtraction(birth_date=raw, raw_text=raw, confidence=1.0)
            console.print(f"  [green]✓[/]  {raw!r:30}  →  {out.birth_date}")
        except ValidationError as e:
            msg = e.errors()[0]["msg"]
            console.print(f"  [red]✗[/]  {raw!r:30}  →  {msg}")


def _expected_marker(expected: str | None) -> str:
    if expected == "REJECT":
        return "[yellow](should reject)[/]"
    if expected is None:
        return "[dim](no date)[/]"
    return expected


def _run_eval(extractor, schema_label: str) -> None:
    """Shared eval driver used by both main() and main_native()."""
    section(f"Lesson 21 · full pipeline — {schema_label}")
    console.print(f"[dim]Today is {TODAY.isoformat()}.[/]\n")

    correct = 0
    for label, text, expected in EVAL_SET:
        result = extractor(text)
        if isinstance(result, (BirthDateExtraction, BirthDateNative)):
            got_raw = result.birth_date
            got = got_raw.isoformat() if isinstance(got_raw, date) else got_raw
            ok = (
                (expected == "REJECT" and got_raw is None) or
                (expected is None and got_raw is None) or
                (expected == got)
            )
        else:
            got = f"(rejected: {result})"
            ok = expected == "REJECT"

        marker = "[green]✓[/]" if ok else "[red]✗[/]"
        console.print(f"{marker} {label:22} expected={_expected_marker(expected):14}  got={got}")
        if ok:
            correct += 1

    console.print(f"\n[bold]{correct}/{len(EVAL_SET)} cases passed[/]")


def main() -> None:
    """SOLUTION A — birth_date: str + validator (universal)."""
    _run_eval(extract_birth_date, "Solution A · str + validator")


def main_native() -> None:
    """SOLUTION B — birth_date: date + sanitizer wrapper (native typing)."""
    _run_eval(extract_birth_date_native, "Solution B · date + sanitizer")


if __name__ == "__main__":
    if "--validation-only" in sys.argv:
        demo_validation()
    elif "--native" in sys.argv:
        main_native()
    else:
        main()
