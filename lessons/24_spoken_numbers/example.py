"""Lesson 24 · Spoken-number → digit normalization.

The follow-up to lessons 21 and 23. When a customer speaks
"yirmi beş on iki yetmiş dokuz" (25 / 12 / 79), we need to turn those
words into digits BEFORE letting the LLM anywhere near them — otherwise
we hit the decomposed-field hallucination from lesson 21.

No PyPI library covers Turkish (text2num supports EN/FR/ES/DE/PT/IT/NL
but not TR; zemberek does morphology, not number parsing). So we roll
our own ~80-line rule-based parser — Turkish numbers are regular
enough that this is the right shape.

This file provides three things:

  1. `turkish_words_to_int(text)`         — the core: a single Turkish
                                            number phrase to int
  2. `parse_turkish_dob_phrase(text)`     — the segmenter: a
                                            day-month-year sequence
                                            to (day, month, year)
  3. demos showing both, plus a fallback
     pattern using an LLM ONLY as a single-task converter

Run:
    uv run python -m lessons.24_spoken_numbers.example
"""

from __future__ import annotations

import sys

from shared.pretty import console, section

# ───────────────────────────────────────────────────────────────────────────
# Turkish number vocabulary (covers 0 to 999_999)
# ───────────────────────────────────────────────────────────────────────────
TR_UNITS: dict[str, int] = {
    "sıfır": 0, "sifir": 0,
    "bir":   1, "iki":  2, "üç":   3, "uc":   3, "dört": 4, "dort": 4,
    "beş":   5, "bes":  5, "altı": 6, "alti": 6,
    "yedi":  7, "sekiz": 8, "dokuz": 9,
}
TR_TENS: dict[str, int] = {
    "on":     10, "yirmi": 20, "otuz":  30, "kırk":   40, "kirk":  40,
    "elli":   50, "altmış": 60, "altmis": 60, "yetmiş": 70, "yetmis": 70,
    "seksen": 80, "doksan": 90,
}
TR_HUNDRED  = {"yüz", "yuz"}
TR_THOUSAND = {"bin"}


def turkish_words_to_int(text: str) -> int | None:
    """Parse one Turkish number phrase into an integer.

    Handles 0–999_999. Diacritic-insensitive (accepts both "beş" and "bes").

    Examples:
        "yirmi beş"                  → 25
        "doksan yedi"                → 97
        "iki bin bir"                → 2001
        "bin dokuz yüz seksen yedi"  → 1987
        "yüz on bir"                 → 111

    Returns None if the phrase isn't a recognisable Turkish number.
    """
    tokens = text.lower().strip().split()
    if not tokens:
        return None

    # Recurse: split on "bin" (thousand) to handle the multiplicative structure.
    if any(t in TR_THOUSAND for t in tokens):
        idx = next(i for i, t in enumerate(tokens) if t in TR_THOUSAND)
        left_tokens = tokens[:idx]
        right_tokens = tokens[idx + 1:]
        left_value = _parse_under_thousand(left_tokens) if left_tokens else 1
        right_value = _parse_under_thousand(right_tokens) if right_tokens else 0
        if left_value is None or right_value is None:
            return None
        return left_value * 1000 + right_value

    return _parse_under_thousand(tokens)


def _parse_under_thousand(tokens: list[str]) -> int | None:
    """Parse a Turkish number < 1000 from already-split tokens."""
    if not tokens:
        return 0

    # Split on "yüz" (hundred) for the multiplicative form.
    if any(t in TR_HUNDRED for t in tokens):
        idx = next(i for i, t in enumerate(tokens) if t in TR_HUNDRED)
        left_tokens = tokens[:idx]
        right_tokens = tokens[idx + 1:]
        left_value = _parse_under_hundred(left_tokens) if left_tokens else 1
        right_value = _parse_under_hundred(right_tokens) if right_tokens else 0
        if left_value is None or right_value is None:
            return None
        return left_value * 100 + right_value

    return _parse_under_hundred(tokens)


def _parse_under_hundred(tokens: list[str]) -> int | None:
    """Parse a Turkish number < 100: optional tens word + optional unit word."""
    if not tokens:
        return 0
    if len(tokens) == 1:
        t = tokens[0]
        if t in TR_TENS:  return TR_TENS[t]
        if t in TR_UNITS: return TR_UNITS[t]
        return None
    if len(tokens) == 2:
        tens, unit = tokens
        if tens in TR_TENS and unit in TR_UNITS:
            return TR_TENS[tens] + TR_UNITS[unit]
    return None


# ───────────────────────────────────────────────────────────────────────────
# Date-phrase segmenter — "yirmi beş on iki yetmiş dokuz" → (25, 12, 79)
# ───────────────────────────────────────────────────────────────────────────
def parse_turkish_dob_phrase(text: str) -> tuple[int, int, int] | None:
    """Split a spoken Turkish DOB into (day, month, year).

    Assumes the order is day-month-year (Turkish convention).

    Two segmentation strategies:
      A. If a Turkish month NAME appears (mart, mayıs, kasım, ...), use it
         as the month and parse before/after as day/year. Unambiguous.
      B. Otherwise, all three slots are numeric. Try every 3-segment
         partition and accept the first one that satisfies the range
         constraints (day ∈ [1,31], month ∈ [1,12], year valid).

    Examples:
        "yirmi beş on iki yetmiş dokuz"         → (25, 12, 79)
        "on beş mart bin dokuz yüz seksen yedi" → (15, 3, 1987)
        "yirmi sekiz şubat iki bin"             → (28, 2, 2000)
        "bir bir iki bin bir"                   → (1, 1, 2001)
    """
    tokens = text.lower().split()
    if not tokens:
        return None

    # Strategy A — a Turkish month name pins the month unambiguously.
    for i, tok in enumerate(tokens):
        if tok in TR_MONTH_NAMES:
            day_tokens  = tokens[:i]
            year_tokens = tokens[i + 1:]
            if not day_tokens or not year_tokens:
                continue
            day  = turkish_words_to_int(" ".join(day_tokens))
            year = turkish_words_to_int(" ".join(year_tokens))
            if day is None or year is None:
                continue
            month = TR_MONTH_NAMES[tok]
            if 1 <= day <= 31 and (1 <= year <= 99 or 1900 <= year <= 2100):
                return (day, month, year)
            return None

    # Strategy B — all-numeric. Try every 3-segment partition.
    n = len(tokens)
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            day  = turkish_words_to_int(" ".join(tokens[:i]))
            mon  = turkish_words_to_int(" ".join(tokens[i:j]))
            year = turkish_words_to_int(" ".join(tokens[j:]))
            if day is None or mon is None or year is None:
                continue
            if 1 <= day <= 31 and 1 <= mon <= 12 and (1 <= year <= 99 or 1900 <= year <= 2100):
                return (day, mon, year)
    return None


TR_MONTH_NAMES = {
    "ocak":1, "şubat":2, "mart":3, "nisan":4, "mayıs":5, "haziran":6,
    "temmuz":7, "ağustos":8, "eylül":9, "ekim":10, "kasım":11, "aralık":12,
    "subat":2, "mayis":5, "agustos":8, "eylul":9, "kasim":11, "aralik":12,
}


# ───────────────────────────────────────────────────────────────────────────
# FUZZY PARTIAL MATCHING — for typos / speech-to-text glitches
# ───────────────────────────────────────────────────────────────────────────
# When the parser fails outright on a token, fall back to a fuzzy match against
# the known vocabulary. Three confidence tiers drive the next action.

_TR_VOCAB: set[str] = (
    set(TR_UNITS) | set(TR_TENS) | TR_HUNDRED | TR_THOUSAND | set(TR_MONTH_NAMES)
)


def _fuzzy_normalize_token(token: str, threshold: int = 85) -> tuple[str | None, float]:
    """Snap a possibly-misspelled token to the nearest Turkish vocab word.

    Returns (best_match, score). Returns (None, score) if the best score
    is below `threshold`.

    >>> _fuzzy_normalize_token("yirmı")      # bad dot on the ı
    ('yirmi', 88.9)
    >>> _fuzzy_normalize_token("xyzzy")      # nonsense
    (None, ...)
    """
    from rapidfuzz import process, fuzz
    match = process.extractOne(
        token.lower(), _TR_VOCAB, scorer=fuzz.ratio, score_cutoff=threshold,
    )
    if match is None:
        # Try partial_ratio as a fallback (lower bar — substrings)
        match = process.extractOne(
            token.lower(), _TR_VOCAB, scorer=fuzz.partial_ratio,
            score_cutoff=max(threshold, 80),
        )
    if match is None:
        return (None, 0.0)
    return (match[0], match[1])


# Three confidence tiers — drives the escalation policy below
ACCEPT_THRESHOLD = 95   # auto-snap silently
CONFIRM_THRESHOLD = 80  # snap but flag for user confirmation
# Below 80 → don't snap; reject and ask user to repeat


def turkish_words_to_int_fuzzy(
    text: str,
    threshold: int = ACCEPT_THRESHOLD,
) -> tuple[int | None, list[str]]:
    """Fuzzy variant — normalises typos before parsing.

    Returns (parsed_int_or_None, list_of_corrections_made).
    The corrections list lets the caller surface what was auto-corrected.
    """
    tokens = text.lower().strip().split()
    corrections: list[str] = []
    normalised: list[str] = []
    for tok in tokens:
        if tok in _TR_VOCAB:
            normalised.append(tok)
            continue
        candidate, score = _fuzzy_normalize_token(tok, threshold)
        if candidate is None:
            normalised.append(tok)  # leave it; parser will fail and tell us where
        else:
            normalised.append(candidate)
            if candidate != tok:
                corrections.append(f"{tok!r}→{candidate!r} ({score:.0f}%)")
    parsed = turkish_words_to_int(" ".join(normalised))
    return (parsed, corrections)


# ───────────────────────────────────────────────────────────────────────────
# Multilingual @tool — wraps `text2num` for EN/FR/ES/DE/PT/IT/NL,
# routes to our Turkish parser for TR.
# ───────────────────────────────────────────────────────────────────────────
from langchain_core.tools import tool


@tool
def parse_spoken_number(text: str, locale: str = "en") -> dict:
    """Parse a spoken-form number phrase into an integer.

    Args:
        text:   The spoken-form number phrase exactly as the user said it.
                Examples: 'one thousand nine hundred eighty seven',
                          'mille neuf cent quatre-vingt-sept',
                          'yirmi beş'
        locale: ISO language code. Supported:
                'en', 'fr', 'es', 'de', 'pt', 'it', 'nl' (via text2num)
                'tr' (via this repo's Turkish parser)

    Returns:
        {"ok": True,  "value": int, "engine": str}                on success
        {"ok": False, "error": str, "supported_locales": [...]}  on failure

    The agent should call this tool RATHER THAN doing the conversion in-prompt.
    See lesson 21's "decomposed-fields hallucination" — every numeric digit
    the LLM generates itself is a chance to drift. Tool output is deterministic.
    """
    locale = locale.lower()

    if locale == "tr":
        value, corrections = turkish_words_to_int_fuzzy(text)
        if value is None:
            return {"ok": False, "error": f"could not parse Turkish phrase {text!r}",
                    "corrections_attempted": corrections}
        return {"ok": True, "value": value, "engine": "turkish_rule_parser",
                "corrections": corrections}

    if locale in {"en", "fr", "es", "de", "pt", "it", "nl"}:
        try:
            from text_to_num import text2num
            return {"ok": True, "value": text2num(text, locale), "engine": "text2num"}
        except (ValueError, Exception) as e:
            return {"ok": False, "error": str(e)}

    return {
        "ok": False,
        "error": f"unsupported locale {locale!r}",
        "supported_locales": ["en", "fr", "es", "de", "pt", "it", "nl", "tr"],
    }


# ───────────────────────────────────────────────────────────────────────────
# Hybrid pattern — try rule parser first, fall back to a single-task LLM
# ───────────────────────────────────────────────────────────────────────────
def resolve_dob_with_fallback(spoken: str) -> tuple[int, int, int] | None:
    """Try the rule parser first; on failure, ask an LLM for ONE thing only.

    The LLM call is a *narrow* converter — single field, single task,
    minimal hallucination surface. Lesson 21's wide-schema anti-pattern
    is exactly what we're avoiding by NOT asking the LLM to fill day /
    month / year as separate ints.
    """
    rule_result = parse_turkish_dob_phrase(spoken)
    if rule_result is not None:
        return rule_result

    # Fallback — ONE LLM call, ONE field, ISO output.
    try:
        from pydantic import BaseModel, Field
        from shared.llm import get_llm

        class IsoDob(BaseModel):
            iso_date: str = Field(
                description="Date of birth in ISO 8601 (YYYY-MM-DD). "
                "Convert the Turkish spoken number phrase exactly. "
                "If the year has 2 digits, assume 19xx if >40, else 20xx.",
                pattern=r"^\d{4}-\d{2}-\d{2}$",
            )

        llm = get_llm().with_structured_output(IsoDob)
        parsed = llm.invoke(
            "Convert this Turkish-spoken DOB to ISO 8601:\n\n" + spoken
        )
        from datetime import date
        d = date.fromisoformat(parsed.iso_date)
        return (d.day, d.month, d.year)
    except Exception as e:
        console.print(f"[red]LLM fallback failed: {e}[/]")
        return None


# ───────────────────────────────────────────────────────────────────────────
# DEMOS
# ───────────────────────────────────────────────────────────────────────────
SINGLE_NUMBER_CASES = [
    ("yirmi beş",                     25),
    ("doksan yedi",                   97),
    ("on iki",                        12),
    ("yetmiş dokuz",                  79),
    ("iki bin bir",                   2001),
    ("bin dokuz yüz seksen yedi",     1987),
    ("yüz on bir",                    111),
    ("üç bin dört yüz elli altı",     3456),
    ("sıfır",                         0),
    ("doksan",                        90),
    ("kırk",                          40),     # diacritic
    ("kirk",                          40),     # no diacritic
]

DOB_PHRASE_CASES = [
    ("yirmi beş on iki yetmiş dokuz",        (25, 12, 79)),
    ("bir bir iki bin bir",                  (1, 1, 2001)),
    ("üç altı doksan yedi",                  (3, 6, 97)),
    ("on beş mart bin dokuz yüz seksen yedi", (15, 3, 1987)),
    ("yirmi sekiz şubat iki bin",            (28, 2, 2000)),
]


def demo_single_numbers() -> None:
    section("Demo 1 · Turkish number-words → int (the core parser)")
    passed = 0
    for text, expected in SINGLE_NUMBER_CASES:
        got = turkish_words_to_int(text)
        ok = got == expected
        marker = "[green]✓[/]" if ok else "[red]✗[/]"
        console.print(f"  {marker} {text:36}  →  {got}  (expected {expected})")
        passed += int(ok)
    console.print(f"\n  [bold]{passed}/{len(SINGLE_NUMBER_CASES)} cases passed[/]")


def demo_dob_segmenter() -> None:
    section("Demo 2 · Turkish DOB-phrase segmenter (day / month / year)")
    passed = 0
    for text, expected in DOB_PHRASE_CASES:
        got = parse_turkish_dob_phrase(text)
        ok = got == expected
        marker = "[green]✓[/]" if ok else "[red]✗[/]"
        console.print(f"  {marker} {text:48}  →  {got}  (expected {expected})")
        passed += int(ok)
    console.print(f"\n  [bold]{passed}/{len(DOB_PHRASE_CASES)} cases passed[/]")


def demo_other_locales() -> None:
    section("Demo 3 · For non-Turkish locales: use `text2num` (PyPI)")
    console.print(
        "  text2num supports EN, FR, ES, PT, DE, NL, IT — not TR.\n"
        "  Install with:  uv add text2num\n"
        "  Use like:\n"
        "    from text_to_num import text2num\n"
        "    text2num('one thousand nine hundred eighty-seven', 'en')  # → 1987\n"
        "    text2num('mille neuf cent quatre-vingt-sept',     'fr')  # → 1987\n"
    )


def demo_full_pipeline() -> None:
    section("Demo 4 · Full pipeline — Turkish spoken DOB → ISO 8601 string")
    cases = [
        "yirmi beş on iki yetmiş dokuz",         # 1979-12-25
        "on beş mart bin dokuz yüz seksen yedi", # 1987-03-15
        "bir bir iki bin bir",                   # 2001-01-01
    ]
    from datetime import date
    for spoken in cases:
        parts = parse_turkish_dob_phrase(spoken)
        if parts is None:
            console.print(f"  [red]✗[/]  {spoken!r}  →  unparseable")
            continue
        day, mon, year = parts
        # Expand 2-digit years (DOB heuristic: assume 19xx for >40, else 20xx)
        if year < 100:
            year = 1900 + year if year > 40 else 2000 + year
        try:
            iso = date(year, mon, day).isoformat()
            console.print(f"  [green]✓[/]  {spoken!r:48}  →  {iso}")
        except ValueError as e:
            console.print(f"  [red]✗[/]  {spoken!r}  →  invalid: {e}")


def demo_fuzzy_partial_matching() -> None:
    section("Demo 5 · Fuzzy partial matching — handling typos / STT glitches")
    console.print(
        "  Escalation policy:\n"
        "    score ≥ 95  →  auto-correct silently\n"
        "    score 80-94 →  correct AND surface for user confirmation\n"
        "    score < 80  →  reject and ask user to repeat\n"
    )
    cases = [
        "yirmı beş on iki yetmiş dokuz",   # bad dot on ı, otherwise clean
        "doksan yedı",                       # bad dot
        "iki bın bir",                       # i → ı substitution in 'bin'
        "yirmi beş xyzzy yetmiş dokuz",     # one nonsense word
    ]
    for spoken in cases:
        # Per-token fuzzy normalize
        from rapidfuzz import process, fuzz
        tokens = spoken.lower().split()
        per_token = []
        min_score = 100.0
        for t in tokens:
            if t in _TR_VOCAB:
                per_token.append((t, 100.0)); continue
            match = process.extractOne(t, _TR_VOCAB, scorer=fuzz.ratio, score_cutoff=0)
            score = match[1] if match else 0
            per_token.append((match[0] if match else t, score))
            min_score = min(min_score, score)
        tier = ("[green]accept[/]" if min_score >= ACCEPT_THRESHOLD
                else "[yellow]confirm[/]" if min_score >= CONFIRM_THRESHOLD
                else "[red]reject[/]")
        result, corrections = turkish_words_to_int_fuzzy(spoken, threshold=CONFIRM_THRESHOLD)
        console.print(f"  [{tier}] (min token-score {min_score:.0f})  {spoken!r}")
        console.print(f"     corrections: {corrections}  →  parsed value: {result}")


def _greedy_split_turkish(text: str) -> str:
    """`num2words(25, lang='tr')` returns `'yirmibeş'` (concatenated). We want
    `'yirmi beş'`. Greedy-split against the Turkish vocab — longest match wins.

    Works because Turkish number-word formation uses a small fixed vocab
    with no overlapping prefixes.
    """
    vocab = sorted(_TR_VOCAB, key=len, reverse=True)
    out, i = [], 0
    while i < len(text):
        for word in vocab:
            if text.startswith(word, i):
                out.append(word)
                i += len(word)
                break
        else:
            i += 1   # skip unknown char (shouldn't happen for valid num2words output)
    return " ".join(out)


def num2words_turkish_spaced(n: int) -> str:
    """Round-trip-friendly Turkish: `25 → 'yirmi beş'`."""
    from num2words import num2words
    return _greedy_split_turkish(num2words(n, lang="tr"))


def demo_round_trip() -> None:
    """Property test: generate words with num2words, parse them back with our parser.

    Using a separate library as the ground-truth generator is a clean way
    to exercise many cases without hand-writing each one.
    """
    section("Demo 7 · round-trip with num2words (int → words → int)")
    targets = [0, 1, 7, 10, 12, 25, 79, 90, 97, 100, 111, 999,
               1000, 1987, 2001, 3456, 9999]
    passed = 0
    for n in targets:
        spaced = num2words_turkish_spaced(n)
        parsed = turkish_words_to_int(spaced)
        ok = parsed == n
        marker = "[green]✓[/]" if ok else "[red]✗[/]"
        console.print(f"  {marker} {n:5}  → {spaced!r:36}  → parsed={parsed}")
        passed += int(ok)
    console.print(f"\n  [bold]{passed}/{len(targets)} round-trips matched[/]")


def demo_multilingual_tool() -> None:
    section("Demo 6 · The multilingual @tool — text2num for EU langs, rule for TR")
    cases = [
        ("one thousand nine hundred eighty seven", "en"),
        ("mille neuf cent quatre-vingt-sept",      "fr"),
        ("mil novecientos ochenta y siete",        "es"),
        ("neunzehnhundertsiebenundachtzig",        "de"),
        ("bin dokuz yüz seksen yedi",              "tr"),
        ("twenty five",                            "jp"),   # unsupported locale
    ]
    for text, loc in cases:
        result = parse_spoken_number.invoke({"text": text, "locale": loc})
        ok_marker = "[green]✓[/]" if result.get("ok") else "[red]✗[/]"
        console.print(f"  {ok_marker} ({loc}) {text!r:48}  →  {result}")


def main() -> None:
    args = set(sys.argv[1:])
    if not args or "--all" in args:
        args = {"--numbers", "--dob", "--others", "--full", "--fuzzy",
                "--multilingual", "--roundtrip"}
    if "--numbers" in args:      demo_single_numbers()
    if "--dob" in args:          demo_dob_segmenter()
    if "--others" in args:       demo_other_locales()
    if "--full" in args:         demo_full_pipeline()
    if "--fuzzy" in args:        demo_fuzzy_partial_matching()
    if "--multilingual" in args: demo_multilingual_tool()
    if "--roundtrip" in args:    demo_round_trip()


if __name__ == "__main__":
    main()
