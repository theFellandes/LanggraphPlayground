"""Lesson 23 · Date computation tools + localized output.

The OUTPUT side of the date problem (lesson 21 was the input side).

Three demos in this file:

  demo_localization()       — pure Python; format an ISO date across 8 locales
                              ("23 Mayıs 2026", "May 23, 2026", "23. Mai 2026", ...)
  demo_calendars()           — Persian Jalali + Islamic Hijri + Japanese era
  demo_date_tools()          — agentic version: today_iso(), compute_date(), parse_relative()
                              gives the LLM deterministic date arithmetic instead of letting it guess
                              (needs an API key)

Run:
    uv run python -m lessons.23_date_localization.example                    # all three
    uv run python -m lessons.23_date_localization.example --localization     # just demo 1
    uv run python -m lessons.23_date_localization.example --calendars        # just demo 2
    uv run python -m lessons.23_date_localization.example --tools            # just demo 3
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from typing import Literal

from babel.dates import format_date
from langchain_core.tools import tool

from shared.pretty import console, section

TODAY = date.today()


# ───────────────────────────────────────────────────────────────────────────
# Reusable helper — the canonical "localize this date" function
# ───────────────────────────────────────────────────────────────────────────
FormatName = Literal["short", "medium", "long", "full"]


def localize_date(
    d: date,
    locale: str = "en_US",
    fmt: FormatName | str = "long",
) -> str:
    """Render an ISO date in the user's locale.

    Args:
        d: A datetime.date (your internal representation, always ISO 8601).
        locale: BCP-47-style locale code: en_US, tr_TR, de_DE, fa_IR, ar_SA, ja_JP, ...
        fmt: One of CLDR's named formats ("short" / "medium" / "long" / "full")
            or a custom CLDR pattern string like "EEEE d MMMM y".

    Returns:
        Localized string. Examples:
          locale="en_US", fmt="long"  → "May 23, 2026"
          locale="tr_TR", fmt="long"  → "23 Mayıs 2026"
          locale="de_DE", fmt="long"  → "23. Mai 2026"
          locale="fa_IR", fmt="long"  → "۲ خرداد ۱۴۰۵"   (uses Persian calendar!)
          locale="ja_JP", fmt="long"  → "2026年5月23日"

    Note: ``babel`` automatically picks the right calendar for the locale —
    Persian/Farsi locales (``fa_IR``, ``fa_AF``) return Jalali dates,
    Japanese ``ja_JP`` includes era markers when ``fmt="full"``, etc.
    """
    return format_date(d, format=fmt, locale=locale)


# ───────────────────────────────────────────────────────────────────────────
# Demo 1 — localized output across locales (the user's main question)
# ───────────────────────────────────────────────────────────────────────────
def demo_localization() -> None:
    section("Demo 1 · the same ISO date across 8 locales")
    target = date(2026, 5, 23)
    console.print(f"[dim]Internal representation (always ISO 8601): {target.isoformat()}[/]\n")

    locales: list[tuple[str, str]] = [
        ("en_US", "American English"),
        ("en_GB", "British English"),
        ("tr_TR", "Turkish"),
        ("de_DE", "German"),
        ("fr_FR", "French"),
        ("es_ES", "Spanish"),
        ("ar_SA", "Arabic (Saudi)"),
        ("fa_IR", "Persian (Iran) — uses Jalali calendar!"),
        ("ja_JP", "Japanese"),
        ("zh_CN", "Chinese (Simplified)"),
    ]
    for code, label in locales:
        try:
            console.print(
                f"  [cyan]{code:8}[/]  [{label:42}]  "
                f"{localize_date(target, locale=code, fmt='long')}"
            )
        except Exception as e:
            console.print(f"  [red]{code:8}[/]  failed: {e}")

    section("Demo 1b · same date, four format levels (en_US vs tr_TR)")
    for fmt in ("short", "medium", "long", "full"):
        en = localize_date(target, locale="en_US", fmt=fmt)
        tr = localize_date(target, locale="tr_TR", fmt=fmt)
        console.print(f"  [cyan]{fmt:8}[/]  en_US={en!r:30}  tr_TR={tr!r}")

    section("Demo 1c · custom CLDR pattern (a Turkish full sentence)")
    custom_tr = localize_date(target, locale="tr_TR", fmt="EEEE, d MMMM y")
    console.print(f"  Pattern \"EEEE, d MMMM y\" → {custom_tr!r}")
    custom_en = localize_date(target, locale="en_US", fmt="EEEE, MMMM d, y")
    console.print(f"  Pattern \"EEEE, MMMM d, y\" → {custom_en!r}")


# ───────────────────────────────────────────────────────────────────────────
# Demo 2 — calendar conversions (Jalali, Hijri, Japanese era)
# ───────────────────────────────────────────────────────────────────────────
def demo_calendars() -> None:
    section("Demo 2 · calendar conversions for 2026-05-23")
    target = date(2026, 5, 23)

    # ── Persian Jalali ────────────────────────────────────────────────────
    try:
        import jdatetime
        jalali = jdatetime.date.fromgregorian(date=target)
        console.print(
            f"  [cyan]Jalali (Persian)[/]   {jalali.strftime('%Y/%m/%d')}  "
            f"({jalali.year}/{jalali.month}/{jalali.day} in Jalali calendar)"
        )
        # Persian-localized formatting via babel automatically picks Jalali for fa_IR
        console.print(
            f"  [cyan]via babel (fa_IR)[/]  "
            f"{format_date(target, format='long', locale='fa_IR')}  ← same date, locale-formatted"
        )
    except ImportError:
        console.print("  [yellow]skipped — jdatetime not installed[/]")

    # ── Islamic Hijri ─────────────────────────────────────────────────────
    try:
        from hijridate import Gregorian
        hijri = Gregorian(target.year, target.month, target.day).to_hijri()
        console.print(
            f"  [cyan]Hijri (Islamic)[/]    "
            f"{hijri.year:04d}-{hijri.month:02d}-{hijri.day:02d}  ({hijri.month_name()})"
        )
    except ImportError:
        console.print("  [yellow]skipped — hijridate not installed[/]")

    # ── Japanese era ──────────────────────────────────────────────────────
    # `babel` knows about Japanese eras when you use a full format.
    jpn = format_date(target, format="full", locale="ja_JP_TRADITIONAL")
    console.print(f"  [cyan]Japanese era[/]      {jpn}")


# ───────────────────────────────────────────────────────────────────────────
# Demo 3 — date-arithmetic tools for an LLM agent
# (this is what the user meant by "queries that require date computation")
# ───────────────────────────────────────────────────────────────────────────
@tool
def today_iso() -> str:
    """Return today's date as an ISO 8601 string (YYYY-MM-DD)."""
    return date.today().isoformat()


@tool
def add_days(iso_date: str, days: int) -> str:
    """Return iso_date shifted by `days`. Negative for the past."""
    d = date.fromisoformat(iso_date)
    return (d + timedelta(days=days)).isoformat()


@tool
def add_months(iso_date: str, months: int) -> str:
    """Return iso_date shifted by `months`. Clamps to last day of month if needed."""
    d = date.fromisoformat(iso_date)
    new_month = d.month + months
    year_delta, new_month0 = divmod(new_month - 1, 12)
    new_year = d.year + year_delta
    new_month = new_month0 + 1
    # clamp day
    import calendar
    last_day = calendar.monthrange(new_year, new_month)[1]
    return date(new_year, new_month, min(d.day, last_day)).isoformat()


@tool
def parse_relative_date(text: str) -> str:
    """Parse a relative English / multilingual date expression into ISO 8601.

    Examples: "5 days ago", "next Tuesday", "3 weeks from now",
              "yesterday", "5 mayıs 1987" (Turkish).
    Returns ISO 8601 or "" if unparseable.
    """
    import dateparser
    d = dateparser.parse(text, settings={"RELATIVE_BASE": datetime.combine(date.today(), datetime.min.time())})
    return d.date().isoformat() if d else ""


@tool
def date_diff_days(iso_a: str, iso_b: str) -> int:
    """Return (iso_a - iso_b) in days. Positive if iso_a is later."""
    a = date.fromisoformat(iso_a)
    b = date.fromisoformat(iso_b)
    return (a - b).days


@tool
def render_localized(iso_date: str, locale: str = "en_US") -> str:
    """Render an ISO date in the user's locale. Use this for the FINAL user-facing reply.

    Example: render_localized("2026-05-23", "tr_TR") → "23 Mayıs 2026"
    """
    return localize_date(date.fromisoformat(iso_date), locale=locale, fmt="long")


DATE_TOOLS = [
    today_iso,
    add_days,
    add_months,
    parse_relative_date,
    date_diff_days,
    render_localized,
]


def demo_date_tools() -> None:
    section("Demo 3 · agent with deterministic date tools (needs API key)")
    try:
        from langchain.agents import create_agent
        from shared.llm import get_llm
    except Exception as e:
        console.print(f"[red]skipped — could not import LLM stack: {e}[/]")
        return

    agent = create_agent(
        model=get_llm(),
        tools=DATE_TOOLS,
        system_prompt=(
            "You answer date / time questions for users. Rules:\n"
            "- NEVER compute dates yourself. ALWAYS call a tool.\n"
            "- Use today_iso() before any relative-date question.\n"
            "- For the FINAL user-facing answer, call render_localized() with the user's locale.\n"
        ),
    )

    questions: list[tuple[str, str]] = [
        ("en_US", "What's today's date?"),
        ("tr_TR", "Bugün hangi tarih? Türkçe formatta yanıtla."),
        ("en_US", "What date will it be 3 weeks from today?"),
        ("de_DE", "Wie viele Tage sind es bis Weihnachten? (Antwort auf Deutsch)"),
        ("tr_TR", "5 hafta sonra hangi tarih olacak?"),
    ]
    for locale, q in questions:
        console.print(f"\n[bold cyan]Q ({locale}):[/] {q}")
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": q}]})
            console.print(f"[bold green]A:[/] {result['messages'][-1].content}")
        except Exception as e:
            console.print(f"[red]error: {e}[/]")


# ───────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = set(sys.argv[1:])
    run_all = not (args & {"--localization", "--calendars", "--tools"})

    if run_all or "--localization" in args:
        demo_localization()
    if run_all or "--calendars" in args:
        demo_calendars()
    if run_all or "--tools" in args:
        demo_date_tools()


if __name__ == "__main__":
    main()
