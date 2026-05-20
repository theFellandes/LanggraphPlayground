"""Lesson 25 · Tool design patterns for reliable LLM agents.

Five worked tool shapes — copy any of them as a starting point. All run
without an API key (the demos call the tools directly, not through an LLM).

  1. Read-only data tool       — `lookup_order` with structured output
  2. Computation + metadata    — `parse_birth_date` returns value + confidence
  3. Side-effect + HITL gate   — `process_refund` raises a recoverable error
  4. Router / dispatch         — `parse_number` routes to N implementations
  5. Wrapper / enricher        — `cached_lookup` decorates another tool

Run:
    uv run python -m lessons.25_tool_design.example
"""

from __future__ import annotations

import sys
import time
from datetime import date
from typing import Annotated

import dateparser
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from shared.pretty import console, section


# ───────────────────────────────────────────────────────────────────────────
# Pattern 1 · Read-only data tool with structured output
# ───────────────────────────────────────────────────────────────────────────
# Returns a STRUCTURED dict, not a bare value. The shape itself documents
# the result — the agent doesn't have to guess what fields exist.

@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by its id.

    Args:
        order_id: The order identifier exactly as the customer gave it
                  (e.g., 'ABC-1234'). Do not modify, normalise, or guess.

    Returns:
        {ok: True,  order: {...}}            on success
        {ok: False, error: str, suggestion?: str}  on failure

    The `suggestion` field on failure is what the agent should mention to
    the user — never invent a suggestion if one isn't returned.
    """
    fake_db = {
        "ABC-1": {"id": "ABC-1", "amount": 42.0,  "status": "shipped"},
        "ABC-2": {"id": "ABC-2", "amount": 250.0, "status": "delivered"},
    }
    if order_id in fake_db:
        return {"ok": True, "order": fake_db[order_id]}
    return {
        "ok": False,
        "error": f"no order matching {order_id!r}",
        "suggestion": "Ask the customer to double-check the order id.",
    }


# ───────────────────────────────────────────────────────────────────────────
# Pattern 2 · Computation tool that returns CONFIDENCE + metadata
# ───────────────────────────────────────────────────────────────────────────
# The deep lesson: returning a bare value tells the agent "trust me".
# Returning value + confidence + alternatives tells the agent ENOUGH to
# decide its next action by itself — no need to send the whole
# conversation back to a second LLM call.

@tool
def parse_birth_date(text: str) -> dict:
    """Parse a date-of-birth phrase into ISO 8601.

    Args:
        text: The customer's spoken or written date phrase EXACTLY as said.
              Do not paraphrase, summarise, or pre-normalise.

    Returns:
        {
          ok:          bool,
          iso:         "YYYY-MM-DD" | None,
          confidence:  float 0-1,
          engine:      str,              # which parser succeeded
          alternatives: [{iso, score}],  # other candidates the agent can ask about
          warnings:    [str],            # ambiguities to flag to the user
        }

    Agent decision policy (the agent's prompt should encode this):
      confidence >= 0.95     → use `iso` directly
      0.80 <= conf < 0.95    → ask the user to confirm: "Did you mean X?"
      confidence < 0.80      → ask the user to repeat the date
    """
    parsed = dateparser.parse(
        text,
        settings={"STRICT_PARSING": False, "DATE_ORDER": "DMY",
                  "RELATIVE_BASE": _now()},
    )
    if parsed is None:
        return {
            "ok": False, "iso": None, "confidence": 0.0,
            "engine": "dateparser", "alternatives": [],
            "warnings": [f"could not parse {text!r}"],
        }
    iso = parsed.date().isoformat()
    # Score: 1.0 for clean parses, lower if we had to recover from typos / ambiguity.
    # In a real implementation you'd use rapidfuzz against the original spans.
    confidence = 0.95 if len(text.split()) <= 3 else 0.80
    warnings: list[str] = []
    if parsed.date() > date.today():
        warnings.append("date is in the future — likely wrong")
        confidence *= 0.5
    return {
        "ok": True,
        "iso": iso,
        "confidence": confidence,
        "engine": "dateparser",
        "alternatives": [],
        "warnings": warnings,
    }


def _now():
    from datetime import datetime
    return datetime.combine(date.today(), datetime.min.time())


# ───────────────────────────────────────────────────────────────────────────
# Pattern 3 · Side-effect tool with a recoverable-error contract
# ───────────────────────────────────────────────────────────────────────────
# RAISE for unrecoverable infrastructure problems. RETURN error dict for
# policy violations the agent can route around (this is what makes the
# agent loop self-correct gracefully).

class RefundDeniedError(Exception):
    """Unrecoverable — agent should NOT retry; surface to the user."""


@tool
def process_refund(order_id: str, amount: float, reason: str = "") -> dict:
    """Issue a refund. Refunds > $200 require human approval.

    Returns:
        {ok: True,  refund_id: str}                          when issued
        {ok: False, error: "needs_human_approval", ...}      when over policy cap
        {ok: False, error: "amount_invalid",       ...}      when caller is wrong

    Raises:
        RefundDeniedError: For unrecoverable policy violations (fraud, blocked
        account, etc.). The agent must NOT retry; the user must be told.
    """
    if amount <= 0:
        # Caller (the LLM) is wrong — recoverable; the LLM will fix and retry.
        return {"ok": False, "error": "amount_invalid",
                "detail": f"amount must be positive, got {amount}"}

    if amount > 200:
        # Policy block — recoverable; the agent escalates to HITL middleware.
        return {"ok": False, "error": "needs_human_approval",
                "amount": amount,
                "suggestion": "Pause and ask the operator to /approve."}

    # Side effect
    return {"ok": True, "refund_id": f"ref_{order_id}_{int(time.time())}"}


# ───────────────────────────────────────────────────────────────────────────
# Pattern 4 · Router / dispatch tool
# ───────────────────────────────────────────────────────────────────────────
# ONE tool surface, N implementations behind it. The agent picks the kind
# via a `locale` (or `mode`, `provider`, `kind`) parameter.
# Failure case (unsupported locale) returns a list of options — the agent
# uses that to retry with a valid value.

# Design choice: `locale` is typed `str`, not `Literal[...]`. A Literal
# would have Pydantic REJECT unknown locales at framework level — the LLM
# then gets a terse Pydantic ValidationError. By keeping it `str`, our
# function can return a structured error (with `supported_locales`) that
# the LLM can act on. Trade-off: lose schema-time validation; gain
# explainability. For router tools, the latter wins.

@tool
def parse_number(
    text: Annotated[str, "Number phrase exactly as the user said it."],
    locale: Annotated[str, "ISO language code (en/fr/es/de/tr). The tool returns the supported list on error."],
) -> dict:
    """Parse a spoken-form number into an integer (router for multiple engines)."""
    if locale == "tr":
        # In a real impl, call the lesson 24 Turkish parser
        return {"ok": True, "value": _toy_turkish(text), "engine": "turkish_rule_parser"}
    if locale in ("en", "fr", "es", "de"):
        try:
            from text_to_num import text2num
            return {"ok": True, "value": text2num(text, locale), "engine": "text2num"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return {"ok": False, "error": f"unsupported locale {locale!r}",
            "supported_locales": ["en", "fr", "es", "de", "tr"]}


def _toy_turkish(text: str) -> int | None:
    return {"yirmi beş": 25, "doksan yedi": 97, "bin dokuz yüz seksen yedi": 1987}.get(text.lower())


# ───────────────────────────────────────────────────────────────────────────
# Pattern 5 · Wrapper / enricher — decorates another tool, adds metadata
# ───────────────────────────────────────────────────────────────────────────
# CAVEAT — a *generic* higher-order @tool wrapper is fragile: LangChain
# inspects the wrapper's signature to build the tool schema, and a
# generic `**kwargs` wrapper loses the inner tool's param names.
#
# Cleaner real-world pattern: write a domain-specific cached version
# that re-declares the signature explicitly. More typing, but the
# schema the LLM sees is identical to the original tool.

_cache: dict[str, dict] = {}


@tool
def cached_lookup_order(order_id: str) -> dict:
    """Cached version of `lookup_order` — identical args, identical return shape.

    Adds `from_cache: bool` to the return so the agent can see whether
    it hit cache or paid the round-trip. Useful for read-only tools the
    agent is likely to call multiple times in one loop.
    """
    key = f"order:{order_id}"
    if key in _cache:
        return {**_cache[key], "from_cache": True}
    result = lookup_order.invoke({"order_id": order_id})
    if isinstance(result, dict) and result.get("ok"):
        _cache[key] = result
    return {**result, "from_cache": False}


# Alternative: if you really need a generic wrapper, build the StructuredTool
# yourself from the inner tool's `args_schema`:
#
#     from langchain_core.tools import StructuredTool
#     def with_cache_generic(inner: StructuredTool) -> StructuredTool:
#         def _wrapped(**kw):
#             key = repr(sorted(kw.items()))
#             if key in _cache: return {**_cache[key], "from_cache": True}
#             out = inner.invoke(kw)
#             if isinstance(out, dict) and out.get("ok"): _cache[key] = out
#             return {**out, "from_cache": False}
#         return StructuredTool.from_function(
#             func=_wrapped,
#             name=inner.name + "_cached",
#             description=inner.description,
#             args_schema=inner.args_schema,   # ← reuse the original schema
#         )


# ───────────────────────────────────────────────────────────────────────────
# DEMOS
# ───────────────────────────────────────────────────────────────────────────
def demo_read_only_tool() -> None:
    section("Pattern 1 · read-only data tool — `lookup_order`")
    for oid in ("ABC-1", "ABC-2", "DOES-NOT-EXIST"):
        result = lookup_order.invoke({"order_id": oid})
        console.print(f"  lookup_order({oid!r}) → {result}")


def demo_computation_with_metadata() -> None:
    section("Pattern 2 · computation tool with confidence + metadata")
    for text in ("1987-04-05", "April 5, 1987", "5 nisan 1987", "lol no date here", "2099-01-01"):
        result = parse_birth_date.invoke({"text": text})
        tier = ("ACCEPT" if result["confidence"] >= 0.95
                else "CONFIRM" if result["confidence"] >= 0.80
                else "REJECT")
        console.print(f"  [{tier:7}] {text!r:30} → conf={result['confidence']:.2f}  "
                      f"iso={result['iso']}  warnings={result['warnings']}")


def demo_side_effect_with_hitl() -> None:
    section("Pattern 3 · side-effect tool — recoverable vs unrecoverable")
    cases = [
        ("ABC-1",  50.0,   "wrong size"),       # ok
        ("ABC-2",  250.0,  "damaged"),          # needs approval
        ("ABC-3",  -5.0,   "n/a"),              # invalid amount — LLM's fault
    ]
    for oid, amt, reason in cases:
        result = process_refund.invoke({"order_id": oid, "amount": amt, "reason": reason})
        tag = "[green]ok[/]" if result["ok"] else f"[yellow]{result.get('error')}[/]"
        console.print(f"  process_refund({oid}, ${amt}) → {tag}  detail={result}")


def demo_router() -> None:
    section("Pattern 4 · router / dispatch — `parse_number`")
    cases = [
        ("one thousand nine hundred eighty seven", "en"),
        ("mille neuf cent quatre-vingt-sept",      "fr"),
        ("yirmi beş",                              "tr"),
        ("twenty five",                            "jp"),   # unsupported
    ]
    for text, loc in cases:
        result = parse_number.invoke({"text": text, "locale": loc})
        marker = "[green]✓[/]" if result.get("ok") else "[red]✗[/]"
        console.print(f"  {marker} parse_number({text!r}, {loc!r}) → {result}")


def demo_enricher_with_cache() -> None:
    section("Pattern 5 · wrapper / enricher — `with_cache`")
    _cache.clear()
    for _ in range(3):
        result = cached_lookup_order.invoke({"order_id": "ABC-1"})
        console.print(f"  cached_lookup_order('ABC-1') → "
                      f"from_cache={result.get('from_cache')}  ok={result.get('ok')}")


def main() -> None:
    args = set(sys.argv[1:])
    run_all = not args
    if run_all or "--read-only" in args:        demo_read_only_tool()
    if run_all or "--computation" in args:      demo_computation_with_metadata()
    if run_all or "--side-effect" in args:      demo_side_effect_with_hitl()
    if run_all or "--router" in args:           demo_router()
    if run_all or "--enricher" in args:         demo_enricher_with_cache()


if __name__ == "__main__":
    main()
