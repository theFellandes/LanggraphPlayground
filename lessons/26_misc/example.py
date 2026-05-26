"""Lesson 26 · The miscellaneous lesson — things you'll need but didn't
get their own dedicated lesson elsewhere.

Five high-severity topics, three with runnable demos (no API key needed):

  demo_token_counting()    — count tokens before the call (tiktoken)
  demo_cost_estimate()      — turn tokens into dollars (per-provider pricing)
  demo_caching()           — LangChain's InMemoryCache speedup with a FakeListChatModel
  demo_eval_framework()    — pytest-style eval suite with a FakeListChatModel as oracle
  demo_self_correction()   — small Reflexion-style retry loop

The remaining two topics (MCP, reranking) need running external services
(an MCP server, the Cohere API). README walks through them with code.

Run:
    uv run python -m lessons.26_misc.example
    uv run python -m lessons.26_misc.example --tokens
    uv run python -m lessons.26_misc.example --cost
    uv run python -m lessons.26_misc.example --caching
    uv run python -m lessons.26_misc.example --eval
    uv run python -m lessons.26_misc.example --self-correct
"""

from __future__ import annotations

import sys
import time
from typing import Callable

from shared.pretty import console, section


# ───────────────────────────────────────────────────────────────────────────
# 1 · Token counting with tiktoken
# ───────────────────────────────────────────────────────────────────────────
# tiktoken is the canonical OpenAI tokenizer (BPE). The same encoder works
# for GPT-4o and most modern OpenAI models. For Anthropic, use
# anthropic.beta.messages.count_tokens for exact counts.

def count_tokens_openai(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in `text` using the tokenizer for `model`."""
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for newer models tiktoken doesn't recognise yet
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def demo_token_counting() -> None:
    section("Demo 1 · count tokens before you call the model (tiktoken)")
    samples = [
        ("English",       "The quick brown fox jumps over the lazy dog."),
        ("Turkish",       "Pijamalı hasta yağız şoföre çabucak güvendi."),
        ("Code (Python)", "def hello():\n    return 'world'\n"),
        ("Long",          "Lorem ipsum dolor sit amet, " * 50),
        ("Empty",         ""),
    ]
    for label, text in samples:
        tokens = count_tokens_openai(text)
        chars = len(text)
        ratio = chars / max(tokens, 1)
        console.print(f"  [cyan]{label:14}[/]  tokens={tokens:5}  chars={chars:5}  chars/token={ratio:5.2f}")
    console.print("\n  [dim]English ~4 chars/token, Turkish ~2 chars/token, code ~3-4. Plan budgets accordingly.[/]")


# ───────────────────────────────────────────────────────────────────────────
# 2 · Cost estimation — tokens × price = dollars
# ───────────────────────────────────────────────────────────────────────────
# Pricing here is approximate May 2026; ALWAYS re-check vendor pricing
# pages before quoting numbers in production.
PRICING_USD_PER_M_TOKENS = {
    # (input, output) per 1M tokens
    "claude-sonnet-4-6":   (3.00, 15.00),
    "claude-haiku-4-5":    (0.80,  4.00),
    "gpt-4.1":             (5.00, 15.00),
    "gpt-4o-mini":         (0.15,  0.60),
    "gpt-4o":              (5.00, 15.00),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the request cost in USD. Returns 0 if model is unknown."""
    if model not in PRICING_USD_PER_M_TOKENS:
        return 0.0
    p_in, p_out = PRICING_USD_PER_M_TOKENS[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


def demo_cost_estimate() -> None:
    section("Demo 2 · estimate cost per call & per million users")
    workloads = [
        ("simple Q&A",      300,  150),
        ("RAG with 4 chunks", 2_000,  500),
        ("agent w/ 3 tool turns", 8_000, 1_500),
        ("long chat (50 turns)", 30_000, 5_000),
    ]
    console.print(f"  [bold]{'workload':28} {'in':>7} {'out':>7}   " +
                  "  ".join(f"{m:>14}" for m in PRICING_USD_PER_M_TOKENS))
    for label, in_t, out_t in workloads:
        row = f"  {label:28} {in_t:>7} {out_t:>7}   "
        for m in PRICING_USD_PER_M_TOKENS:
            usd = estimate_cost_usd(m, in_t, out_t)
            row += f"   ${usd:>9.4f}  "
        console.print(row)
    console.print("\n  [dim]At 1 M users with one 'agent w/ 3 tool turns' each:[/]")
    for m in PRICING_USD_PER_M_TOKENS:
        total = estimate_cost_usd(m, 8000, 1500) * 1_000_000
        console.print(f"    {m:25}  $ {total:>10,.0f}")


# ───────────────────────────────────────────────────────────────────────────
# 3 · Caching — LangChain's global InMemoryCache + per-provider native cache
# ───────────────────────────────────────────────────────────────────────────
# Two distinct caching layers:
#   A · LangChain-level cache: caches the entire (model, prompt) → response
#       round-trip. Works with any provider. Use InMemoryCache (dev) or
#       SQLiteCache (persistent).
#   B · Provider-level prompt caching: Anthropic and OpenAI cache prefix
#       tokens server-side. Anthropic exposes explicit control via
#       cache_control headers; LangChain's AnthropicPromptCachingMiddleware
#       handles this automatically.

def demo_caching() -> None:
    section("Demo 3 · LangChain InMemoryCache — speedup with a fake (slow) model")
    from langchain_core.caches import InMemoryCache
    from langchain_core.globals import set_llm_cache
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    # Enable the global LLM cache
    set_llm_cache(InMemoryCache())

    class SlowFakeModel(FakeListChatModel):
        """Simulates a 0.5-second network round-trip."""
        def _call(self, *args, **kwargs):
            time.sleep(0.5)
            return super()._call(*args, **kwargs)

    llm = SlowFakeModel(responses=["The answer is 42."])
    prompt = "What is the answer?"

    # First call — cache miss; actually 'calls' the model
    t0 = time.perf_counter()
    r1 = llm.invoke(prompt)
    t_miss = time.perf_counter() - t0

    # Second call — cache hit; should be ~0 seconds
    t0 = time.perf_counter()
    r2 = llm.invoke(prompt)
    t_hit = time.perf_counter() - t0

    console.print(f"  miss: {t_miss*1000:7.1f} ms  →  {r1.content!r}")
    console.print(f"  hit:  {t_hit*1000:7.1f} ms  →  {r2.content!r}")
    speedup = t_miss / max(t_hit, 1e-6)
    console.print(f"\n  [bold]speedup ≈ {speedup:.0f}×[/]  (cache returns response without re-invoking)")

    set_llm_cache(None)  # cleanup


# ───────────────────────────────────────────────────────────────────────────
# 4 · Evaluation framework — the single highest-leverage missing piece
# ───────────────────────────────────────────────────────────────────────────
# An eval is just: { input, expected, scorer }. Run the system on each
# input, run scorer(actual, expected) → pass/fail, aggregate.
# Three scorer flavours: exact match, schema match, LLM-as-judge.

EvalCase = tuple[str, str, str]  # (label, input, expected)


def exact_match(actual: str, expected: str) -> bool:
    return actual.strip() == expected.strip()


def contains_match(actual: str, expected: str) -> bool:
    return expected.lower() in actual.lower()


def schema_match(actual: str, expected: str) -> bool:
    """Check that JSON output has the required keys."""
    import json
    try:
        got = json.loads(actual)
        required = json.loads(expected)
        return all(k in got for k in required)
    except json.JSONDecodeError:
        return False


def run_eval_suite(
    system_under_test: Callable[[str], str],
    cases: list[EvalCase],
    scorer: Callable[[str, str], bool] = exact_match,
) -> dict:
    """Run a list of eval cases through the system. Return {passed, total, failures}."""
    failures = []
    for label, inp, expected in cases:
        try:
            actual = system_under_test(inp)
        except Exception as e:
            actual = f"<error: {type(e).__name__}: {e}>"
        if not scorer(actual, expected):
            failures.append({"label": label, "input": inp,
                             "expected": expected, "got": actual})
    return {"passed": len(cases) - len(failures), "total": len(cases),
            "failures": failures}


def demo_eval_framework() -> None:
    section("Demo 4 · a working eval suite (in 40 lines)")
    # System under test — a deliberately-flawed "summariser"
    def buggy_summariser(text: str) -> str:
        if "Paris" in text: return "Paris is the capital of France."
        if "Berlin" in text: return "Berlin is in France."   # wrong (bug)
        if "Tokyo" in text: return "Tokyo is the capital of Japan."
        return "Unknown."

    cases: list[EvalCase] = [
        ("paris_correct",  "Paris is a city in France.",  "Paris is the capital of France."),
        ("berlin_correct", "Berlin is a city in Germany.", "Berlin is the capital of Germany."),  # will fail
        ("tokyo_correct",  "Tokyo is a city in Japan.",   "Tokyo is the capital of Japan."),
    ]
    result = run_eval_suite(buggy_summariser, cases, scorer=exact_match)
    console.print(f"  [bold]{result['passed']}/{result['total']} passed[/]")
    for f in result["failures"]:
        console.print(f"  [red]✗ {f['label']}[/]")
        console.print(f"     expected: {f['expected']!r}")
        console.print(f"     got:      {f['got']!r}")
    console.print("\n  [dim]Real evals: 30-50 cases per task, run on every prompt change, " +
                  "track pass-rate in CI. Use LangSmith / promptfoo / inspect-ai for the framework.[/]")


# ───────────────────────────────────────────────────────────────────────────
# 5 · Self-correction (Reflexion-lite) — retry with the previous error
# ───────────────────────────────────────────────────────────────────────────
# This is the manual version of what lesson 21 / lesson 24 already do via
# Pydantic field_validator. The pattern: run → if invalid, build a
# corrective prompt that includes the previous output AND the error,
# retry up to N times.

def self_correcting_call(
    generator: Callable[[str], str],
    validator: Callable[[str], tuple[bool, str]],   # (is_valid, error_message)
    initial_prompt: str,
    max_attempts: int = 3,
) -> tuple[str | None, int]:
    """Reflexion-lite loop. Returns (final_output_or_None, attempts_used)."""
    prompt = initial_prompt
    for attempt in range(1, max_attempts + 1):
        output = generator(prompt)
        ok, error = validator(output)
        if ok:
            return output, attempt
        # Build the corrective prompt — include the previous attempt + error
        prompt = (f"{initial_prompt}\n\n"
                  f"Previous attempt: {output!r}\n"
                  f"Error: {error}\n"
                  f"Try again, fixing the error this time.")
    return None, max_attempts


def demo_self_correction() -> None:
    section("Demo 5 · self-correction loop (Reflexion-lite)")
    # Fake generator that gets it right on the 2nd attempt
    attempt_counter = [0]
    def fake_gen(prompt: str) -> str:
        attempt_counter[0] += 1
        if attempt_counter[0] == 1: return "twenty-five"           # wrong shape
        return "1987-04-05"                                          # right shape

    def is_iso_date(text: str) -> tuple[bool, str]:
        import re
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text.strip()):
            return True, ""
        return False, f"expected ISO 8601 (YYYY-MM-DD), got {text!r}"

    result, attempts = self_correcting_call(
        generator=fake_gen,
        validator=is_iso_date,
        initial_prompt="Give me a date in ISO 8601 format.",
    )
    console.print(f"  result after {attempts} attempt(s): {result!r}")
    console.print("\n  [dim]Real systems: cap at 3 attempts, log each retry for monitoring, " +
                  "graduate to Instructor / Marvin which automate this.[/]")


# ───────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = set(sys.argv[1:])
    run_all = not args
    if run_all or "--tokens" in args:        demo_token_counting()
    if run_all or "--cost" in args:          demo_cost_estimate()
    if run_all or "--caching" in args:       demo_caching()
    if run_all or "--eval" in args:          demo_eval_framework()
    if run_all or "--self-correct" in args:  demo_self_correction()


if __name__ == "__main__":
    main()
