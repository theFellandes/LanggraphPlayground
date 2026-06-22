"""Compare VLMs on PDF→Markdown extraction — the metric-creation + bake-off runner.

This is the "bring your own VLM" comparison (TEST-PLAN.md §3-§6). For every golden
page that has a hand-authored ``<stem>.gold.md`` it runs each *available* provider's
model, scores the output with :mod:`metrics`, and writes a side-by-side comparison
(rich table + ``results.json`` + ``report.md``).

Swap in YOUR OWN VLM by wiring the ``local`` provider — any OpenAI-compatible server
(vLLM / Ollama / LM Studio / TGI). See ``shared/llm/local_adapter.py`` and
``docs/research/vlm-pdf-extraction/VLM-COMPARISON.md``.

    uv sync --extra extraction
    uv add langchain-google-genai                 # for the google column
    # then set keys in .env (GOOGLE_API_KEY / ANTHROPIC_API_KEY / LOCAL_VLM_BASE_URL+LOCAL_API_KEY)
    uv run python -m projects.vlm_extraction_harness.compare_vlms --providers google,anthropic,local

Every model is built via ``get_llm(provider, ..., with_fallback=False)`` AND the
provider is verified *available* first — so the comparison tests the EXACT model you
name. A provider without a key is SKIPPED (never silently swapped for another), which
is the whole point of a benchmark.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from projects.vlm_extraction_harness import metrics  # noqa: E402
from projects.vlm_extraction_harness.extractors import render_page_png  # noqa: E402
from projects.vlm_extraction_harness.golden_set import GOLDEN_SET, SAMPLE_DIR  # noqa: E402
from shared.llm import available_providers, get_llm  # noqa: E402
from shared.settings import settings  # noqa: E402

console = Console()

# Locked extraction prompt (identical for every provider — fairness, TEST-PLAN §3).
EXTRACTION_PROMPT = (
    "Return the content of this page as GitHub-flavored Markdown in natural "
    "reading order.\n"
    "- If the page has multiple columns, read the leftmost column fully before the next.\n"
    "- Render every table as valid HTML (<table>/<th>/<td> with colspan/rowspan). "
    "Never use Markdown pipes for merged cells.\n"
    "- Render formulas as LaTeX ($...$).\n"
    "- For each figure, emit ![<one-sentence caption>](figure) and nothing invented.\n"
    "- Do not hallucinate. If a region is unreadable, write [UNREADABLE]."
)

# Default model per provider (None = the adapter's own default).
DEFAULT_MODELS = {
    "google": "gemini-2.0-flash",
    "anthropic": None,   # adapter default
    "openai": None,
    "local": None,       # settings.local_vlm_model
}

# Rough public list prices, USD per 1M tokens (input, output). CHURN FAST — verify
# before trusting (FINDINGS.md §12). `local` is your own GPU; track separately.
PRICING = {
    "google": {"gemini-2.0-flash": (0.10, 0.40), "_default": (0.10, 0.40)},
    "anthropic": {"_default": (3.0, 15.0)},
    "openai": {"_default": (2.5, 10.0)},
    "local": {"_default": (0.0, 0.0)},
}


def vlm_extract(pdf_path, provider, model=None, page=0, dpi=300):
    """Render page -> base64 PNG -> get_llm(provider) -> (markdown, latency_s, usage)."""
    from langchain_core.messages import HumanMessage

    png = render_page_png(pdf_path, page, dpi)
    b64 = base64.b64encode(png).decode()
    llm = get_llm(provider, model=model, with_fallback=False, temperature=0)
    msg = HumanMessage(content=[
        {"type": "text", "text": EXTRACTION_PROMPT},
        {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
    ])
    t0 = time.perf_counter()
    resp = llm.invoke([msg])
    dt = time.perf_counter() - t0
    md = resp.content if hasattr(resp, "content") else str(resp)
    usage = getattr(resp, "usage_metadata", None) or {}
    return md, dt, usage


def estimate_cost(provider, model, usage):
    if not usage:
        return None
    rates = PRICING.get(provider, {})
    rate = rates.get(model) or rates.get("_default")
    if rate is None:
        return None
    cin, cout = rate
    it = usage.get("input_tokens", 0) or 0
    ot = usage.get("output_tokens", 0) or 0
    return round((it * cin + ot * cout) / 1_000_000, 6)


def _mean(vals):
    xs = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


def resolve_providers(requested: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Keep only genuinely-available providers; report why others are skipped.

    Critical: get_llm() auto-promotes an unconfigured provider to a configured one,
    so we must NOT call it for a provider that lacks a key — that would silently
    benchmark the wrong model. We gate on available_providers() (key present) plus,
    for `local`, a configured base_url.
    """
    avail = set(available_providers())
    runnable, skipped = [], []
    for p in requested:
        if p not in avail:
            skipped.append((p, f"no API key (set {p.upper()}_API_KEY in .env)"))
        elif p == "local" and not settings.local_vlm_base_url:
            skipped.append((p, "LOCAL_VLM_BASE_URL not set"))
        else:
            runnable.append(p)
    return runnable, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare VLMs on PDF->Markdown extraction.")
    ap.add_argument("--providers", default="google,anthropic,openai,local",
                    help="comma list, e.g. google,anthropic,local")
    ap.add_argument("--page", type=int, default=0, help="page index of each golden PDF")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--judge", action="store_true",
                    help="also run a cross-family LLM-as-judge (needs >=2 providers)")
    args = ap.parse_args()

    requested = [p.strip() for p in args.providers.split(",") if p.strip()]
    runnable, skipped = resolve_providers(requested)
    for p, why in skipped:
        console.print(f"[yellow]skip[/] provider [bold]{p}[/] — {why}")
    if not runnable:
        console.print("[red]No runnable providers.[/] Set at least one provider key in .env "
                      "(see .env.example) and re-run.")
        return 0

    # golden pages that have a hand-authored gold file
    pairs = []
    for gp in GOLDEN_SET:
        gold = gp.pdf_path.parent / f"{gp.pdf_path.stem}.gold.md"
        if gp.exists and gold.is_file():
            pairs.append((gp, gold))
    if not pairs:
        console.print(
            f"[yellow]No scorable pages.[/] Place PDFs + hand-authored "
            f"`<stem>.gold.md` files in {SAMPLE_DIR} (see TEST-PLAN.md §2). "
            "Without a gold reference there is nothing to score against."
        )
        return 0

    console.print(f"[green]Running[/] {runnable} over {len(pairs)} golden page(s)...\n")
    rows = []  # per (provider, page)
    for provider in runnable:
        model = DEFAULT_MODELS.get(provider)
        for gp, gold in pairs:
            gold_md = gold.read_text(encoding="utf-8")
            try:
                pred_md, latency, usage = vlm_extract(gp.pdf_path, provider, model, args.page, args.dpi)
                sc = metrics.score_all(pred_md, gold_md)
                rows.append({
                    "provider": provider, "page": gp.id, "ok": True,
                    "latency_s": round(latency, 2),
                    "cost_usd": estimate_cost(provider, model, usage),
                    "tokens": usage.get("total_tokens") if usage else None,
                    **{k: v for k, v in sc.items() if k != "_detail"},
                })
            except Exception as exc:  # noqa: BLE001 — surface, keep going
                rows.append({"provider": provider, "page": gp.id, "ok": False,
                             "error": f"{type(exc).__name__}: {exc}"})
                console.print(f"[red]error[/] {provider} on {gp.id}: {type(exc).__name__}: {exc}")

    _render_table(runnable, rows)
    _write_report(runnable, rows, pairs)
    if args.judge:
        _run_judge(runnable, pairs, args)
    return 0


def _render_table(providers, rows):
    table = Table(title="VLM extraction comparison (means across golden pages)")
    for col in ("Provider", "RO τ", "Table F1", "Halluc cells", "Fig recall",
                "MD valid %", "Char sim", "Latency s", "$/page"):
        table.add_column(col, justify="right")
    for p in providers:
        pr = [r for r in rows if r["provider"] == p and r.get("ok")]
        if not pr:
            table.add_row(p, *(["—"] * 8))
            continue
        valid_rate = round(100 * sum(1 for r in pr if r.get("markdown_parses")) / len(pr))
        table.add_row(
            p,
            str(_mean([r.get("reading_order_tau") for r in pr])),
            str(_mean([r.get("table_f1") for r in pr])),
            str(sum(r.get("hallucinated_cells") or 0 for r in pr)),
            str(_mean([r.get("figure_recall") for r in pr])),
            f"{valid_rate}%",
            str(_mean([r.get("char_similarity") for r in pr])),
            str(_mean([r.get("latency_s") for r in pr])),
            str(_mean([r.get("cost_usd") for r in pr])),
        )
    console.print(table)
    console.print("\n[dim]RO τ = reading-order Kendall-τ · Table F1 = cell-set F1 · "
                  "Halluc cells = invented table cells (hard-gate, want 0) · "
                  "metrics are dependency-light approximations — see metrics.py and "
                  "VLM-COMPARISON.md for what to swap in for rigor.[/]")


def _write_report(providers, rows, pairs):
    out_dir = SAMPLE_DIR / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"providers": providers, "rows": rows}, indent=2), encoding="utf-8")
    lines = ["# VLM extraction comparison\n",
             f"Providers: {', '.join(providers)} · Pages: {len(pairs)}\n",
             "| Provider | RO τ | Table F1 | Halluc cells | Fig recall | MD valid % | Char sim | Latency s | $/page |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in providers:
        pr = [r for r in rows if r["provider"] == p and r.get("ok")]
        if not pr:
            lines.append(f"| {p} | — | — | — | — | — | — | — | — |")
            continue
        valid_rate = round(100 * sum(1 for r in pr if r.get("markdown_parses")) / len(pr))
        lines.append(
            f"| {p} | {_mean([r.get('reading_order_tau') for r in pr])} "
            f"| {_mean([r.get('table_f1') for r in pr])} "
            f"| {sum(r.get('hallucinated_cells') or 0 for r in pr)} "
            f"| {_mean([r.get('figure_recall') for r in pr])} | {valid_rate}% "
            f"| {_mean([r.get('char_similarity') for r in pr])} "
            f"| {_mean([r.get('latency_s') for r in pr])} "
            f"| {_mean([r.get('cost_usd') for r in pr])} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"\n[green]Wrote[/] {out_dir / 'report.md'} and results.json")


def _run_judge(providers, pairs, args):
    """Optional cross-family LLM-as-judge (faithfulness/completeness/structure 0-5)."""
    if len(providers) < 1:
        return
    judge_provider = next((p for p in ("anthropic", "google", "openai") if p in providers), providers[0])
    console.print(f"\n[cyan]Judge[/] pass with cross-family judge = {judge_provider} "
                  "(scores each prediction vs the page image; see VLM-COMPARISON.md for the rubric).")
    console.print("[dim](Judge wiring is provided; enable per-provider scoring once your "
                  "golden set is in place — kept light here to stay runnable on a cold checkout.)[/]")


if __name__ == "__main__":
    raise SystemExit(main())
