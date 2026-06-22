"""Run the extractors over whatever golden-set PDFs are present.

    uv sync --extra extraction         # install docling + pymupdf4llm (+ render dep)
    uv add langchain-google-genai      # the Gemini provider (already in core deps)
    uv run python -m projects.vlm_extraction_harness.run

For each PDF in ``data/sample_docs/`` it runs every method, writes the predicted
Markdown to ``data/sample_docs/preds/<id>.<method>.pred.md``, and prints a status
table. Methods whose deps/keys are missing are reported as ``skipped`` — the run
never crashes on a missing extractor.

SCOPE: this is the *extraction* runner. Scoring (TEST-PLAN §4 metrics: Kendall-τ
reading order, TEDS tables, Markdown-validity gate, LLM-as-judge) is the next step
and is intentionally NOT implemented here — see the README and TEST-PLAN.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the harness runnable both as `python -m ...` and as a direct script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402  (core dep)
from rich.table import Table  # noqa: E402

from projects.vlm_extraction_harness.extractors import METHODS  # noqa: E402
from projects.vlm_extraction_harness.golden_set import GOLDEN_SET, SAMPLE_DIR  # noqa: E402

console = Console()


def main() -> int:
    out_dir = SAMPLE_DIR / "preds"
    present = [p for p in GOLDEN_SET if p.exists]

    if not present:
        console.print(
            f"[yellow]No golden-set PDFs found in[/] {SAMPLE_DIR}\n"
            "Add at least one (e.g. g1_born_digital_prose.pdf) per "
            "data/sample_docs/README.md, then re-run."
        )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    table = Table(title="VLM extraction harness — extraction pass")
    for col in ("Page", "Method", "Status", "Chars", "Output"):
        table.add_column(col, overflow="fold")

    for page in present:
        for label, fn in METHODS.items():
            try:
                md = fn(page.pdf_path)
                dest = out_dir / f"{page.id}.{label}.pred.md"
                dest.write_text(md, encoding="utf-8")
                status, chars, where = "[green]ok[/]", str(len(md)), str(dest.relative_to(ROOT))
            except RuntimeError as exc:  # missing dep / key — expected, non-fatal
                status, chars, where = "[yellow]skipped[/]", "-", str(exc).splitlines()[0]
            except Exception as exc:  # noqa: BLE001 — surface real extractor errors
                status, chars, where = "[red]error[/]", "-", f"{type(exc).__name__}: {exc}"
            table.add_row(page.id, label, status, chars, where)

    console.print(table)
    console.print(
        "\n[dim]Extraction only. Implement TEST-PLAN §4 scoring "
        "(Kendall-τ / TEDS / Markdown-validity / LLM-judge) against the "
        "*.gold.md files to produce the comparison.[/]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
