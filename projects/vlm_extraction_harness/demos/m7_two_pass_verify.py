"""Method 7 — Two-pass / verification (LLM-as-judge re-extraction).

    uv sync --extra extraction         # docling + pymupdf4llm + PyMuPDF (render dep)
    uv add langchain-google-genai      # the Gemini provider (pass-2 verifier VLM)
    uv run python -m projects.vlm_extraction_harness.demos.m7_two_pass_verify [pdf]

THE IDEA (FINDINGS §4, method 7)
--------------------------------
Extraction is cheap to *do* and expensive to *trust*. Method 7 separates the two:

  pass 1  ── DRAFT   : extract Markdown with any method (here: the text-layer
                       baseline ``pymupdf4llm`` — fast, free, but blind to whether
                       a table merged wrong or a scanned region came out blank).
  pass 2  ── VERIFY  : a *second* model (a frontier VLM via the adapter) sees the
                       PAGE IMAGE **and** the draft side by side and acts as an
                       LLM-as-judge. It returns a structured verdict: a list of
                       per-region issues, a severity, and — only when it finds
                       real problems — a corrected Markdown.
  merge   ── DECIDE  : if the judge says the draft is faithful, keep it (cheap path
                       wins). If it flags issues, accept the judge's corrected
                       Markdown (targeted re-extraction). Either way you ship the
                       version the judge has signed off on, plus an audit trail.

This is the same discipline as LLM-as-judge / self-consistency for generation,
applied to document extraction: the generator and the critic are different roles,
the critic is grounded on the source pixels (not just the draft text), and the
critic must justify each edit so the merge is auditable rather than a black box.

WHY A SECOND MODEL SEES THE IMAGE
---------------------------------
A verifier that only reads the draft can catch *internal* inconsistencies (a table
row with the wrong column count) but cannot catch *fidelity* errors — a number the
extractor misread, a column the reading-order scrambled, a figure caption it
invented. Grounding pass 2 on the rendered page is what makes the verification
meaningful: every claimed issue must be checkable against the pixels.

Heavy deps are imported LAZILY so importing this module never forces the
``extraction`` extra. The VLM is reached ONLY through ``get_llm("google", ...)``
so it stays switchable and inherits the ``Runnable.with_fallbacks`` chain.
"""

from __future__ import annotations

import base64
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Make the demo runnable both as ``python -m ...`` and as a direct script.
ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.llm import get_llm  # noqa: E402

from projects.vlm_extraction_harness.extractors import (  # noqa: E402
    pymupdf4llm_extract,
    render_page_png,
)

# --------------------------------------------------------------------------- #
# Pass-2 verifier prompt. The judge is told its job, what counts as an issue,
# and to return a single JSON object. Asking for the corrected Markdown only
# when issues exist keeps the cheap path cheap and the audit trail honest.
# --------------------------------------------------------------------------- #
VERIFIER_PROMPT = """You are a meticulous document-extraction VERIFIER (an LLM judge).

You are given (1) the rendered image of ONE PDF page, and (2) a DRAFT Markdown
extraction of that same page produced by another tool. Your job is NOT to
re-transcribe from scratch — it is to AUDIT the draft against the page image and
report whether it is faithful.

Check, grounded ONLY on what the image actually shows:
- Reading order (in multi-column pages, columns must be read top-to-bottom,
  left column fully before the right — never interleaved line by line).
- Tables: correct rows/columns, merged cells preserved (HTML <table> with
  colspan/rowspan), and NO invented or dropped cells or numbers.
- Numbers and text transcribed exactly (watch digit swaps, e.g. 5↔6, 1↔7).
- Figures represented as a caption only; nothing about a figure invented.
- Missing regions (e.g. a scanned block the draft left blank).

Return EXACTLY ONE JSON object, no prose, no code fences, with this shape:
{
  "faithful": true | false,
  "issues": [
    {"region": "<short label, e.g. 'table row 3' or 'left column'>",
     "severity": "low" | "medium" | "high",
     "problem": "<what is wrong, checkable against the image>"}
  ],
  "corrected_markdown": "<full corrected Markdown for the page, ONLY if faithful
                          is false; otherwise the empty string>"
}

Rules:
- If the draft is already faithful, set "faithful": true, "issues": [], and
  "corrected_markdown": "".
- Only set "faithful": false when you can point to a concrete, image-grounded
  error. Do not invent issues to look busy.
- "corrected_markdown" must be the WHOLE page corrected, not a diff."""


@dataclass
class VerifyResult:
    """Outcome of the two-pass run — the shipped Markdown plus the audit trail."""

    final_markdown: str
    faithful: bool
    issues: list[dict] = field(default_factory=list)
    draft_markdown: str = ""
    used_correction: bool = False
    note: str = ""


# --------------------------------------------------------------------------- #
# Pass 1 — DRAFT. Any extractor works; we use the text-layer baseline because it
# is the cheap default and its failure modes (blank scanned pages, scrambled
# columns, mangled tables) are exactly what pass 2 is there to catch.
# --------------------------------------------------------------------------- #
def extract_draft(pdf_path: str | Path, page_number: int = 0) -> str:
    """Pass 1: produce a draft Markdown extraction (cheap, may be wrong)."""
    return pymupdf4llm_extract(pdf_path, page_number=page_number)


def _parse_verdict(raw: str) -> dict:
    """Lenient JSON parse — strip code fences / surrounding prose if the model added any.

    We do NOT use ``with_structured_output`` here on purpose: the verifier runs
    through the fallback chain (possibly hitting a provider whose strict-mode
    schema support differs), and a teaching demo should stay robust to a model
    that wraps its JSON in ```json fences. So we parse defensively instead.
    """
    text = raw.strip()
    # Drop a leading ```json / ``` fence and a trailing ``` if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: grab the outermost {...} block.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError("verifier did not return parseable JSON")


# --------------------------------------------------------------------------- #
# Pass 2 — VERIFY. A frontier VLM sees the page image + the draft and judges.
# --------------------------------------------------------------------------- #
def verify_and_correct(
    pdf_path: str | Path,
    draft_markdown: str,
    page_number: int = 0,
    dpi: int = 300,
    model: str = "gemini-2.0-flash",
) -> VerifyResult:
    """Pass 2: render the page, hand image + draft to the judge, merge the verdict.

    Requires ``GOOGLE_API_KEY`` and ``langchain-google-genai``. The VLM is reached
    only via ``get_llm("google", ...)`` (switchable; inherits the fallback chain).
    """
    from langchain_core.messages import HumanMessage

    png = render_page_png(pdf_path, page_number, dpi)
    b64 = base64.b64encode(png).decode()
    judge = get_llm("google", model=model, temperature=0)

    # Same string-shorthand image_url form the harness's gemini_direct uses, which
    # langchain-google-genai accepts as a base64 data URL.
    msg = HumanMessage(
        content=[
            {"type": "text", "text": VERIFIER_PROMPT},
            {"type": "text", "text": f"\n--- DRAFT MARKDOWN ---\n{draft_markdown}"},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        ]
    )
    out = judge.invoke([msg])
    raw = out.content if hasattr(out, "content") else str(out)

    verdict = _parse_verdict(raw)
    faithful = bool(verdict.get("faithful", True))
    issues = verdict.get("issues", []) or []
    corrected = (verdict.get("corrected_markdown") or "").strip()

    # MERGE / DECIDE: trust the cheap draft when the judge signs off; otherwise
    # ship the judge's corrected page (targeted re-extraction). Guard against a
    # judge that says "not faithful" but forgets to supply a correction.
    if faithful or not corrected:
        return VerifyResult(
            final_markdown=draft_markdown,
            faithful=faithful,
            issues=issues,
            draft_markdown=draft_markdown,
            used_correction=False,
            note="draft accepted" if faithful else "flagged but no correction supplied; kept draft",
        )
    return VerifyResult(
        final_markdown=corrected,
        faithful=False,
        issues=issues,
        draft_markdown=draft_markdown,
        used_correction=True,
        note="draft replaced by verifier correction",
    )


def two_pass_extract(
    pdf_path: str | Path,
    page_number: int = 0,
    dpi: int = 300,
    model: str = "gemini-2.0-flash",
) -> VerifyResult:
    """Primary function: pass 1 draft → pass 2 verify/correct → merged result."""
    draft = extract_draft(pdf_path, page_number=page_number)
    return verify_and_correct(
        pdf_path, draft, page_number=page_number, dpi=dpi, model=model
    )


def demo(pdf_path: str | Path) -> None:
    """Run the two-pass pipeline on ``pdf_path`` and print a short result."""
    pdf_path = Path(pdf_path)
    print(f"[m7] two-pass verify on: {pdf_path.name}")

    # Pass 1 always runs (no key needed) — show what the cheap draft produced.
    try:
        draft = extract_draft(pdf_path)
    except RuntimeError as exc:  # missing extraction extra
        print(f"[m7] skipped: {exc}")
        return
    print(f"[m7] pass 1 (pymupdf4llm draft): {len(draft)} chars")
    if not draft.strip():
        print("[m7]   draft is BLANK — likely a scanned page; pass 2 should flag it.")

    # Pass 2 needs the VLM (key + provider). Degrade gracefully if unavailable.
    try:
        result = verify_and_correct(pdf_path, draft)
    except RuntimeError as exc:  # PyMuPDF render guard
        print(f"[m7] skipped pass 2: {exc}")
        return
    except Exception as exc:  # missing GOOGLE_API_KEY / provider / network
        print(f"[m7] skipped pass 2 (verifier unavailable): {exc}")
        print("[m7]   set GOOGLE_API_KEY and `uv add langchain-google-genai` to enable.")
        return

    print(f"[m7] pass 2 verdict: faithful={result.faithful}, "
          f"issues={len(result.issues)}, used_correction={result.used_correction}")
    for issue in result.issues[:5]:
        print(f"[m7]   - [{issue.get('severity', '?')}] "
              f"{issue.get('region', '?')}: {issue.get('problem', '')}")
    print(f"[m7] note: {result.note}")
    print(f"[m7] final markdown: {len(result.final_markdown)} chars")
    preview = result.final_markdown.strip().splitlines()[:8]
    for line in preview:
        print(f"      | {line}")


def _first_present_pdf() -> Path | None:
    try:
        from projects.vlm_extraction_harness.golden_set import GOLDEN_SET
    except Exception:
        return None
    for page in GOLDEN_SET:
        if page.exists:
            return page.pdf_path
    return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target: Path | None = Path(sys.argv[1])
    else:
        target = _first_present_pdf()

    if target is None or not Path(target).is_file():
        print("[m7] no PDF found. Pass one as argv[1], or drop a golden-set PDF "
              "into data/sample_docs/ (e.g. g4_table_heavy.pdf).")
        sys.exit(0)

    demo(target)
