"""Dependency-light scoring metrics for PDF→Markdown extraction.

Implements the TEST-PLAN.md §4 metric family with **stdlib only** (``re``,
``difflib``, ``math``, ``collections``) so the comparison runs on a cold
``uv sync`` with no extra installs. Each function takes a *prediction* Markdown
string and a *gold* Markdown string and returns a small score/dict.

Where a heavier, more rigorous metric exists, the docstring names what to swap in
— this module is the honest, runnable floor, not the last word:

  - reading order  → Kendall-τ over fuzzy-matched text blocks   (rigorous: ``scipy.stats.kendalltau``)
  - table fidelity → cell-set F1 + hallucinated-cell count       (rigorous: TEDS via ``apted`` + ``lxml``)
  - markdown valid → fence/heading heuristic, or markdown-it-py  (rigorous: full CommonMark AST)
  - formula        → LaTeX-span recall (presence only)           (rigorous: LLM-as-judge — compare_vlms --judge)

Hallucination note: ``table_cell_f1`` reports ``hallucinated_cells`` (cells in
the prediction absent from gold). A single invented number in a RAG index is a
high-confidence wrong answer, so treat that count as a hard gate (TEST-PLAN §6).
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

__all__ = [
    "reading_order_tau",
    "table_cell_f1",
    "figure_recall",
    "formula_recall",
    "markdown_validity",
    "char_similarity",
    "score_all",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _blocks(md: str) -> list[str]:
    """Split Markdown into normalized non-empty blocks (paragraphs / rows)."""
    out: list[str] = []
    for raw in re.split(r"\n\s*\n", md or ""):
        t = _norm(raw)
        if t:
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
# reading order — Kendall-τ over fuzzy-matched blocks
# --------------------------------------------------------------------------- #
def reading_order_tau(pred_md: str, gold_md: str, min_ratio: float = 0.6) -> dict:
    """1.0 = pred presents gold's blocks in the same order; -1.0 = fully reversed.

    Each gold block is matched to its best (unused) fuzzy match in pred, then we
    compute Kendall-τ on the pred positions ordered by gold position. Penalizes
    multi-column scramble and out-of-sequence footnotes (TEST-PLAN §4, G3).
    """
    gold, pred = _blocks(gold_md), _blocks(pred_md)
    matched: list[tuple[int, int]] = []
    used: set[int] = set()
    for gi, g in enumerate(gold):
        best_j, best_r = -1, min_ratio
        for pj, p in enumerate(pred):
            if pj in used:
                continue
            r = SequenceMatcher(None, g, p).ratio()
            if r >= best_r:
                best_r, best_j = r, pj
        if best_j >= 0:
            used.add(best_j)
            matched.append((gi, best_j))
    if len(matched) < 2:
        return {"tau": None, "matched": len(matched), "gold_blocks": len(gold),
                "note": "too few matched blocks to order"}
    seq = [pj for _, pj in sorted(matched)]
    concord = discord = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] < seq[j]:
                concord += 1
            elif seq[i] > seq[j]:
                discord += 1
    denom = concord + discord
    tau = (concord - discord) / denom if denom else None
    return {"tau": round(tau, 3) if tau is not None else None,
            "matched": len(matched), "gold_blocks": len(gold),
            "coverage": round(len(matched) / max(1, len(gold)), 3)}


# --------------------------------------------------------------------------- #
# table fidelity — cell-set F1 + hallucinated cells
# --------------------------------------------------------------------------- #
_PIPE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_SEP_ROW = re.compile(r"^[\s:\-|]+$")


def _table_cells(md: str) -> list[str]:
    """Extract table cell texts from Markdown pipe tables AND HTML <td>/<th>."""
    cells: list[str] = []
    for line in (md or "").splitlines():
        m = _PIPE_ROW.match(line)
        if not m or _SEP_ROW.match(line):
            continue
        for c in m.group(1).split("|"):
            c = _norm(c)
            if c:
                cells.append(c)
    for raw in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", md or "", flags=re.I | re.S):
        c = _norm(re.sub(r"<[^>]+>", " ", raw))
        if c:
            cells.append(c)
    return cells


def table_cell_f1(pred_md: str, gold_md: str) -> dict:
    """Multiset cell F1 (TEDS-lite). ``hallucinated_cells`` = pred cells not in gold."""
    gold, pred = Counter(_table_cells(gold_md)), Counter(_table_cells(pred_md))
    if not gold and not pred:
        return {"f1": None, "note": "no tables in gold or pred"}
    tp = sum((gold & pred).values())
    g_total, p_total = sum(gold.values()), sum(pred.values())
    precision = tp / p_total if p_total else 0.0
    recall = tp / g_total if g_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"f1": round(f1, 3), "precision": round(precision, 3),
            "recall": round(recall, 3), "hallucinated_cells": sum((pred - gold).values()),
            "gold_cells": g_total, "pred_cells": p_total}


# --------------------------------------------------------------------------- #
# figures / formulas — presence recall (coarse)
# --------------------------------------------------------------------------- #
_IMG = re.compile(r"!\[[^\]]*\]\([^)]+\)|<img\b", re.I)
_FORMULA = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$", re.S)


def figure_recall(pred_md: str, gold_md: str) -> dict:
    g, p = len(_IMG.findall(gold_md or "")), len(_IMG.findall(pred_md or ""))
    return {"gold_figures": g, "pred_figures": p,
            "recall": round(min(p, g) / g, 3) if g else None}


def formula_recall(pred_md: str, gold_md: str) -> dict:
    g, p = len(_FORMULA.findall(gold_md or "")), len(_FORMULA.findall(pred_md or ""))
    return {"gold_formulas": g, "pred_formulas": p,
            "recall": round(min(p, g) / g, 3) if g else None,
            "note": "coarse presence only — use compare_vlms --judge for correctness"}


# --------------------------------------------------------------------------- #
# markdown validity + holistic similarity
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"^```", re.M)
_HEADING = re.compile(r"^#{1,6}\s+\S", re.M)


def markdown_validity(pred_md: str, gold_md: str | None = None) -> dict:
    """Hard gate (parses) + heading preservation vs gold. Uses markdown-it-py if present."""
    md = pred_md or ""
    fences_ok = len(_FENCE.findall(md)) % 2 == 0
    headings = len(_HEADING.findall(md))
    try:
        from markdown_it import MarkdownIt

        MarkdownIt().parse(md)
        parses = True
    except ModuleNotFoundError:
        parses = fences_ok  # stdlib heuristic fallback
    except Exception:
        parses = False
    out = {"parses": parses, "fences_balanced": fences_ok, "headings": headings}
    if gold_md is not None:
        gold_h = len(_HEADING.findall(gold_md))
        out["heading_preservation"] = round(min(headings, gold_h) / gold_h, 3) if gold_h else None
    return out


def char_similarity(pred_md: str, gold_md: str) -> float:
    """Cheap holistic signal: normalized character-level similarity (0..1)."""
    return round(SequenceMatcher(None, _norm(pred_md), _norm(gold_md)).ratio(), 3)


# --------------------------------------------------------------------------- #
# combined
# --------------------------------------------------------------------------- #
def score_all(pred_md: str, gold_md: str) -> dict:
    """Run every metric; return a flat dict (+ ``_detail`` for drill-down)."""
    ro = reading_order_tau(pred_md, gold_md)
    tbl = table_cell_f1(pred_md, gold_md)
    fig = figure_recall(pred_md, gold_md)
    frm = formula_recall(pred_md, gold_md)
    val = markdown_validity(pred_md, gold_md)
    return {
        "reading_order_tau": ro.get("tau"),
        "table_f1": tbl.get("f1"),
        "hallucinated_cells": tbl.get("hallucinated_cells"),
        "figure_recall": fig.get("recall"),
        "formula_recall": frm.get("recall"),
        "markdown_parses": val.get("parses"),
        "heading_preservation": val.get("heading_preservation"),
        "char_similarity": char_similarity(pred_md, gold_md),
        "_detail": {"reading_order": ro, "table": tbl, "figure": fig,
                    "formula": frm, "validity": val},
    }


if __name__ == "__main__":  # tiny self-test on synthetic strings
    gold = "# Title\n\nAlpha beta gamma.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n![fig](f.png)"
    good = "# Title\n\nAlpha beta gamma.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n![fig](f.png)"
    bad = "Alpha.\n\n| A | B |\n|---|---|\n| 1 | 9 |\n"  # wrong cell, no title/figure
    import json

    print("perfect:", json.dumps({k: v for k, v in score_all(good, gold).items() if k != "_detail"}))
    print("degraded:", json.dumps({k: v for k, v in score_all(bad, gold).items() if k != "_detail"}))
