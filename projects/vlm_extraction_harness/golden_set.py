"""The golden set definition (TEST-PLAN.md §2).

Pure data — the actual PDFs live (uncommitted) in ``data/sample_docs/`` and are
described in that folder's README. ``run.py`` iterates these and runs whichever
PDFs are present.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.settings import settings

SAMPLE_DIR = settings.data_dir / "sample_docs"


@dataclass(frozen=True)
class GoldenPage:
    id: str
    filename: str
    stresses: str
    expected_hard_part: str

    @property
    def pdf_path(self):
        return SAMPLE_DIR / self.filename

    @property
    def exists(self) -> bool:
        return self.pdf_path.is_file()


GOLDEN_SET: list[GoldenPage] = [
    GoldenPage(
        "G1", "g1_born_digital_prose.pdf",
        "Born-digital single-column prose, real text layer",
        "Should be easy — the baseline; catches over-engineering.",
    ),
    GoldenPage(
        "G2", "g2_scanned_page.pdf",
        "Scanned, image-only page (no text layer)",
        "pymupdf4llm emits silent blank output — the baseline's hard floor.",
    ),
    GoldenPage(
        "G3", "g3_multicolumn_paper.pdf",
        "Two-column academic page, footnotes crossing columns",
        "Reading-order scramble: columns merged L→R instead of column-first.",
    ),
    GoldenPage(
        "G4", "g4_table_heavy.pdf",
        "Dense table with merged cells (colspan/rowspan) + numbers",
        "Markdown pipes can't express merged cells; catches hallucinated cells.",
    ),
    GoldenPage(
        "G5", "g5_figure_heavy.pdf",
        "Figures/charts with captions",
        "Figure-pixel extraction + caption association + chart-to-data.",
    ),
    GoldenPage(
        "G6", "g6_turkish.pdf",
        "Turkish page (dotless-ı / dotted-İ, ş/ğ/ç) — Guillotine-critical",
        "Turkish casing trap; no verified TR score exists for any tool.",
    ),
    GoldenPage(
        "G6b", "g6_arabic.pdf",
        "Arabic RTL page — Guillotine-critical",
        "RTL reading-order + glyph joining; no verified AR score exists.",
    ),
]
