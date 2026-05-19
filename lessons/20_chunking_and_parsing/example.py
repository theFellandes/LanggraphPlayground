"""Lesson 20 · Chunking and parsing strategies.

Tiny side-by-side demos of:

  PARSERS (text → Documents)
    1. TextLoader   — plain text / markdown
    2. PyPDFLoader  — PDFs (skipped gracefully if no PDF on disk)

  CHUNKERS (Documents → smaller Documents)
    1. CharacterTextSplitter         — split on one separator
    2. RecursiveCharacterTextSplitter — try paragraph → sentence → word
    3. TokenTextSplitter             — tiktoken-based, model-aware budgets
    4. MarkdownHeaderTextSplitter     — preserve section structure

Run:
    uv run python -m lessons.20_chunking_and_parsing.example
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from shared.pretty import console, section
from shared.settings import settings


SAMPLE = settings.data_dir / "sample_docs" / "langgraph_intro.md"


# ───────────────────────────────────────────────────────────────────────────
# PARSERS
# ───────────────────────────────────────────────────────────────────────────
def parse_text(path: Path) -> list[Document]:
    """Plain text / markdown — every file becomes one Document."""
    return TextLoader(str(path), encoding="utf-8").load()


def parse_pdf(path: Path) -> list[Document]:
    """PyPDFLoader yields one Document per page.

    Best for *born-digital* PDFs. Scanned PDFs need OCR (Tesseract,
    or AI-powered tools like Docling / Unstructured / LlamaParse).
    """
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(str(path)).load()


# ───────────────────────────────────────────────────────────────────────────
# CHUNKERS
# ───────────────────────────────────────────────────────────────────────────
def chunk_character(docs: list[Document], size: int = 300, overlap: int = 40):
    """Single-separator split. Brittle on real text — start here, then upgrade."""
    splitter = CharacterTextSplitter(
        separator="\n\n", chunk_size=size, chunk_overlap=overlap
    )
    return splitter.split_documents(docs)


def chunk_recursive(docs: list[Document], size: int = 300, overlap: int = 40):
    """Try ['\\n\\n', '\\n', ' ', ''] in order. The default for prose RAG."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(docs)


def chunk_token(docs: list[Document], size: int = 80, overlap: int = 10):
    """Tiktoken-based budgeting — chunks fit a model's context exactly.

    Pure character splitters can produce chunks that look ~300 chars but
    cost wildly different token counts (especially in non-English text).
    Token-based splitting fixes that.
    """
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",   # GPT-4 / 4o family
        chunk_size=size,
        chunk_overlap=overlap,
    )
    return splitter.split_documents(docs)


def chunk_markdown(docs: list[Document]):
    """MarkdownHeaderTextSplitter preserves section context as metadata."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    )
    out: list[Document] = []
    for d in docs:
        for chunk in splitter.split_text(d.page_content):
            out.append(chunk)   # already a Document with `metadata` carrying headers
    return out


# ───────────────────────────────────────────────────────────────────────────
# helpers
# ───────────────────────────────────────────────────────────────────────────
def preview(label: str, chunks: list[Document], n: int = 1) -> None:
    """Print a compact summary of a chunk list."""
    console.print(f"[bold cyan]{label:32}[/] {len(chunks):>3} chunks  "
                  f"avg={sum(len(c.page_content) for c in chunks) // max(len(chunks),1):>4} chars")
    for i, c in enumerate(chunks[:n], 1):
        first_line = c.page_content.strip().split("\n", 1)[0][:90]
        console.print(f"  [dim][{i}][/] {first_line!r}")
        if c.metadata:
            meta = ", ".join(f"{k}={v!r}" for k, v in c.metadata.items() if k != "source")
            if meta:
                console.print(f"      [dim]metadata: {meta}[/]")


# ───────────────────────────────────────────────────────────────────────────
def main() -> None:
    section("Lesson 20 · parsing")
    docs = parse_text(SAMPLE)
    console.print(f"TextLoader → {len(docs)} document(s), "
                  f"{sum(len(d.page_content) for d in docs)} chars total\n")

    # PDF demo — skip cleanly if no .pdf in data/
    pdf_candidates = list((settings.data_dir / "sample_docs").glob("*.pdf"))
    if pdf_candidates:
        pdf_docs = parse_pdf(pdf_candidates[0])
        console.print(f"PyPDFLoader → {len(pdf_docs)} page(s) from {pdf_candidates[0].name}")
    else:
        console.print("[dim](no PDF in data/sample_docs/ — drop one in to see PyPDFLoader)[/]")

    section("Lesson 20 · chunking (same input, four strategies)")
    preview("CharacterTextSplitter",            chunk_character(docs))
    preview("RecursiveCharacterTextSplitter",   chunk_recursive(docs))
    preview("TokenTextSplitter (tiktoken)",      chunk_token(docs))
    preview("MarkdownHeaderTextSplitter",        chunk_markdown(docs))


if __name__ == "__main__":
    main()
