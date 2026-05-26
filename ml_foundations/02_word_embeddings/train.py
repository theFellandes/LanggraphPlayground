"""Train a Skip-gram Word2Vec on the local corpus.

Run:
    uv sync --extra ml
    uv run python -m ml_foundations.02_word_embeddings.train
"""

from __future__ import annotations

import re
from pathlib import Path

from shared import settings
from shared.pretty import console, section

EMBED_DIR = settings.data_dir / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_RE = re.compile(r"[A-Za-zçğıöşüÇĞİÖŞÜ]+(?:'[A-Za-z]+)?")


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _gather_sentences() -> list[list[str]]:
    sents: list[list[str]] = []
    for p in (settings.data_dir / "sample_docs").glob("*.md"):
        text = p.read_text(encoding="utf-8")
        # Naive sentence split — fine for the size of our demo corpus.
        for chunk in re.split(r"(?<=[.!?])\s+", text):
            tokens = _tokenise(chunk)
            if len(tokens) >= 3:
                sents.append(tokens)
    return sents


def train_w2v(sents: list[list[str]]) -> Path:
    try:
        from gensim.models import Word2Vec
    except ImportError:
        console.print("[yellow]Missing `gensim`. Run: uv sync --extra ml[/]")
        raise

    section("Training Word2Vec Skip-gram (dim=100, window=5)")
    model = Word2Vec(
        sentences=sents,
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,             # skip-gram (vs CBOW)
        epochs=20,
        workers=2,
    )
    path = EMBED_DIR / "w2v.kv"
    model.wv.save(str(path))
    console.print(f"Saved KeyedVectors → {path}  vocab={len(model.wv)}")
    return path


def explore(path: Path) -> None:
    from gensim.models import KeyedVectors

    section("Exploring the trained vector space")
    wv = KeyedVectors.load(str(path))

    for w in ("refund", "policy", "agent", "rag", "tool"):
        if w in wv:
            console.print(f"[bold]{w}[/] ≈ {[t for t, _ in wv.most_similar(w, topn=5)]}")
        else:
            console.print(f"[dim]{w!r} not in vocab — corpus too small[/]")


def project_2d(path: Path) -> None:
    """Save a 2D PCA projection of the top-100 vocab to disk."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        console.print("[yellow]Skipping projection (matplotlib/sklearn not installed)[/]")
        return

    from gensim.models import KeyedVectors

    section("2D projection (PCA) of top-100 vocab")
    wv = KeyedVectors.load(str(path))
    words = [w for w, _ in wv.most_similar(positive=[wv.index_to_key[0]], topn=100)]
    if len(words) < 5:
        console.print("[yellow]Vocab too small to project; skipping[/]")
        return

    X = [wv[w] for w in words]
    coords = PCA(n_components=2).fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], s=8)
    for (x, y), w in zip(coords, words):
        ax.annotate(w, (x, y), fontsize=7)
    out = EMBED_DIR / "projection.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    console.print(f"Saved → {out}")


def main() -> None:
    sents = _gather_sentences()
    console.print(f"Sentences: {len(sents)}")
    if not sents:
        console.print("[yellow]Empty corpus. Add files under data/sample_docs/[/]")
        return
    path = train_w2v(sents)
    explore(path)
    project_2d(path)


if __name__ == "__main__":
    main()
