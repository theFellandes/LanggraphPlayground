"""Train two tokenizers from scratch, then compare them against GPT-4's.

Run:
    uv sync --extra ml
    uv run python -m ml_foundations.01_tokenizers.train

Outputs:
    - data/tokenizers/bpe_lesson01.json  (Hugging Face fast tokenizer)
    - data/tokenizers/sp_lesson01.model  (SentencePiece unigram)
    - A side-by-side encoding table.
"""

from __future__ import annotations

from pathlib import Path

from shared import settings
from shared.pretty import console, section

TOK_DIR = settings.data_dir / "tokenizers"
TOK_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_TEXTS = [
    "Hello, how are you?",
    "Merhaba, nasılsın?",
    "lesson 29 — vector DB comparison",
    "def foo():\n    pass",
    "こんにちは世界",
    "Anthropic is a Claude maker",
    "Türkçe metin sıkıştırması daha az verimli",
]


def _gather_corpus() -> list[str]:
    """Concatenate every .md under data/sample_docs/, then split into lines."""
    lines: list[str] = []
    for p in (settings.data_dir / "sample_docs").glob("*.md"):
        text = p.read_text(encoding="utf-8")
        lines.extend(t for t in text.splitlines() if t.strip())
    return lines


def train_bpe(corpus: list[str]) -> Path:
    section("Training BPE tokenizer (vocab=2048)")
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    except ImportError:
        console.print("[yellow]Missing `tokenizers`. Run: uv sync --extra ml[/]")
        raise

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=2048,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<SUPPORT_TICKET>"],
        min_frequency=2,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    path = TOK_DIR / "bpe_lesson01.json"
    tok.save(str(path))
    console.print(f"Saved BPE → {path}  vocab_size={tok.get_vocab_size()}")
    return path


def train_sp(corpus: list[str]) -> Path:
    section("Training SentencePiece unigram tokenizer (vocab=2048)")
    try:
        import sentencepiece as spm
    except ImportError:
        console.print("[yellow]Missing `sentencepiece`. Run: uv sync --extra ml[/]")
        raise

    # SentencePiece wants a file on disk.
    txt_path = TOK_DIR / "_corpus.txt"
    txt_path.write_text("\n".join(corpus), encoding="utf-8")

    model_prefix = TOK_DIR / "sp_lesson01"
    spm.SentencePieceTrainer.train(
        input=str(txt_path),
        model_prefix=str(model_prefix),
        vocab_size=2048,
        model_type="unigram",
        character_coverage=0.9995,
        bos_id=1, eos_id=2, unk_id=0, pad_id=3,
    )
    console.print(f"Saved SP → {model_prefix}.model")
    return Path(f"{model_prefix}.model")


def compare(bpe_path: Path, sp_path: Path) -> None:
    section("Side-by-side encoding comparison")
    from tokenizers import Tokenizer
    bpe = Tokenizer.from_file(str(bpe_path))

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_path))

    try:
        import tiktoken
        gpt4 = tiktoken.encoding_for_model("gpt-4o")
    except Exception:
        gpt4 = None

    header = f"{'string':50}  {'chars':>6}  {'bpe':>5}  {'sp':>5}  {'gpt-4o':>7}"
    console.print(f"[bold]{header}[/]")
    console.print("-" * len(header))
    for s in SAMPLE_TEXTS:
        bpe_n = len(bpe.encode(s).ids)
        sp_n = len(sp.encode(s, out_type=int))
        gpt_n = len(gpt4.encode(s)) if gpt4 else -1
        display = s.replace("\n", "\\n")[:48].ljust(50)
        console.print(f"{display}  {len(s):6}  {bpe_n:5}  {sp_n:5}  {gpt_n:7}")


def main() -> None:
    corpus = _gather_corpus()
    if not corpus:
        console.print("[yellow]No corpus found in data/sample_docs/*.md[/]")
        return
    console.print(f"Corpus: {len(corpus)} non-empty lines")

    bpe_path = train_bpe(corpus)
    sp_path = train_sp(corpus)
    compare(bpe_path, sp_path)


if __name__ == "__main__":
    main()
