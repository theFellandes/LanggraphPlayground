"""Contrastive fine-tune a small sentence-transformer.

Run:
    uv sync --extra ml
    uv run python -m ml_foundations.05_finetuning_encoders.train
"""

from __future__ import annotations

import random
from pathlib import Path

from shared import settings
from shared.pretty import console, section

OUT_DIR = settings.data_dir / "models" / "embedder_finetuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)


PAIRS = [
    ("How many PTO days do I get?",        "Full-time employees receive 20 paid time off (PTO) days per year."),
    ("Refund policy?",                     "Refunds are processed within 7 business days for orders under $500."),
    ("Remote work allowed?",               "Employees may work remotely up to 3 days per week with manager approval."),
    ("What is a StateGraph?",              "A StateGraph is LangGraph's stateful workflow primitive — nodes plus typed shared state."),
    ("Difference between LCEL and LangGraph?",
     "LCEL composes Runnables with the pipe. LangGraph adds cycles, branches, and persistence."),
    ("How does retrieval work?",
     "Retrieval embeds the query, looks up the nearest vectors in a vector store, and returns the matching chunks."),
    ("What is a checkpointer?",
     "A checkpointer saves graph state after every step so the graph can resume from where it left off."),
    ("Parental leave policy?",
     "New parents are entitled to 16 weeks of paid leave following the birth or adoption of a child."),
    ("What's a tool?",
     "A tool is a Python function decorated with @tool that the LLM can call by name with structured arguments."),
    ("HITL?",
     "Human-in-the-loop pauses the graph with interrupt(...) and resumes when the operator provides a decision."),
]


def _split(seed: int = 42, eval_n: int = 3):
    random.seed(seed)
    shuffled = PAIRS.copy()
    random.shuffle(shuffled)
    return shuffled[eval_n:], shuffled[:eval_n]


def _eval_recall(model, eval_pairs, all_docs):
    import numpy as np
    queries = [q for q, _ in eval_pairs]
    docs = list(all_docs)
    Q = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    D = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    scores = Q @ D.T
    hits = 0
    for i, (_, gold) in enumerate(eval_pairs):
        top_idx = np.argsort(-scores[i])[:5]
        if gold in [docs[j] for j in top_idx]:
            hits += 1
    return hits / len(eval_pairs)


def main() -> None:
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        console.print("[yellow]Missing sentence-transformers. Run: uv sync --extra ml[/]")
        raise

    train_pairs, eval_pairs = _split()
    all_docs = [doc for _, doc in PAIRS]

    section("Loading base model")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    section("Eval BEFORE fine-tuning")
    r5_before = _eval_recall(model, eval_pairs, all_docs)
    console.print(f"recall@5 = {r5_before:.2f}")

    section("Fine-tuning with MultipleNegativesRankingLoss")
    examples = [InputExample(texts=[q, d]) for q, d in train_pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=4)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=10, warmup_steps=2)

    section("Eval AFTER fine-tuning")
    r5_after = _eval_recall(model, eval_pairs, all_docs)
    console.print(f"recall@5 = {r5_after:.2f}    (Δ = {r5_after - r5_before:+.2f})")

    section("Saving")
    model.save(str(OUT_DIR))
    console.print(f"Saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
