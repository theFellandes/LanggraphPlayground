# ml_foundations · 05 · Fine-tuning encoders for retrieval

> Where embedding models come from.

Lesson 06's retriever uses `BAAI/bge-small-en-v1.5`. That model is an
encoder that was **contrastively fine-tuned** on `(query, passage)`
pairs to make semantically related text cluster together in vector
space. This lesson walks you through that training process on a small
dataset so you can do it for your own domain when off-the-shelf
embedders underperform.

## When you need this

Off-the-shelf embedders (BGE, E5, OpenAI `text-embedding-3-*`) work
well on general English. They underperform when:

- **Your domain has unusual vocabulary** — medical (ICD-10 codes),
  legal (citation forms), code (function names), Turkish/Arabic/CJK.
- **Your retrieval contract is unusual** — you want to retrieve
  *contradictions* instead of *paraphrases*; or *step-3-given-step-2*
  instead of *similar text*.
- **You have lots of in-domain training data** — clicks, accepted
  Stack Overflow answers, `(question, accepted_answer)` pairs.

If you have 1000+ examples of `(query, relevant_doc)` pairs, a
half-day of fine-tuning usually moves recall@5 +5 to +15 points.

## The objective in one sentence

Pull `(query, relevant_doc)` close in vector space; push
`(query, irrelevant_doc)` apart. This is **contrastive learning**.

```
loss(q, p_pos, p_neg) = -log(  exp(sim(q, p_pos)) / Σ_i exp(sim(q, p_i))  )

where p_i ranges over [p_pos, p_neg_1, p_neg_2, ...]
sim(a, b) = cos(a, b) / temperature
```

The trick is picking the negatives. **In-batch negatives** (use other
examples in the same minibatch as negatives) gets you ~80% of the
quality for ~0% of the work. **Hard negatives** (mined by an existing
embedder as "almost relevant but actually not") gets the last ~20%.

## What you'll build

Take `sentence-transformers/all-MiniLM-L6-v2` (22M params, tiny) and
fine-tune it on a small synthetic dataset of `(question, snippet)`
pairs derived from the company handbook. After fine-tuning, recall@5
on a held-out set jumps by a measurable margin.

## Run it

```bash
uv sync --extra ml
uv run python -m ml_foundations.05_finetuning_encoders.train
```

The script:

1. Builds synthetic `(query, positive_snippet)` pairs from `data/sample_docs/`.
2. Loads `sentence-transformers/all-MiniLM-L6-v2`.
3. Fine-tunes with `MultipleNegativesRankingLoss` (in-batch negatives).
4. Evaluates recall@5 on a held-out split, before vs after.
5. Saves the fine-tuned model to `data/models/embedder_finetuned/`.

The before/after recall on a tiny synthetic dataset will show a
non-zero gap; in real settings with thousands of pairs, the gap is
larger and stable.

## Try it yourself

1. Use **hard negatives** mined by the *base* model: retrieve top-10 with the unfinetuned model, mark "not the gold positive" as hard negatives, feed them in.
2. Plug the fine-tuned embedder into [Lesson 06 · RAG basics](../../lessons/06_rag_basics/README.md) — Chroma takes any `Embeddings` object — and re-measure RAG quality.
3. Compare to a `text-embedding-3-small` baseline (you'll need an OpenAI key). At what dataset size does fine-tuning win?

## Pairs with

- [Lesson 06](../../lessons/06_rag_basics/README.md) and [Lesson 29](../../lessons/29_vector_databases/README.md) — this is the model you've been using
- [Sentence-Transformers training docs](https://www.sbert.net/docs/sentence_transformer/training_overview.html)

## References

- [Sentence-BERT paper · Reimers & Gurevych 2019](https://arxiv.org/abs/1908.10084) — the bi-encoder pattern
- [SimCSE paper · Gao et al. 2021](https://arxiv.org/abs/2104.08821) — contrastive sentence embeddings
- [BGE paper · Xiao et al. 2023](https://arxiv.org/abs/2309.07597) — the lineage of `bge-small-en-v1.5`
- [MTEB benchmark](https://huggingface.co/spaces/mteb/leaderboard) — measure what you trained
