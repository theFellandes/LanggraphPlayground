# ml_foundations · 02 · Word embeddings — Word2Vec / GloVe / FastText

> The before-time of "embedding."

Modern RAG uses **contextual** embeddings: `bge`, `e5`, OpenAI's
`text-embedding-3`. They have transformers under the hood. But the
direct ancestors — Word2Vec (2013), GloVe (2014), FastText (2016) —
are still useful when:

- You need offline lookup with **zero compute** per query (a hash map).
- You're working in a **low-resource** language where transformer encoders are scarce.
- You're studying NLP and want to see embeddings made out of nothing more than co-occurrence statistics.

This lesson trains Word2Vec (Skip-gram) on a small corpus with
`gensim`, then explores the trained vector space: nearest neighbours,
analogies, projection to 2D.

## What you'll build

1. Skip-gram Word2Vec on `data/sample_docs/*.md`, dim=100, window=5, vocab≈auto.
2. A small interactive demo: `most_similar("refund")`, `most_similar("policy")`, the analogy `"king" - "man" + "woman"` (works less well on small corpora — that's pedagogical).
3. A PCA / UMAP projection of the top 100 most-frequent words to 2D, saved as an image.

## The Skip-gram objective in one sentence

Given a centre word, predict the surrounding context words. Train a
small NN to do that. The **embedding matrix you trained becomes the
output you actually wanted** — the prediction task was just an excuse.

```
"the cat sat on the mat"
                ↑
              centre = "sat"
              context = ["cat", "on"]   (window=1)

Train:
   P(cat | sat) and P(on | sat) → high
   P(banana | sat) → low

Result:
   The hidden-layer row for "sat" — a `d`-dim vector — is the
   embedding. "sat" ends up close to other verbs that take similar
   contexts.
```

## What changes vs modern embedders

| | Word2Vec | `bge-small-en-v1.5` |
|---|---|---|
| Atom | Word | Subword token |
| Context | Local window (~5) | Whole sentence |
| Training data | Wikipedia / WMT (single-language typical) | Diverse + retrieval-tuned (`MS MARCO`, etc.) |
| Output | One vector per word | One vector per sentence/passage |
| OOV | UNK or skip | Subword fallback — never OOV |
| Best at | Word similarity, analogies | Retrieval, classification |

## Run it

```bash
uv sync --extra ml
uv run python -m ml_foundations.02_word_embeddings.train
```

Outputs:

- `data/embeddings/w2v.kv` — the trained KeyedVectors
- Console output: nearest neighbours of "refund", "policy", "agent"
- `data/embeddings/projection.png` — 2D visualisation

## Try it yourself

1. Re-train with `window=2` vs `window=10`. Window=2 captures syntactic neighbours ("dog" ≈ "cat"); window=10 captures topical ("vet" ≈ "leash").
2. Train FastText instead — `gensim.models.FastText`. It handles OOV via subword n-grams, so "refunded" works even if only "refund" appeared.
3. Use the trained vectors as the **embedder** in a Chroma index. Compare retrieval quality to `bge-small-en-v1.5`. (Spoiler: bge wins — but now you know *why*.)

## Pairs with

- [Lesson 06 · RAG basics](../../lessons/06_rag_basics/README.md) — what's *inside* the embedder you've been using
- [Lesson 29 · Vector databases](../../lessons/29_vector_databases/README.md) — the dim and OOV behaviours are choices made in lessons like this

## References

- [Word2Vec original paper · Mikolov et al. 2013](https://arxiv.org/abs/1301.3781) — short, very readable
- [GloVe paper · Pennington et al. 2014](https://nlp.stanford.edu/pubs/glove.pdf)
- [FastText paper · Bojanowski et al. 2016](https://arxiv.org/abs/1607.04606)
- [`gensim` docs](https://radimrehurek.com/gensim/) — the Python library
