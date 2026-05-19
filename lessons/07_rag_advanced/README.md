# Lesson 07 · RAG advanced

## What you'll learn

- **`MultiQueryRetriever`** — uses the LLM to rephrase the query into several variants, then unions the hits
- **`ContextualCompressionRetriever`** — runs each retrieved chunk through an LLM filter and keeps only the parts that actually answer the question
- A quick framework for **comparing retrievers** side-by-side on the same query

## Why it matters

The retriever is the most common source of bad answers in a RAG app.
Better retrieval = better answers, no model change required. These two
techniques are the workhorses you'll reach for first.

## Key concepts

- **Query rewriting** (`MultiQueryRetriever`) — one user question becomes 3–4 paraphrases. Helpful when the user's wording doesn't match the documents' wording.
- **Result filtering** (`ContextualCompressionRetriever` + `LLMChainExtractor`) — for each retrieved chunk, ask the LLM "is anything here relevant? If so, keep only those sentences." Reduces noise in the final prompt.
- **Composability** — both are *wrappers* around a base retriever. You can stack them or swap them without touching the rest of the pipeline.

## Walk through `example.py`

The script asks the same question of three retrievers and prints the
chunks each returned:

1. **Baseline** — vanilla similarity search with `k=4`.
2. **MultiQueryRetriever** — the LLM expands the query; the union of hits is returned.
3. **ContextualCompression** — baseline retrieval then per-chunk filtering.

Compare the three outputs — you'll see the multi-query version pulls
in chunks the baseline missed, and the compressed version drops
irrelevant prose.

## Run it

```bash
uv run python -m lessons.07_rag_advanced.example
```

## Debug it

Put `breakpoint()` between the three retriever calls and compare the
returned `list[Document]` objects side-by-side.

## Try it yourself

- Combine techniques: wrap a `MultiQueryRetriever` with a `ContextualCompressionRetriever`.
- Try `EnsembleRetriever` to mix BM25 with vector similarity for hybrid search.

## Next →

[Lesson 08 · LangGraph basics](../08_langgraph_basics/README.md)
