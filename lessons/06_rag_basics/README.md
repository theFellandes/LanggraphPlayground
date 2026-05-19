# Lesson 06 · RAG basics

## What you'll learn

- The six stages of a vanilla RAG pipeline: **load → split → embed → store → retrieve → generate**
- How to use `RecursiveCharacterTextSplitter` and what `chunk_size` / `chunk_overlap` actually mean
- How to build a local Chroma vector store with FastEmbed (no extra API key)
- How to wire a retriever into an LCEL chain so the LLM sees retrieved context

## Why it matters

RAG is the default architecture for "talk to my documents" use cases.
Every later lesson and capstone that touches knowledge (the support
bot, the research assistant, the QA API) builds on this exact
pipeline.

## Key concepts

- **Document loader** — wraps a source (file, URL, DB) into LangChain `Document` objects with `page_content` + `metadata`.
- **Text splitter** — slices documents into chunks small enough to fit in the model's context. `RecursiveCharacterTextSplitter` tries paragraph → sentence → word boundaries in turn.
- **Embeddings** — turn each chunk into a vector. We use `FastEmbedEmbeddings("BAAI/bge-small-en-v1.5")` — runs locally on CPU, ~33 MB model, no API key.
- **Vector store** — stores vectors + metadata and supports k-NN search. Chroma persists to disk, so the second run reuses the index.
- **Retriever** — a `Runnable` over a vector store. Calling `.invoke("query")` returns a `list[Document]`.
- **Context formatting** — the retriever returns docs; you concatenate their `page_content` into the `context` slot of your prompt.

## Walk through `example.py`

1. **`build_vectorstore()`** — loads the two markdown files from `data/sample_docs/`, splits them, embeds with FastEmbed, and writes a Chroma index under `data/chroma_lesson_06/`.
2. **`ask(store, question)`** — composes an LCEL chain whose `context` slot is filled by the retriever before the prompt is rendered. The `{context: retriever | format_context, question: RunnablePassthrough()}` dict-syntax is the LCEL idiom for "build these two fields in parallel."
3. **`main()`** — asks three questions: one answered by the company handbook, one by the LangGraph intro, one by the handbook again.

## Run it

```bash
uv run python -m lessons.06_rag_basics.example
```

The first run downloads the FastEmbed model (~33 MB) — subsequent runs are instant.

## Debug it

Put `breakpoint()` inside `ask()` right after the chain definition and
inspect what the retriever returns:

```text
ipdb> pp retriever.invoke("How many days of PTO?")
```

This is the fastest way to diagnose "model gave a bad answer" — most
of the time the retriever returned irrelevant chunks.

## Try it yourself

- Drop a new markdown file into `data/sample_docs/` and add it to `SOURCE_DOCS`. Delete `data/chroma_lesson_06/` first so the index rebuilds.
- Lower `chunk_size` to `150` and see how it affects answer quality.

## Next →

[Lesson 07 · RAG advanced](../07_rag_advanced/README.md)
