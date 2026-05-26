# `ml_foundations/` — what's inside the model you're calling

> A **sibling** track to `lessons/`, not a sub-track. The LangGraph
> curriculum teaches you to *use* language models well. This track
> teaches you what they're made of — tokenizers, embeddings,
> classifiers, fine-tuning.

## Why it's separate

The repository is called *LanggraphPlayground*. The 26+ lessons under
`lessons/` all build toward "ship an agentic application with
LangChain 1.x + LangGraph 1.x." Tokenizer training and encoder
fine-tuning are different muscles:

- They use **PyTorch + Hugging Face**, not LangChain/LangGraph.
- The data, evaluation, and infrastructure look nothing like RAG/agent code.
- The audience overlaps but isn't identical (some come for agents and want to stay in the agent loop; others come because they're also curious about model internals).

Putting them under `lessons/` would confuse the curriculum's promise.
Putting them in a separate repo entirely would force you to context-switch
your environment. **Sibling folder is the compromise** — same repo
(`uv sync` covers everything), same shared infrastructure (Docker stack,
`shared/llm/`, `shared/pretty.py`), but a clean conceptual boundary.

The same approach will work for the **`gnn/`** track when you add it:
sibling folder, opt-in dependency group, separate README that
explains what it teaches and what it doesn't.

## Curriculum

| # | Lesson | What you'll build |
|---|---|---|
| 00 | [Overview](00_overview/README.md) | The conceptual map: tokenizer → embeddings → encoder → decoder → instruction-tuning. Where each piece sits in the LLM stack |
| 01 | [Tokenizers from scratch](01_tokenizers/README.md) | Train a BPE tokenizer with `tokenizers`, then a SentencePiece unigram tokenizer. Why subwords beat words. Why GPT-4's tokenizer is 100k tokens and Llama's is 32k |
| 02 | [Word embeddings — Word2Vec / GloVe / FastText](02_word_embeddings/README.md) | Train a Skip-gram Word2Vec on a small corpus. The before-time of "embedding". Why these still matter (offline lookup, low resource) |
| 03 | [Transformer architecture](03_transformer_architecture/README.md) | Implement scaled dot-product attention by hand; compare against `nn.MultiheadAttention`; visualise attention maps. Encoder vs decoder vs encoder-decoder; RoPE, flash-attention, GQA, MoE — what every modern LLM is made of |
| 04 | [Text classification (encoder-only)](04_text_classification/README.md) | Fine-tune `distilbert-base-uncased` on an intent task. Train/eval loop. Why this is the right tool when "the LLM" is overkill |
| 05 | [Fine-tuning encoders for embedding](05_finetuning_encoders/README.md) | Contrastive fine-tune `sentence-transformers/all-MiniLM` on a small triplet dataset. How modern embedding models are actually trained |

This is a **starter pack**, not a full deep-learning course. Each
lesson is 1-2 evenings, end-to-end, with a runnable script. For a
proper DL curriculum, see the references at the bottom.

## What this track teaches you about LangGraph

By the end:

- You understand what `text_embedding-3-small` actually does — and why two embedders' vectors aren't interchangeable.
- You can train an embedding model on your **own** corpus when off-the-shelf ones underperform (lesson 29's MTEB leaderboard isn't the answer for every domain).
- You know when a fine-tuned encoder beats an LLM prompt (cheaper, faster, better for narrow classification).
- You understand the tokenizer math behind lesson 26's cost estimation — Turkish prose costs ~2x what English prose costs because the tokenizer wasn't trained on it.

## Setup

These lessons add ML deps. They live in an `ml` extra so the core
LangGraph install stays slim:

```bash
uv sync --extra ml
```

What you get:

- `tokenizers` — the fast Rust-backed tokenizer trainer (Hugging Face)
- `sentencepiece` — Google's BPE/unigram trainer
- `torch` + `transformers` + `datasets` — for lessons 03-04
- `sentence-transformers` — contrastive embedding fine-tuning
- `gensim` — Word2Vec / GloVe / FastText

## How each ml_foundations lesson is laid out

Same shape as `lessons/`:

```
ml_foundations/NN_topic/
├── README.md      ← What you'll learn, why, how to run
├── train.py       ← Runnable script: load → train → save
└── eval.py        ← (optional) measure what you just trained
```

## What this track deliberately does NOT cover

| Topic | Where to go |
|---|---|
| **Pretraining from scratch** | Not in 1 weekend. See [`nanoGPT`](https://github.com/karpathy/nanoGPT) (Karpathy) and Hugging Face's `accelerate` docs |
| **RLHF / DPO / KTO** | Hugging Face `trl` library; Anthropic's [constitutional AI paper](https://arxiv.org/abs/2212.08073) for theory |
| **Quantisation** (4-bit, 8-bit) | `bitsandbytes`, `auto-gptq`, `llama.cpp`. Useful when you self-host |
| **Inference servers** (vLLM, TGI) | Production deployment of open-weights models |
| **Mechanistic interpretability** | Anthropic's [transformer-circuits.pub](https://transformer-circuits.pub/), Neel Nanda's tutorials |

## Pairs with

- **[Lesson 26 Topic 1](../lessons/26_misc/README.md)** — tokeniser-aware cost math becomes intuitive once you've trained one
- **[Lesson 29](../lessons/29_vector_databases/README.md)** — pick of embedder downstream of which embedding model you understand
- **[`skills/llm-expert`](../skills/llm-expert/SKILL.md)** — when to fine-tune vs prompt vs RAG; this track gives you the muscle to act on that decision

## Why "ml_foundations" and not "nlp"

"NLP" implies a wider scope (parsing, NER, summarisation, MT) that
isn't covered here. "ml_foundations" names what's actually inside:
the foundational pieces of modern ML-for-text systems. When you add
the GNN track, name it `gnn/` (not `graph_nlp/`) for the same reason —
narrow names hold up better as content grows.

## References

- [Andrej Karpathy · "Let's build the GPT Tokenizer"](https://www.youtube.com/watch?v=zduSFxRajkE) — the canonical 2 hours
- [Hugging Face NLP course](https://huggingface.co/learn/nlp-course) — free, comprehensive
- [d2l.ai · Natural Language Processing chapters](https://d2l.ai/) — textbook-quality
- [Sebastian Ruder · NLP newsletter archive](https://ruder.io/) — the field's news for the last decade
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — embedding model benchmarks
