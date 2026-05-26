# ml_foundations · 00 · Overview

## The mental map

```
        ┌──────────────────────────────────────────────────────────┐
        │                    a piece of text                       │
        └─────────────────────────┬────────────────────────────────┘
                                  ▼
                         ┌────────────────┐
                         │   tokenizer    │   ← lesson 01
                         │  (BPE / SP)    │
                         └───────┬────────┘
                                 ▼
                       integer token IDs
                                 │
                  ┌──────────────┼──────────────┐
                  ▼                             ▼
        ┌──────────────────┐         ┌──────────────────┐
        │ embedding lookup │         │ embedding lookup │
        │ (a matrix V × d) │         │ (a matrix V × d) │
        └────────┬─────────┘         └────────┬─────────┘
                 ▼                            ▼
        ┌──────────────────┐         ┌──────────────────┐
        │  encoder stack   │         │  decoder stack   │
        │  (BERT family)   │         │   (GPT family)   │
        │   ← lesson 03    │         │                  │
        └────────┬─────────┘         └────────┬─────────┘
                 ▼                            ▼
       fixed-dim sentence              token-by-token text
       vector for downstream                generation
       tasks (classify,
       retrieve, cluster)
                 │
                 ▼  fine-tune for retrieval
         ┌──────────────────┐
         │ sentence encoder │  ← lesson 04 — what powers your RAG
         └──────────────────┘
```

## The five steps every LLM does

1. **Tokenise** the input string → list of integers.
2. **Embed** each integer → a `d`-dim vector via a lookup table.
3. **Transform** the sequence with stacked attention + MLP blocks.
4. **Project** the final hidden state → a vocab-sized logits vector.
5. **Sample** the next token from the logits.

LangChain hides all five of these. **Each is its own art.** This
track gives you working code for steps 1, 2, and (for encoders) 3-4 of
the encoder track. Decoder pretraining is intentionally out of scope —
you don't pretrain a decoder in a weekend; you fine-tune one.

## What you can predict once you understand step 1

- **Why Turkish costs more.** GPT's BPE was trained on English-dominant data. "Merhaba" is 4 tokens; "Hello" is 1. Same word count → ~4x cost.
- **Why your prompt has more "tokens" than words.** Punctuation, whitespace, and rare words each can be their own token.
- **Why a different model's `1000 tokens` is not the same as another's.** They have different tokenizers. 1000 GPT tokens ≠ 1000 Llama tokens.
- **Why fine-tuning a vocab item is a thing.** Adding `<SUPPORT_TICKET>` as a new token + training the embedding row for it lets the model treat it as atomic.

## What you can predict once you understand step 2

- **Why two embedding models' outputs are not interchangeable.** Different training, different vector spaces.
- **Why `text-embedding-3-large` (3072 dim) ≠ `text-embedding-3-small` (1536 dim) ≠ `bge-small` (384 dim).** Three different teachers with three different alphabets.
- **Why the dim matters less than you think.** 768 vs 1536 isn't the recall lift you'd hope; the *training data* dominates.

## What you can predict once you understand steps 3-4 (encoders)

- **When fine-tuning beats prompting.** A 200-line distilbert finetune classifies 10k emails per second on a CPU; an LLM does ~10/second and costs more. Pick the right tool.
- **Why retrieval-tuned encoders beat generic ones for RAG.** They were trained on `(query, passage)` pairs. Generic encoders weren't.

## How to use this overview

Skim each step's link below. Spend 1-2 evenings per lesson. Then come
back to lessons 26, 29, and the pro capstones — you'll see them differently.

| Step | Lesson |
|---|---|
| Tokenise | [01 · Tokenizers](../01_tokenizers/README.md) |
| Embed (lookup) | [02 · Word embeddings](../02_word_embeddings/README.md) |
| Attention + the rest of the model | [03 · Transformer architecture](../03_transformer_architecture/README.md) |
| Encoder for classification | [04 · Text classification](../04_text_classification/README.md) |
| Encoder for retrieval | [05 · Fine-tuning encoders](../05_finetuning_encoders/README.md) |

## Run it

This lesson is conceptual. No script to run. Pick the next.
