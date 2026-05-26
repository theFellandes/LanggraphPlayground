# ml_foundations · 01 · Tokenizers from scratch

> Before there's a model, there's a tokenizer.

This lesson trains two tokenizers from scratch on a small corpus, then
compares them with GPT-4 / Claude / Llama tokenizers on the same text.
By the end you'll understand the **cost math** in lesson 26, the
"Turkish costs 2x" phenomenon, and how vocab size trades off speed
against compression.

## What you'll learn

1. **Why subwords beat both words and characters.** Words: huge vocab + OOV problem. Characters: long sequences + slow inference. Subwords: best of both.
2. **The three subword algorithms** — BPE, WordPiece, SentencePiece (unigram). What each is doing.
3. **Vocab size as a knob** — 32k (Llama) vs 50k (GPT-3) vs 100k (GPT-4). What you trade.
4. **The "special tokens" trick** — `<|im_start|>`, `[CLS]`, `<s>`. Why they exist; why they're not just regular tokens.

## Three algorithms in 90 seconds

### BPE (Byte-Pair Encoding) — GPT family, Llama

Start with characters. Repeatedly merge the most frequent adjacent pair
into a new symbol. Stop at `vocab_size` merges.

```
"low" "lower" "newer"
→ start chars: l, o, w, e, r, n
→ merge "lo" (frequent), "lo w" (frequent), "low er" (frequent), …
→ final vocab: l, o, w, e, r, n, lo, low, low er, new er, ...
```

Encoding is greedy: replace longest matching sub-piece.

### WordPiece — BERT family

Same idea, but the merge criterion is *likelihood* not *frequency*. Slight
quality lift; same operational shape.

### SentencePiece (unigram) — Llama, T5, Gemini

Inverts the process: start with a **huge** vocab of candidate pieces,
then iteratively prune the least useful. Probabilistic — gives you
sampling-during-tokenisation (useful for some kinds of robustness).

## Vocab size: the knob

| Vocab | Examples | Tradeoff |
|---|---|---|
| ~32k | Llama, Mistral | Smaller embedding matrix; longer sequences per text; faster training step |
| ~50k | GPT-3, GPT-3.5 | Middle ground; the historical sweet spot |
| ~100k | GPT-4, GPT-4o, o3 | Shorter sequences per text → cheaper per request; embedding matrix is huge |
| ~256k | Gemini 1.5+ | Compresses non-English much better; "Turkish cost penalty" smaller |

**Operational rule:** if 80%+ of your traffic is one language, a vocab
trained on that language compresses ~25-50% better than a multilingual
default. Worth checking once you're at scale.

## Special tokens — they're not free

Modern chat models reserve token IDs for control:

| Token | Used for |
|---|---|
| `<|im_start|>` / `<|im_end|>` | ChatML message boundaries (OpenAI) |
| `[CLS]`, `[SEP]` | BERT classification / separation |
| `<s>`, `</s>` | Sentence boundary (Llama-style) |
| `<|tool_call|>` | Tool calling (in some models) |

These are **trained as part of the model** — the embedding row for
`[CLS]` is meaningful, not random. You can't add a new special token
to a pretrained tokenizer and expect the model to do anything sensible
with it without further training.

## Run it

```bash
uv sync --extra ml
uv run python -m ml_foundations.01_tokenizers.train
```

The script:

1. Loads a small text corpus (`data/sample_docs/*.md`).
2. Trains a BPE tokenizer (vocab=2048) using `tokenizers` (the Hugging Face fast library).
3. Trains a SentencePiece unigram tokenizer (vocab=2048).
4. Encodes the same five sentences with **all four** of: your BPE, your SP, GPT-4's `tiktoken` encoder, and Claude's character estimate.
5. Prints a comparison table.

Expected output sketch:

```
       string                 chars  bpe-yours  sp-yours  gpt-4  est-claude
       'Hello, how are you?'   19      6          5         5      ~5
       'Merhaba, nasılsın?'    18     12         11         9      ~9
       'lesson 29 — vector DB' 21      8          7         8      ~8
       'def foo():\n    pass'  18      6          5         6      ~6
       'こんにちは世界'           6      4          3         4      ~4
```

(Numbers are illustrative — yours will vary by corpus.)

## Try it yourself

1. Re-train your BPE with vocab=4096 and again with vocab=512. Plot tokens/sentence vs vocab size. The curve is steep at first then flattens.
2. Train on **only Turkish** text. Compare on the same Turkish sentence — you'll see ~50% fewer tokens than GPT-4 uses.
3. Add a special token `<SUPPORT_TICKET>` to your BPE and tokenise text containing it. It should be one token.
4. Look at the trained `vocab.json` — sort by length. The longest pieces are often surprising (`" because"`, `" function"`, etc.).

## Pairs with

- [Lesson 26 Topic 1](../../lessons/26_misc/README.md) — once you understand tokenizers, the cost math stops feeling arbitrary
- [Lesson 24 · Spoken numbers](../../lessons/24_spoken_numbers/README.md) — explains why "tokenizers aren't the right tool" for some normalisation tasks
- [Andrej Karpathy · Let's build the GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — 2 hours, the canonical deep dive

## References

- [Hugging Face `tokenizers`](https://github.com/huggingface/tokenizers) — fast Rust trainer
- [Google `sentencepiece`](https://github.com/google/sentencepiece) — unigram trainer
- [`tiktoken`](https://github.com/openai/tiktoken) — OpenAI's tokenizer; ships pre-built encoders for GPT-3.5/4/4o
- [BPE original paper · Sennrich et al. 2015](https://arxiv.org/abs/1508.07909) — short, readable
- [SentencePiece paper · Kudo & Richardson 2018](https://arxiv.org/abs/1808.06226)
