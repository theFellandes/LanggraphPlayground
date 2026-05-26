# ml_foundations · 03 · Transformer architecture

You've trained a tokenizer (lesson 01). You've trained Word2Vec
(lesson 02). Both produce *static* embeddings — one vector per word,
regardless of context. The word "bank" has the same vector in "river
bank" and "investment bank."

The transformer fixes that. It produces **contextual** embeddings:
each token's vector depends on the whole input sequence. This is the
architecture behind BERT, GPT, Claude, Gemini, Llama, and basically
every LLM since 2017's *Attention Is All You Need*.

This lesson is **how it actually works** — the math, the wiring, the
modern variants. By the end you'll write a tiny working self-attention
block in PyTorch and run it on a real sentence.

## What you'll learn

1. **Why RNNs lost** — sequential dependency, vanishing gradients, no parallelism on the sequence axis.
2. **The self-attention math** — Q, K, V matrices; the scaled dot-product formula in 6 lines of NumPy.
3. **Multi-head attention** — what "heads" mean and why 8/12/96 of them.
4. **Position embeddings** — sinusoidal, learned, ALiBi, RoPE; why "the model doesn't know order without help."
5. **The full transformer block** — attention + residual + LayerNorm + MLP, in the right order.
6. **Three architectures** — encoder-only (BERT), decoder-only (GPT/Claude), encoder-decoder (T5).
7. **Modern improvements** — flash-attention, multi-query / grouped-query attention, sliding window, MoE.
8. **How this connects to LangGraph** — every `llm.invoke(...)` runs N transformer blocks for every token; the cost math from lesson 26 finally makes sense.

## Part 1 · Why we needed something new

Before 2017, "sequence model" meant **RNN / LSTM / GRU**:

```
h_0 →  h_1 →  h_2 →  ...  →  h_N
       ↑      ↑              ↑
       x_1    x_2            x_N
```

Each hidden state depended on the previous one. Three problems:

1. **No sequence-axis parallelism.** You couldn't compute `h_5` until
   `h_4` was done. GPUs sat idle.
2. **Vanishing / exploding gradients.** Information from `x_1` had to
   survive 100 multiplications by recurrent weights before reaching `h_100`.
   LSTM helped; it didn't solve it.
3. **Limited context.** Most LSTMs effectively forgot anything more than
   ~30-50 tokens back.

The transformer's answer is brutal: **drop recurrence entirely**.
Process all positions in parallel; let every token directly look at
every other token through *attention*.

## Part 2 · Self-attention in 6 lines

Take a sequence `X ∈ ℝ^{n × d}` (n tokens, each d-dim). Project it into
three "views" with learned matrices `W_Q, W_K, W_V ∈ ℝ^{d × d_k}`:

```
Q = X · W_Q       # "what am I asking?"
K = X · W_K       # "what do I offer?"
V = X · W_V       # "the actual content I carry"
```

The output is

```
Attention(Q, K, V) = softmax(Q · K^T / sqrt(d_k)) · V
```

That's it. Six lines. Let's unpack:

- `Q · K^T ∈ ℝ^{n × n}` — each entry `[i, j]` is the dot product of
  token `i`'s query with token `j`'s key. **A similarity score.**
- Divide by `sqrt(d_k)` — keeps softmax inputs in a sane range so
  gradients don't vanish.
- `softmax` (row-wise) — turns each row into a probability distribution
  over the other tokens. *"How much attention does token i pay to
  token j?"*
- Multiply by `V` — each row of the output is a weighted average of the
  value vectors, weighted by attention.

```
       Q       ·         K^T       =     scores      → softmax →  attn        ·    V         =   output
   ┌───────┐         ┌───────┐         ┌───────┐                ┌───────┐         ┌───────┐    ┌───────┐
   │ n × d │    ·    │ d × n │    =    │ n × n │     ──→        │ n × n │    ·    │ n × d │  = │ n × d │
   └───────┘         └───────┘         └───────┘                └───────┘         └───────┘    └───────┘
```

In plain English: **for each output position, look at every input
position, weight by how relevant they are, blend.**

### Why "self"-attention?

Because Q, K, V all come from the same X. In *cross*-attention
(encoder→decoder), Q comes from one sequence and K/V from another.

## Part 3 · Multi-head attention

One attention "head" looks at the sequence one way. Multiple heads
let the model attend to different patterns at once:

```
head_i = Attention(X · W_Q^i, X · W_K^i, X · W_V^i)            # i = 1..h
MultiHead(X) = Concat(head_1, ..., head_h) · W_O
```

Each head has its own `(W_Q^i, W_K^i, W_V^i)`. Outputs are concatenated
and projected back to `d`. Typical numbers:

| Model | d | h | d_k per head |
|---|---|---|---|
| BERT-base | 768 | 12 | 64 |
| GPT-3 175B | 12288 | 96 | 128 |
| Llama-2-7B | 4096 | 32 | 128 |
| Claude / GPT-4 (rumored) | similar shape; details proprietary |

**The intuition** (post-hoc, but useful): one head learns syntactic
agreement (subject ↔ verb), another tracks coreference, another picks
out punctuation patterns. Different heads at different layers
specialise differently. You can visualise this with attention maps.

## Part 4 · Position embeddings — "the model doesn't know order"

The attention mechanism is **permutation-invariant**: shuffle the
tokens and you get a shuffled output. Order has to be injected
explicitly.

Four approaches:

### 4.1 · Sinusoidal (original transformer)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Added to the token embedding. The cute property: positional offsets are
linear transformations of each other, so the model can in theory
generalise to longer sequences than it was trained on (with caveats).

### 4.2 · Learned position embeddings (BERT, GPT-2)

Just a lookup table `PosEmb ∈ ℝ^{max_seq × d}`. Simpler, slightly
better in-distribution, doesn't generalise to longer sequences.

### 4.3 · RoPE (Rotary Position Embeddings) — Llama, GPT-NeoX, modern LLMs

Instead of *adding* position info to embeddings, **rotate the
Q and K vectors in 2D planes** based on position. The rotation
preserves the dot product when positions are equal and reduces it
proportionally as positions differ. Practically: relative position
falls out of the dot product naturally.

```
For each 2D pair (x_2i, x_2i+1):
    apply rotation by angle θ_i · pos
```

Why it wins: extends easily (lengths beyond training) and works well
empirically. **2024-onwards default for most decoder-only LLMs.**

### 4.4 · ALiBi (Attention with Linear Biases) — used in BLOOM, MPT

Add a fixed bias to the attention scores that *linearly decreases*
with the distance between query and key positions. No learnable
positional parameters. Strong length extrapolation.

## Part 5 · The full transformer block

One block in a modern model:

```
                   ┌──────────────────┐
            X  →   │  LayerNorm       │ ──┐
                   └──────────────────┘   │
                            │              │
                   ┌──────────────────┐   │
                   │  Multi-Head Attn │   │
                   └──────────────────┘   │
                            │              │
                            +  ←───────────┘   (residual connection)
                            │
                   ┌──────────────────┐ ──┐
                   │  LayerNorm       │   │
                   └──────────────────┘   │
                            │              │
                   ┌──────────────────┐   │
                   │  FFN  (MLP)      │   │
                   │  d → 4d → d      │   │
                   └──────────────────┘   │
                            │              │
                            +  ←───────────┘   (residual connection)
                            │
                            ↓
                          output
```

Components in detail:

- **LayerNorm**: per-token, per-feature normalisation. Stabilises
  training. Modern "pre-norm" placement (before each sub-block, shown
  above) trains better than the original "post-norm" placement
  (after, with residual).
- **Residual connection (`+`)**: passes the input through, unchanged.
  Gives the model the option of "if this sub-block isn't helpful, skip
  it." Crucial for deep nets to train.
- **FFN (Feed-Forward Network)** = `Linear(d → 4d) → activation → Linear(4d → d)`.
  Modern activation: GELU, SwiGLU, GeGLU. About 2/3 of the parameters in
  a transformer live in the FFN, not in attention.

A model is just N such blocks stacked:

| Model | N (layers) | d | h | Params |
|---|---|---|---|---|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |
| GPT-2 small | 12 | 768 | 12 | 117M |
| GPT-3 175B | 96 | 12288 | 96 | 175B |
| Llama-2-7B | 32 | 4096 | 32 | 7B |
| Llama-2-70B | 80 | 8192 | 64 | 70B |

## Part 6 · Three architectures

The same block, three wiring patterns.

### 6.1 · Encoder-only (BERT family)

```
input tokens → embeddings + pos
             → N × Block (bidirectional attention: token i sees ALL tokens)
             → output: contextual vector per token
```

Used for:
- **Classification** (lesson 04 — DistilBERT for intent)
- **Retrieval** (lesson 05 — sentence-transformers, your RAG embedder)
- **NER, entity tagging, fill-in-the-blank**

Trained with **masked language modelling**: randomly mask 15% of
tokens, predict them. Sees the whole context, so "bidirectional."

### 6.2 · Decoder-only (GPT, Claude, Llama)

```
input tokens → embeddings + pos
             → N × Block (CAUSAL attention: token i sees tokens 1..i ONLY)
             → output: predicted next-token distribution
             → sample → append → repeat
```

The "causal" mask prevents tokens from looking at future positions
during training, so the model can only generate left-to-right.

Used for:
- **Chat / completion** (every `llm.invoke(...)` in this repo)
- **Code generation, agentic tool calling**
- **In-context learning** (few-shot prompting)

Trained with **next-token prediction** on huge corpora.

### 6.3 · Encoder-decoder (T5, BART, Whisper)

Two stacks. Encoder reads the input (bidirectional). Decoder generates
the output (causal). The decoder attends to the encoder's outputs via
**cross-attention** between its own blocks.

Used for:
- **Translation, summarisation** (text-in → text-out)
- **Whisper-style speech-to-text**

Largely overtaken by decoder-only models prompted with instructions,
*except* in audio/multimodal where the encoder's structure is useful.

## Part 7 · Modern improvements (2023-2026)

The vanilla transformer from 2017 has been heavily optimised. The
ones you'll encounter in production:

### 7.1 · Flash attention

Mathematically identical to standard attention. **Reorders the
computation** to never materialise the full `n × n` attention matrix in
memory — instead, computes attention in tiles that fit in SRAM, fusing
the softmax with the matmuls. 2-4× speedup, 5-20× memory reduction on
long sequences. **Just turn it on**: PyTorch's `scaled_dot_product_attention`
picks the Flash kernel automatically on supported hardware.

### 7.2 · Multi-Query / Grouped-Query Attention (MQA / GQA)

In standard MHA, each head has its own K and V matrices. **MQA shares K
and V across all heads** — different Q per head, same K/V. **GQA**
groups: e.g. 32 Q-heads but 8 KV-heads (groups of 4). Massively cuts
the KV cache at inference time, with minor quality loss. Llama 2/3,
Mistral, most production decoder-only models in 2026 use GQA.

### 7.3 · Sliding-window attention (Mistral, Longformer)

Token `i` only attends to the last `w` tokens (e.g., w=4096). At depth
`D` layers, the effective context is `D × w`. Linear-time per token
instead of quadratic. Trades exact long-range for scalability.

### 7.4 · KV caching (decoder inference)

At inference, every new token's Q must attend to *all previous* K and
V. If you re-computed K and V every step, you'd do O(n²) work for a
single completion. The **KV cache** stores K and V across steps so each
new token is O(n) instead. The "context window" cost story you read
about — *quadratic during training, linear during inference with KV
cache* — comes from this.

### 7.5 · Mixture of Experts (MoE)

Instead of one FFN per layer, have N "expert" FFNs and a small router.
Each token activates only `k` of the N (e.g., 2 of 8). Total params
huge (e.g., Mixtral 8×7B = 47B), active params small (~13B). Speed of
a 13B at the quality of a 47B. Mixtral, Grok, DeepSeek, GPT-4 (rumored)
use MoE.

### 7.6 · Long context tricks

- **YaRN, NTK-aware scaling** — extend RoPE-trained models to 4-16×
  their training context with minimal fine-tune.
- **Ring attention** (multi-GPU) — shard the sequence axis across GPUs.
- **Mamba / SSMs** — recurrent state-space models that compete with
  attention at very long contexts; not yet the dominant architecture
  but watch this space.

## Part 8 · Counting parameters

A useful exercise. For a model with `L` layers, `d` model dim, `h`
heads, vocab size `V`, FFN expansion 4:

| Block | Params per layer | Per model |
|---|---|---|
| Attention Q/K/V/O projections | 4 · d² | 4 L d² |
| FFN (d → 4d → d) | 8 · d² | 8 L d² |
| LayerNorms | ~2d | 4 L d (negligible) |
| Embedding + LM head | V · d (often tied) | V · d |

Total ≈ **12 L d² + V d**.

Llama-7B: L=32, d=4096, V=32000 → `12 · 32 · 4096² + 32000 · 4096` =
**6.6B + 0.13B ≈ 6.7B**. Matches.

**The implication for cost (lesson 26):** every token of output runs
~12 L d² operations through the transformer. A 70B model with 4096
context, generating 200 tokens, does ~200 · 80 · 8192² · 12 ≈ **130
TFLOPs** per response. The hyperscale fast inference systems make this
look cheap; it isn't free.

## Part 9 · How this connects to everything else in the repo

- **Lesson 01 (tokenizers)** — produces the integer IDs that hit the
  embedding lookup at the bottom of the transformer.
- **Lesson 02 (word embeddings)** — the *static* version that the
  embedding lookup row generalises. Word2Vec is "the bottom layer of a
  transformer, frozen, with no attention above it."
- **Lesson 04 (text classification)** — fine-tunes an encoder-only
  transformer (DistilBERT). Now you know what's inside.
- **Lesson 05 (encoder fine-tuning)** — contrastive training tweaks the
  same encoder for retrieval-friendly outputs.
- **Lesson 26 Topics 1-2 (token/cost)** — the cost math is "tokens ×
  layer-FLOPs"; you can predict cost from architecture now.
- **Lesson 33 (vector DB internals)** — the vectors you index are the
  outputs of an encoder transformer pooled to one vector per passage.

## Run it

```bash
uv sync --extra ml
uv run python -m ml_foundations.03_transformer_architecture.example
```

The script:

1. **Implements scaled dot-product attention** in 10 lines of pure PyTorch (no `nn.MultiheadAttention` cheat).
2. **Builds a tiny single-block transformer** (1 layer, 4 heads, d=64, n=8 tokens) and runs a forward pass on a real sentence tokenised by GPT-4o's `tiktoken`.
3. **Visualises the attention map** — heatmap of which tokens attend to which (saved as `data/transformer/attn_map.png`).
4. **Compares manual attention vs PyTorch's `nn.MultiheadAttention`** — they should agree to 6 decimal places.

Sample output sketch:

```
[STEP 1] tokenize 'The cat sat on the mat.'
ids: [791, 8415, 7731, 389, 279, 5634, 13]  (7 tokens)

[STEP 2] manual self-attention vs nn.MultiheadAttention
max abs diff: 4.7e-07  ✓

[STEP 3] attention map saved → data/transformer/attn_map.png
```

## Try it yourself

1. Increase `n_heads` from 4 to 8 (must divide `d_model`). What changes in the attention map?
2. Replace the manual softmax with **causal masking** — `mask = torch.triu(torch.ones(n, n), diagonal=1).bool()` set to `-inf` before softmax. You've built the decoder pattern.
3. Implement **RoPE** for the Q and K projections (see references). Compare attention maps with/without RoPE — RoPE attention will favour local positions even with no position embeddings added.
4. Stack 6 blocks. Run gradient through a tiny next-token loss. You've built a baby GPT.
5. Swap PyTorch's `F.scaled_dot_product_attention` in for the manual version. Same numbers, much faster on GPU (it's Flash under the hood).

## Anti-patterns

| Smell | Fix |
|---|---|
| Confusing attention with embedding | Attention is the *operator*, embedding is the *input table*. They're different layers |
| "More heads = better" | Beyond ~16, diminishing returns. Many heads end up redundant |
| Stacking layers without residuals | A 12-layer pre-norm transformer trains; without residuals 6 layers won't |
| Using sinusoidal PE in a long-context model | Switch to RoPE or ALiBi for ≥8k contexts |
| Forgetting the causal mask in a decoder | The model "cheats" by seeing future tokens during training, then collapses at inference |
| Implementing softmax manually with `exp / sum(exp)` | Numerical instability. Always subtract the max first, or use `F.softmax` |
| Treating heads as humanly-interpretable | Probes find some interpretable heads, but most are messy. Don't over-anthropomorphise |

## Pairs with

- **[ml_foundations/01 · Tokenizers](../01_tokenizers/README.md)** — the layer below
- **[ml_foundations/04 · Text classification](../04_text_classification/README.md)** — uses an encoder; now you know what's inside
- **[ml_foundations/05 · Fine-tuning encoders](../05_finetuning_encoders/README.md)** — contrastive training of the model you just studied
- **[Lesson 26 · Misc](../../lessons/26_misc/README.md)** — Topics 1-2 (token/cost) are downstream of this lesson's architecture math
- **[Lesson 33 · Vector DB internals](../../lessons/33_vector_database_internals/README.md)** — the embeddings you index come from an encoder transformer

## References

### The papers

- [Vaswani et al. · Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — the canonical paper, still readable
- [Devlin et al. · BERT (2018)](https://arxiv.org/abs/1810.04805) — encoder-only + masked LM
- [Radford et al. · GPT-2 (2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) — decoder-only at scale
- [Su et al. · RoFormer / RoPE (2021)](https://arxiv.org/abs/2104.09864) — rotary position embeddings
- [Dao et al. · FlashAttention (2022)](https://arxiv.org/abs/2205.14135) — the speedup
- [Ainslie et al. · GQA (2023)](https://arxiv.org/abs/2305.13245) — grouped-query attention
- [Touvron et al. · Llama 2 (2023)](https://arxiv.org/abs/2307.09288) — modern decoder-only recipe in detail

### Walkthroughs

- [Andrej Karpathy · Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 2 hours, builds nanoGPT live
- [Jay Alammar · The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the standard visual explanation
- [Lilian Weng · Transformer Family](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — exhaustive reference of variants
- [Anthropic · Transformer Circuits Thread](https://transformer-circuits.pub/) — mechanistic interpretability
- [Sebastian Raschka · Understanding LLMs](https://magazine.sebastianraschka.com/) — the newsletter

### Code

- [`nanoGPT`](https://github.com/karpathy/nanoGPT) — Karpathy's minimal GPT in PyTorch (~600 lines total)
- [`transformer_engine`](https://github.com/NVIDIA/TransformerEngine) — NVIDIA's optimised kernels
- [`flash-attention`](https://github.com/Dao-AILab/flash-attention) — the reference implementation

## Next →

[`04 · Text classification`](../04_text_classification/README.md) — now you can fine-tune the transformer you just understood.
