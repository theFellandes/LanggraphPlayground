---
name: llm-expert
description: Broad technical expertise on large language models — transformer / MoE / state-space / hybrid architectures, pretraining and post-training (SFT, RLHF, DPO, RLAIF, Constitutional AI), inference optimization (quantization, speculative decoding, KV-cache, prefix caching, batching, PagedAttention), evaluation (lm-eval-harness, MMLU/MMLU-Pro/GPQA/SWE-bench/AIME, LLM-as-judge), alignment + safety, scaling laws, the frontier model landscape (May 2026), and the decision tree of when to fine-tune vs prompt vs RAG vs adapter vs distill. Use when the user asks about LLM internals, model selection, training methods, inference cost / speed, evaluation methodology, or "how does X actually work" for a modern LLM concept. Pairs with senior-prompt-engineer for prompt design and scientific-paper-researcher for paper-backed claims.
---

# LLM Expert

Production knowledge of how modern LLMs work, how to evaluate them,
how to make them faster / cheaper, and how to pick the right model
for the job. May 2026 stack.

## When to invoke

- "How does {MoE / state-space / speculative decoding / RLHF / DPO / KV cache / RoPE / GQA} work?"
- "Which model should I use for X?"
- "How do I make inference cheaper / faster?"
- "Should I fine-tune, use RAG, or just prompt better?"
- "How would I evaluate this model on …?"
- "What's the latest with {Claude / GPT / Gemini / Llama / Qwen / DeepSeek / Mistral}?"

## Operating principles

1. **Cost has three axes:** training, inference (per token), latency (per request). Optimising one usually hurts another. State your axis up front.
2. **Always give a concrete recommendation.** "It depends" is a cop-out unless you've enumerated the variables.
3. **Cite the paper when there is one.** "DPO (Rafailov et al., 2023)" beats "preference optimisation."
4. **Benchmarks lie.** A model that's +3pp on MMLU but -10pp on your eval is worse for you. Always ask: what's the user's actual eval?
5. **Default to the cheapest model that meets the bar.** Don't reach for Opus when Haiku will do.
6. **Be honest about what's settled vs what's marketing.** "Test-time compute scaling" is real; "AGI by Q3" is marketing.

## 1 · Architectures

### Transformer (still the substrate)

- **Decoder-only** (GPT-style) is the dominant production shape: causal attention, next-token prediction, KV cache for generation.
- **Encoder-only** (BERT-style) survives in embeddings and classification heads; not used for generation.
- **Encoder-decoder** (T5-style) niche; mostly translation and dense passage retrieval reranking.

### Attention variants (2022 → 2026)

| Variant | What it is | Used by |
|---|---|---|
| **MHA** (multi-head attention) | original — N heads, each with own K/V | GPT-2, original Llama |
| **MQA** (multi-query) | N query heads, **1** shared K/V — drops cache size N× | PaLM, Falcon |
| **GQA** (grouped-query) | N query heads, **G** shared K/V groups — sweet spot | Llama 3+, most production models |
| **MLA** (multi-head latent attention) | compress K/V into a low-rank latent — bigger cache savings | DeepSeek-V2/V3, R1 |
| **Sliding-window attention** | each token only sees N previous tokens | Mistral 7B (combined with global) |
| **Flash Attention 2 / 3** | exact, IO-aware kernel — same math, way faster | training and inference everywhere |

**Quick rule:** for any new training run today, use **GQA + RoPE + RMSNorm + SwiGLU**. That's the consensus default.

### Mixture-of-Experts (MoE)

- N total parameters, only k experts active per token (k usually 2–8).
- "Activated parameters" (e.g. Mixtral 8x7B has 47B total, ~13B active per token) — cost scales with activated, not total.
- **Routing collapse** is the eternal problem: a few experts get all the traffic. Mitigations: auxiliary load-balancing loss, expert-choice routing, fine-grained experts (DeepSeek-V3).
- Active examples (2026): DeepSeek-V3 / R1, Mixtral 8x22B, Qwen2.5 MoE, Snowflake Arctic, gpt-oss-120B.

### State-space / hybrid

- **Mamba / Mamba-2** — linear-time recurrence, constant memory per step. Strong at long-context, weaker at in-context recall.
- **Hybrid Mamba-Transformer** (e.g., Jamba, Zamba) — attention layers for recall + state-space for context length. Best of both for some workloads.
- **RWKV** — RNN with transformer-like training. Niche.

State-space is **not** a drop-in transformer replacement — needs a different mental model and different evals.

### Long-context tricks

- **Position encodings:** RoPE → YaRN / ABF (interpolation/extrapolation) for stretched context.
- **Ring Attention** — split sequence across devices, communicate KVs ring-wise. Powers 1M+ context training.
- **Mistral / Gemini 1.5 1M+** — uses some combination of MoE + sliding-window + interpolation.
- **The benchmark gap:** models that pass needle-in-haystack often still fail at multi-needle, reasoning-over-context, or summarisation-over-context. Always test on your specific workload.

## 2 · Training

### Pretraining (~1% of practitioners ever do this end-to-end)

- **Chinchilla optimal:** ~20 tokens per parameter for a *training-compute*-optimal run. Production models train past Chinchilla (often 100–1000+ tok/param) to make *inference* cheaper.
- **Curricula:** general web → curated/code/math → instruction-style continued pretraining. Recent SOTA puts a lot of "reasoning trace" data near the end.
- **Tokeniser choice matters.** Byte-level BPE (Llama, GPT-4) is default. SentencePiece variants for multilingual.

### Post-training (where models actually become useful)

```
base model
    │
    ▼
SFT (supervised fine-tuning on instruction-response pairs)
    │
    ▼
Preference learning: pick one of:
    │     RLHF        — reward model + PPO. Powerful, finicky, expensive.
    │     DPO         — direct preference loss; no reward model; cheaper, very competitive.
    │     IPO / KTO   — DPO variants with different loss shape; robust to noisy prefs.
    │     RLAIF       — preferences from a strong LLM judge, not humans.
    │     Constitutional AI (Anthropic) — model critiques its own outputs against a constitution.
    │
    ▼
RLVR / process supervision  — for math / code, reward on verifiable outcomes.
    │                          Powers o1-style "thinking" training.
    ▼
production model
```

**Practical defaults (2026):**
- For a new model from scratch: **SFT → DPO**. RLHF still wins at the bleeding edge but the gap is small.
- For reasoning capability: train on **reasoning traces** + **RLVR** on math/code/science with verifiable rewards.
- For domain adaptation (already-aligned model): **LoRA** SFT, often skip the preference stage.

### Parameter-efficient fine-tuning

| Method | Trainable params | When |
|---|---|---|
| **Full FT** | 100% | Have GPU budget and full corpus; want max quality |
| **LoRA** | ~0.1–1% | The default for most fine-tuning. Mergeable into base. |
| **QLoRA** | LoRA + 4-bit base | Fine-tune 70B on one A100 |
| **Adapters / IA³** | <0.1% | When you'll swap many task-specific weights at runtime |
| **Prefix / prompt tuning** | tiny | Mostly historical now — DPO + LoRA usually beats |
| **DoRA** | LoRA + magnitude/direction decomp | Better than LoRA at same param count, slightly slower |

**Don't fine-tune for facts.** RAG is almost always better for factual recall. Fine-tune for style, format, behaviour, domain language.

## 3 · Inference optimisation

The cost lever menu, roughly biggest impact first:

1. **Quantization** — 16-bit → 8-bit (negligible quality loss) → 4-bit (small loss with good quant) → 2-bit (research). Schemes: GPTQ, AWQ, GGUF (llama.cpp), bitsandbytes, FP8 (Hopper+).
2. **KV-cache compression** — GQA / MQA / MLA (architectural); evict / sparsify (post-hoc, mixed quality).
3. **Speculative decoding** — small "draft" model proposes N tokens, big model verifies in parallel. ~2–3× speedup for free quality. Variants: Medusa, EAGLE, draft-then-verify with the same model.
4. **Continuous batching / PagedAttention** — vLLM-style. Saturates GPU vs static batching's idle slots.
5. **Prefix / prompt caching** — cache K/V for repeated system prompts. Massive for chat workloads. Anthropic, OpenAI, and most inference engines support this.
6. **Chunked prefill** — overlap prefill of long prompts with decode of other requests. Throughput win.
7. **MoE-aware serving** — experts on different devices, expert-parallel routing. DeepSpeed-MoE, vLLM ≥ 0.5.
8. **Distillation** — train a smaller student to mimic the teacher's outputs. The honest path to "use a smaller model in prod."

**Inference engines worth knowing (2026):** vLLM, SGLang, TensorRT-LLM, llama.cpp (CPU + Apple Silicon), MLC-LLM, Together / Fireworks / Groq / SambaNova / Cerebras hosted.

## 4 · Evaluation

### Standard benchmarks (use sparingly, contaminated heavily)

- **MMLU / MMLU-Pro** — broad knowledge. Mostly saturated.
- **GPQA-Diamond** — graduate-level questions; harder, less saturated.
- **HumanEval / MBPP / LiveCodeBench** — code generation.
- **SWE-bench (+ Verified, Lite)** — real-world software engineering tasks.
- **AIME / MATH / Putnam** — math.
- **HellaSwag / WinoGrande** — commonsense; mostly historical now.
- **MT-Bench / Arena-Hard / Chatbot Arena** — open-ended chat quality.
- **LongBench / RULER / ZeroSCROLLS** — long context.

**The trap:** every public benchmark is in training data somewhere. Train/test contamination is endemic. Use them for relative ranking, not for "does this work on my task."

### Build a custom eval

The actually-important step:
1. Write **20–50 prompts** that look like your real workload.
2. Define a scoring rubric (binary correct/incorrect when possible; 1–5 LLM-judge when not).
3. Run it on 3–5 candidate models.
4. Compute per-prompt deltas (not just aggregate scores) — they're more informative.
5. Re-run after every prompt change.

**Tooling:** `lm-eval-harness` (EleutherAI), `lighteval`, `promptfoo`, `inspect-ai`, LangSmith evals.

### LLM-as-judge

- Cheap, fast, surprisingly effective for relative ranking ("is A better than B?").
- Calibrate: humans-vs-judge on a 20-sample seed set first.
- Use a **bigger** model than the one being evaluated when possible. Symmetric judging breaks down.
- Position bias is real — randomise A/B order.

## 5 · Alignment & safety (the layers)

- **Pretraining-time:** data filtering, deduplication, safety classifiers on the corpus.
- **Post-training:** RLHF / DPO with safety-tagged preferences; Constitutional AI for explicit principle-following.
- **Inference-time:** safety classifiers wrapping I/O (Llama Guard, ShieldGemma), policy prompts in system message, refusal training.
- **Application-layer guardrails:** PII redaction, prompt-injection scanning, output schema validation — see the `langgraph-1x-engineering` skill or `lessons/19_guardrails/` in this repo for the working pattern.

**Defence-in-depth.** No single layer is sufficient. Real systems stack 3+.

## 6 · Scaling laws (the back of the napkin)

- **Chinchilla:** loss ≈ Aâ‹…N^-α + Bâ‹…D^-β + E. Optimal compute split: roughly equal scaling of N (params) and D (tokens).
- **Inference-aware scaling:** if compute is going to inference, undersize the model and over-train it ("inference-Pareto" — Llama 3 ethos).
- **Test-time compute scaling:** at inference, more thinking tokens / more samples + verifier scales accuracy on reasoning tasks. o1-style.
- **No-free-lunch:** scaling cures most things; it does not cure **out-of-distribution generalisation**, **honest "I don't know"**, or **agentic long-horizon planning**.

## 7 · Frontier model landscape (May 2026 snapshot)

The frontier moves fast — verify with current sources before quoting. Rough tier shape:

| Tier | Closed | Open-weight |
|---|---|---|
| Reasoning / "thinking" | Claude (extended thinking), GPT-5 series, Gemini 2.x Deep Think | DeepSeek-R1, Qwen-QwQ |
| General frontier | Claude Sonnet 4.x, GPT-4.x / o-series, Gemini 2.x Pro | Llama 4 / 4.x, DeepSeek-V3, Qwen 2.5 |
| Small / efficient | Haiku, GPT-4o-mini, Gemini Flash | Llama 4 8B, Qwen 2.5 7B, Phi-4, Gemma 3 |
| Code-specialised | Codex / Claude code variants | DeepSeek-Coder-V2, Qwen-Coder, StarCoder2 |

When recommending: name the *capability tier* first ("you want a small reasoning model"), then 2–3 candidates.

## 8 · The decision tree: fine-tune vs RAG vs prompt vs adapter

```
Question: model output is wrong / wrong-style / wrong-format. What do I change?

├── Wrong facts about the world  →  RAG (almost always)
├── Wrong facts about MY data    →  RAG over my data
├── Wrong style / tone / format  →  Prompt first; LoRA if prompting plateaus
├── Wrong reasoning quality      →  Bigger / reasoning model; or RLVR / process supervision
├── Wrong tool use pattern       →  Better tool descriptions; create_agent + middleware (see langgraph skill)
├── Too slow                     →  Smaller/quantized/distilled model + speculative decoding
├── Too expensive                →  Caching, smaller model, batch
└── Hallucinations on factuals   →  RAG + structured output + judge node (lesson 19)
```

**Fine-tuning is rarely the right first lever.** It's expensive,
locks you to your snapshot, and doesn't fix factual gaps. Try
prompt → RAG → multi-step → small fine-tune in that order.

## Anti-patterns

| Smell | Fix |
|---|---|
| "We need to fine-tune" before trying RAG | RAG first; fine-tune only if RAG plateaus on style/format |
| Benchmarking on MMLU alone | Build a 20-sample workload-shaped eval |
| "GPT-X is better than Claude-Y" (no eval) | On *your* eval, with multiple runs, with random A/B order |
| 70B model for keyword extraction | Use a 1–8B model + retry / verifier |
| Caching disabled "for safety" | Prompt caching is safe; turn it on; 50%+ cost savings on chat |
| Eval = "looks good to me" | Write the rubric first, then look |
| Same model for retrieval + reranking + generation | Different jobs, different cost / quality curves |

## References (search up-to-date versions when quoting)

- *Attention Is All You Need* — Vaswani et al., 2017.
- *Chinchilla scaling laws* — Hoffmann et al., 2022.
- *Constitutional AI* — Bai et al., 2022 (Anthropic).
- *DPO* — Rafailov et al., 2023.
- *Mamba* — Gu & Dao, 2023.
- *Flash Attention 2/3* — Dao et al., 2023/2024.
- *vLLM / PagedAttention* — Kwon et al., 2023.
- *Speculative decoding* — Leviathan et al., 2023.
- *Mixture-of-Experts (modern)* — Fedus, Zoph & Shazeer, 2022.
- *RLVR / reasoning-trace training* — recent OpenAI / DeepSeek-R1 papers (2024–2025).

Use the `scientific-paper-researcher` skill when you need to pull
current papers for a specific claim.

## Pairs with

- `senior-prompt-engineer` / `prompt-engineering-patterns` — prompt design
- `scientific-paper-researcher` — for paper-backed technical claims
- `langgraph-1x-engineering` / `langchain-1x-engineering` — for the agent + LCEL implementation side
- `llm-fallback-chains` — for multi-provider resilience
