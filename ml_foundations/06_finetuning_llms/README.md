# ml_foundations · 06 · Fine-tuning LLMs (LoRA + DPO)

You've seen encoder fine-tuning (lesson 04 — DistilBERT classification;
lesson 05 — contrastive sentence-transformer). Those models are
modest in size (~100M params). This lesson covers **fine-tuning
decoder LLMs** — Llama, Mistral, Qwen — where the model has 7B-70B
parameters and "fine-tune the whole thing" stops being feasible.

The breakthrough that made this practical: **LoRA**. Then **DPO**.
Together they let you fine-tune a 7B model on a single consumer GPU
in a few hours.

## What you'll learn

1. **When to fine-tune** — vs prompt vs RAG vs routing
2. **LoRA / QLoRA** — the math of low-rank adapters; why 0.1% of params is enough
3. **Hugging Face `trl`** — the production fine-tuning library
4. **Instruction fine-tuning (SFT)** — supervised teacher-forcing
5. **DPO, KTO, ORPO** — preference tuning, the modern alternative to RLHF
6. **Synthetic data generation** — the actual hard part
7. **Hosted vs self-hosted** — OpenAI fine-tuning API vs Hugging Face SFT-Trainer
8. **Eval before / after** — same workflow as lesson 35

## Part 1 · When to fine-tune

The decision tree (use this verbatim in an interview):

```
Is your task expressible as a prompt + examples?
├── yes → prompt + few-shot (cheapest, fastest iteration)
│         └── insufficient quality → continue
└── no → fine-tune

Does the model need to know domain-specific FACTS?
├── yes → RAG (lessons 06-07, 29, 33)
└── no  → continue

Does the model need to follow a domain-specific STYLE / FORMAT?
├── yes → fine-tune SFT (this lesson)
└── no  → continue

Do you have human preference data (chose A over B)?
├── yes → fine-tune DPO (this lesson)
└── no  → consider collecting it

Is the gap routable (some queries need stronger model)?
├── yes → routing (lesson 38)
└── no  → fine-tune
```

**Reasonable rule:** RAG for facts, fine-tune for behaviour. The
canonical mistake is to fine-tune for facts that change weekly — your
fine-tuned model is stale on day 8.

## Part 2 · LoRA — the math

The base model has weight matrices `W ∈ ℝ^{d × d}` (millions to
billions of params total). Fine-tuning naively means updating all of
them. LoRA's trick:

```
W' = W + ΔW
ΔW = A · B,  where  A ∈ ℝ^{d × r},  B ∈ ℝ^{r × d},  r ≪ d
```

You **freeze W** entirely. You only train `A` and `B`. With `r = 8`
and `d = 4096`, you've gone from `d²` = 16.7M parameters per matrix
to `2·d·r` = 65k — a **250× reduction**.

At inference: compute `Wx + ABx` instead of `Wx`. The `AB` term is
the learned update; you can merge it back into `W` for deployment.

```
            ┌───────────┐
            │     W     │  frozen, large (e.g. 4096 × 4096)
   x ──→    │  d × d    │  ──→  Wx
            └───────────┘            ↓
                                     +     ──→  output
                                     ↑
            ┌───┐   ┌───────────┐    │
            │ A │ · │     B     │  trainable, small
            └───┘   └───────────┘    (e.g. 8 × 4096 and 4096 × 8)
              ↑           ↑
              x           x
```

**Why this works**: the *intrinsic dimensionality* of most fine-tuning
tasks is low. You're not teaching the model new language — you're
biasing its already-broad knowledge toward your task. A rank-8 update
is plenty for that.

### QLoRA — LoRA on a 4-bit-quantised base model

The base model is **frozen anyway**. So why store it in fp32?
Quantise it to 4-bit. The adapters stay in fp16/bf16. This is what
lets you fine-tune Llama-70B on a single 80GB GPU instead of needing
8.

```python
# pip install peft bitsandbytes
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4",
)
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                            quantization_config=bnb)
lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, lora)
print(model.print_trainable_parameters())
# trainable params: 4,194,304 || all params: 8,034,529,280 || trainable%: 0.0522
```

You're training **0.05%** of the parameters. Still beats prompt-engineering on the right task.

## Part 3 · `trl` — the fine-tuning library

Hugging Face's [`trl`](https://github.com/huggingface/trl) is the
standard. Three trainers you should know:

| Trainer | What it does | Data format |
|---|---|---|
| **SFTTrainer** | Supervised fine-tuning (next-token prediction) | `{"prompt": str, "completion": str}` or just `{"text": str}` |
| **DPOTrainer** | Direct Preference Optimization | `{"prompt": str, "chosen": str, "rejected": str}` |
| **KTOTrainer** | Kahneman-Tversky Optimization (one-sided preferences) | `{"prompt": str, "completion": str, "label": bool}` |
| **GRPOTrainer** | Group Relative Policy Optimization (reasoning-model training) | `{"prompt": str}` + reward function |

### SFT — supervised fine-tuning

```python
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

data = Dataset.from_list([
    {"messages": [
        {"role": "user", "content": "Refund $250 order"},
        {"role": "assistant", "content": "Refunds over $100 require approval. I've escalated."},
    ]},
    # ... more examples ...
])

config = SFTConfig(
    output_dir="./out",
    num_train_epochs=3,
    learning_rate=2e-4,           # LoRA prefers higher LR than full FT
    per_device_train_batch_size=2,
    max_seq_length=2048,
)

trainer = SFTTrainer(model=model, args=config, train_dataset=data)
trainer.train()
trainer.save_model("./my-finetune")
```

That's the entire SFT loop. The `model` from Part 2 (LoRA-wrapped) is
what you train.

### DPO — preference tuning (the RLHF replacement)

The shape of the data: triples of `(prompt, chosen, rejected)`.

```python
from trl import DPOConfig, DPOTrainer

data = Dataset.from_list([
    {
        "prompt": "How many PTO days do I get?",
        "chosen": "Full-time employees get 20 PTO days per year.",
        "rejected": "I'm not sure, you should ask HR.",
    },
    # ... 500-5000 such triples ...
])

config = DPOConfig(
    output_dir="./out_dpo",
    beta=0.1,                     # KL strength — how close to the reference model
    learning_rate=5e-6,
    num_train_epochs=1,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,               # uses base if None
    args=config,
    train_dataset=data,
    tokenizer=tok,
)
trainer.train()
```

What DPO does: increases the log-prob ratio of `chosen` over
`rejected` while penalising drift from the base model (KL divergence,
weighted by `beta`). The result behaves like an RLHF-trained model
without needing a reward model or RL machinery.

**KTO** generalises further — you only need `(prompt, completion,
liked_bool)`, not paired comparisons. Useful when you have thumbs-up /
thumbs-down feedback rather than A/B.

**ORPO** is one-step (no separate SFT phase). Train SFT + preference
in a single objective. Newest, getting traction.

## Part 4 · The synthetic data problem (the actual hard part)

Where do you get 5,000 preference triples?

Three patterns, in order of cost / quality:

### Pattern A — generate with a stronger model

```python
GENERATE_PAIRS = """Given this prompt, produce TWO answers:
- "chosen": follows our policy (Acme support; refunds > $100 escalate; concise)
- "rejected": violates the policy in some realistic way

Prompt: {prompt}

Reply STRICTLY as JSON: {"chosen": "...", "rejected": "..."}"""

# Run on 5000 diverse prompts with a frontier model (gpt-4o / sonnet-4-6).
# Cost: ~$50-200 for 5000 triples. The cheapest dataset you'll ever build.
```

This is the **default** in 2025-2026. Quality is high if the
stronger model knows the policy.

### Pattern B — production logs + human labels

```
1. Sample 1000 real production prompts
2. For each, get 2 candidate answers (or "A vs B from two prompt versions")
3. Show to humans, collect preferences
4. Train DPO
```

Cost: $5-20 per labelled triple × 1000 = $5k-20k. Slow. The highest
quality.

### Pattern C — rejection sampling against a verifier

For tasks with a *programmatic* verifier (math problems, code
correctness, schema compliance):

```
For each prompt:
  Sample 10 completions from the model
  Use the verifier to pick the best one (chosen) and a wrong one (rejected)
```

Cost: only the inference. Highest quality on tasks where you can
build a verifier. This is how reasoning models (R1, o1) were trained
at scale.

## Part 5 · Hosted alternative — OpenAI fine-tuning API

If you don't want to manage compute:

```python
from openai import OpenAI
client = OpenAI()

# Upload your JSONL training file
training = client.files.create(file=open("train.jsonl", "rb"), purpose="fine-tune")

# Kick off the job
job = client.fine_tuning.jobs.create(
    training_file=training.id,
    model="gpt-4o-mini-2024-07-18",      # only certain models are tunable
)

# When done, you get a model id like "ft:gpt-4o-mini:org:custom:abc123"
client.fine_tuning.jobs.retrieve(job.id)
```

Trade-offs vs DIY:

| | Hosted (OpenAI / Anthropic) | DIY (Llama + LoRA) |
|---|---|---|
| Time-to-first-tuned-model | 30 min | A day, end-to-end |
| Cost / 1000 examples | $1-10 (training) + per-token inference premium | GPU rental + your time |
| Model ownership | Lives in vendor account | You own it; can self-host |
| Frontier models | Sometimes (gpt-4o is tunable) | Open-weights only |
| Custom hyperparameters | Limited knobs | Everything |

For most "make the model follow our format" tasks, **OpenAI's hosted
fine-tuning is the right first attempt**. It's cheap and the iteration
loop is fast. Reach for LoRA when you want to own the weights or use
an open-weights model.

## Part 6 · Eval before + after

The same workflow as lesson 35. Build the eval set first. Run the
*base* model through it. Fine-tune. Re-run. Compare.

```python
def score(model_path: str, eval_cases: list) -> float:
    model = load(model_path)
    passes = 0
    for case in eval_cases:
        out = model.generate(case["prompt"])
        if case["scorer"](out, case["expected"]):
            passes += 1
    return passes / len(eval_cases)

before = score("meta-llama/Llama-3.1-8B-Instruct", eval_set)
after = score("./my-finetune", eval_set)
print(f"recall@1 before={before:.2f}  after={after:.2f}")
```

A real fine-tune typically moves an instruction-following score by 5-15
points. If yours moves by less than 2 points, *something is wrong* —
data quality, learning rate, or you've over-trained on the synthetic set.

## Part 7 · Common pitfalls

1. **Catastrophic forgetting.** A heavily-fine-tuned model gets dumber on tasks not in the training set. Mitigate: keep epoch count low; include general-purpose data in the mix.

2. **Synthetic-data echo chamber.** If your synthetic data is generated by GPT-4, you're fine-tuning the small model to mimic GPT-4. That's fine for behaviour but can amplify GPT-4's biases.

3. **Eval-set contamination.** If your eval cases were used (even partially) in training, scores are meaningless. **Hold them out from synthetic generation too.**

4. **Tokenizer mismatch.** If you fine-tune Llama-3 and serve via a stack that uses a slightly different tokenizer, all bets are off.

5. **Forgetting the chat template.** Modern instruct models have a specific chat-template (special tokens like `<|im_start|>`). Your training data must respect it; `trl` mostly handles this but it's a debugging starting point.

## Run it

```bash
uv sync --extra ml
uv run python -m ml_foundations.06_finetuning_llms.train_sft       # tiny SFT demo
uv run python -m ml_foundations.06_finetuning_llms.train_sft --inference   # try the result
```

The demo uses an extremely small base model (`HuggingFaceTB/SmolLM2-135M`)
so it actually trains on a laptop CPU/GPU in minutes. The dataset is
synthetic ~30 (prompt, completion) pairs about an imaginary "Acme"
company. After training:

- Save adapter to `data/models/finetune_demo/`
- Re-run with `--inference` to compare base vs fine-tuned output on a held-out prompt

Don't expect miracles from 30 examples and a 135M model — the goal is
to see the *workflow* with realistic library calls. Scale up to
Llama-3-8B + 5k examples when you do this for real.

## Try it yourself

1. **Bump base model.** Swap `SmolLM2-135M` for `Qwen2.5-0.5B`. Same shape, slightly better outputs. (Up to ~1B model is feasible on a laptop.)
2. **Generate the dataset with a stronger model.** Use `get_llm()` from this repo with a frontier model to produce 100 (prompt, chosen, rejected) triples; switch from SFT to DPO.
3. **Eval before/after.** Build a 10-case eval set (lesson 35 patterns); measure pass-rate delta.
4. **Hosted alternative.** Convert the same 30 examples to OpenAI fine-tune JSONL format. Compare results.

## Anti-patterns

| Smell | Fix |
|---|---|
| Fine-tuning to inject facts | Use RAG. Facts change; weights don't update on a Wednesday |
| Fine-tuning before trying prompt engineering | The prompt is 50 lines of YAML; fine-tuning is half a day |
| Skipping the eval suite | You can't tell if it improved without one. Always |
| Training for 10 epochs because "more is better" | 1-3 epochs for SFT, often <1 for DPO. Overfitting is fast |
| Using the same data for synthesis and eval | Contamination — scores are meaningless |
| Tuning learning rate by feel | 1e-4 to 5e-4 for LoRA; 1e-6 to 1e-5 for full FT. Don't go wild |
| Full FT on a 70B model with one GPU | OOM. Use QLoRA + 4-bit. Or rent multi-GPU |
| No chat template in training data | Model produces garbage at inference. Match the model's expected format |

## Pairs with

- **[ml_foundations/04 · Text classification](../04_text_classification/README.md)** — encoder fine-tune (smaller scale, simpler)
- **[ml_foundations/05 · Fine-tuning encoders](../05_finetuning_encoders/README.md)** — contrastive (different loss)
- **[Lesson 35 · Evaluation](../../lessons/35_evaluation_discipline/README.md)** — measure before/after
- **[Lesson 38 · Reasoning + routing](../../lessons/38_reasoning_and_routing/README.md)** — the alternative to fine-tuning
- **[`skills/llm-expert`](../../skills/llm-expert/SKILL.md)** — covers the fine-tune-vs-prompt-vs-RAG decision

## References

- [LoRA paper · Hu et al. 2021](https://arxiv.org/abs/2106.09685) — the original
- [QLoRA paper · Dettmers et al. 2023](https://arxiv.org/abs/2305.14314) — 4-bit quantisation
- [DPO paper · Rafailov et al. 2023](https://arxiv.org/abs/2305.18290) — the RLHF replacement
- [KTO paper · Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306) — one-sided preferences
- [ORPO paper · Hong et al. 2024](https://arxiv.org/abs/2403.07691) — single-step SFT + preference
- [Hugging Face TRL docs](https://huggingface.co/docs/trl) — the library
- [Hugging Face PEFT docs](https://huggingface.co/docs/peft) — LoRA / adapter library
- [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) — YAML-driven fine-tuning recipes
- [unsloth](https://github.com/unslothai/unsloth) — 2× faster LoRA training
- [OpenAI fine-tuning docs](https://platform.openai.com/docs/guides/fine-tuning)
- [Anthropic fine-tuning](https://docs.anthropic.com/en/docs/build-with-claude/finetune) — Haiku-tier fine-tunes (limited availability)

## Next →

You're at the end of `ml_foundations/`. Pick a [capstone](../../projects/),
revisit [Lesson 38 · routing](../../lessons/38_reasoning_and_routing/README.md)
with a fine-tuned model as one of the routed tiers, or head to the
[skills/llm-expert](../../skills/llm-expert/SKILL.md) skill for the
broader fine-tune-vs-prompt-vs-RAG decision framework.
