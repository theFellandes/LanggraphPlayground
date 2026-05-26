"""ml_foundations · 06 · Tiny LoRA-SFT demo on SmolLM2-135M.

This is a *runnable* example of the SFT workflow — small enough to
train on a laptop CPU/GPU in a few minutes. Don't expect a magic
quality lift from 30 examples; the goal is to see the shape of a real
fine-tuning loop.

Run:
    uv sync --extra ml
    uv run python -m ml_foundations.06_finetuning_llms.train_sft
    uv run python -m ml_foundations.06_finetuning_llms.train_sft --inference
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from shared import settings
from shared.pretty import console, section

OUT_DIR = settings.data_dir / "models" / "finetune_demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _build_dataset(seed: int = 42, n: int = 60):
    """A tiny instruction dataset for a fake 'Acme' support persona."""
    random.seed(seed)
    refund_templates = [
        ("I want a refund on order #{ord} for ${amt}.",
         "Refunds over $100 require human approval. I've escalated your request — an Acme agent will follow up within 1 business day."),
        ("Refund my ${amt} order, it's broken.",
         "I'm sorry to hear that. For refunds over $100 we need a human reviewer — escalating now."),
        ("Refund ${amt} please.",
         "{cond}"),
    ]
    policy_templates = [
        ("How many PTO days do I get?",
         "Full-time employees at Acme receive 20 PTO days per year, accruing monthly from the start date."),
        ("Can I work remote?",
         "Acme allows remote work up to 3 days per week with manager approval."),
        ("What's the parental-leave policy?",
         "New parents at Acme are entitled to 16 weeks of paid leave following birth or adoption."),
        ("Holiday policy?",
         "Acme observes 10 paid public holidays per year plus your 20 PTO days."),
    ]

    examples = []

    for _ in range(n // 2):
        t, resp = random.choice(refund_templates)
        amt = random.choice([42, 75, 99, 120, 250, 600])
        cond = (
            "I've processed your refund — it should arrive within 7 business days."
            if amt < 100
            else "Refunds over $100 require human approval. Escalating now."
        )
        prompt = t.format(ord=random.randint(100, 9999), amt=amt)
        completion = resp.format(cond=cond)
        examples.append({"prompt": prompt, "completion": completion})

    for _ in range(n // 2):
        prompt, completion = random.choice(policy_templates)
        examples.append({"prompt": prompt, "completion": completion})

    random.shuffle(examples)
    return examples


def _build_model_and_tokenizer():
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        console.print(f"[yellow]Missing ML deps ({e.name}). Run: uv sync --extra ml[/]")
        raise

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,           # CPU-friendly default
    )

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tok


def _format_example(tok, ex):
    """Chat-template the example using the tokenizer's own template."""
    msgs = [
        {"role": "user", "content": ex["prompt"]},
        {"role": "assistant", "content": ex["completion"]},
    ]
    return {"text": tok.apply_chat_template(msgs, tokenize=False)}


def train() -> None:
    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        console.print(f"[yellow]Missing ML deps ({e.name}). Run: uv sync --extra ml[/]")
        return

    section("Build dataset")
    examples = _build_dataset(n=60)
    console.print(f"{len(examples)} examples")
    for ex in examples[:3]:
        console.print(f"  PROMPT: {ex['prompt'][:60]}")
        console.print(f"  COMPL : {ex['completion'][:60]}\n")

    section("Build model + LoRA adapters")
    model, tok = _build_model_and_tokenizer()
    ds = Dataset.from_list(examples).map(lambda ex: _format_example(tok, ex))

    section("Train (3 epochs, LR=2e-4)")
    config = SFTConfig(
        output_dir=str(OUT_DIR / "_runs"),
        num_train_epochs=3,
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_seq_length=512,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=ds)
    trainer.train()

    section("Save adapter")
    trainer.save_model(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))
    console.print(f"Saved → {OUT_DIR}")


def inference() -> None:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        console.print(f"[yellow]Missing ML deps ({e.name}). Run: uv sync --extra ml[/]")
        return

    section("Load base vs fine-tuned, compare on a held-out prompt")

    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    if not (OUT_DIR / "adapter_config.json").exists():
        console.print(f"[yellow]No adapter found at {OUT_DIR}. Run training first.[/]")
        return

    tuned = PeftModel.from_pretrained(base, str(OUT_DIR))

    held_out = [
        "I'd like a refund for $250.",
        "How many vacation days does a new hire get?",
        "Can I work from home full-time?",
    ]

    for prompt in held_out:
        msgs = [{"role": "user", "content": prompt}]
        inp = base_tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        with torch.no_grad():
            base_out = base.generate(inp, max_new_tokens=80, do_sample=False)
            tuned_out = tuned.generate(inp, max_new_tokens=80, do_sample=False)
        base_text = base_tok.decode(base_out[0][inp.shape[1]:], skip_special_tokens=True)
        tuned_text = base_tok.decode(tuned_out[0][inp.shape[1]:], skip_special_tokens=True)
        console.rule(f"[bold]Q:[/] {prompt}")
        console.print(f"  [bold cyan]base :[/] {base_text.strip()[:200]}")
        console.print(f"  [bold green]tuned:[/] {tuned_text.strip()[:200]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action="store_true", help="skip training; run inference on the saved adapter")
    args = parser.parse_args()
    if args.inference:
        inference()
    else:
        train()
        inference()


if __name__ == "__main__":
    main()
