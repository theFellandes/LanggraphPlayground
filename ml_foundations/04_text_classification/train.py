"""Fine-tune DistilBERT on a synthetic 3-class intent dataset.

Run:
    uv sync --extra ml
    uv run python -m ml_foundations.04_text_classification.train
"""

from __future__ import annotations

import random
from pathlib import Path

from shared import settings
from shared.pretty import console, section

MODEL_DIR = settings.data_dir / "models" / "intent_classifier"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


LABELS = ["refund", "policy_question", "escalation"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}


def _make_dataset(seed: int = 42, n_per_class: int = 40):
    random.seed(seed)
    refund_templates = [
        "I want a refund on order #{ord}.",
        "Please refund my payment for {prod}.",
        "Can I get my money back for order {ord}?",
        "{prod} is broken, I want my money back.",
        "Refund please — order {ord} arrived late.",
    ]
    policy_templates = [
        "How many PTO days do I get?",
        "What's your remote-work policy?",
        "Is travel expensed?",
        "Can I bring my dog to the office?",
        "What's the policy on parental leave?",
    ]
    escalation_templates = [
        "This is unacceptable, I want to speak to a manager.",
        "Your bot can't help me. Connect me to a human.",
        "I've been waiting 3 days. Escalate this NOW.",
        "Please transfer me to a real agent.",
        "I need a supervisor on this.",
    ]

    def fill(t):
        return t.replace("{ord}", f"#{random.randint(100, 9999)}").replace(
            "{prod}", random.choice(["headphones", "laptop", "monitor", "the desk", "my chair"])
        )

    examples = []
    for templates, label in [
        (refund_templates, "refund"),
        (policy_templates, "policy_question"),
        (escalation_templates, "escalation"),
    ]:
        for _ in range(n_per_class):
            examples.append({"text": fill(random.choice(templates)), "label": LABEL_TO_ID[label]})
    random.shuffle(examples)
    return examples


def train_and_save() -> Path:
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        console.print(f"[yellow]Missing ML deps ({e.name}). Run: uv sync --extra ml[/]")
        raise

    section("Building dataset")
    train_rows = _make_dataset(seed=42, n_per_class=40)
    eval_rows = _make_dataset(seed=99, n_per_class=10)
    console.print(f"train={len(train_rows)}  eval={len(eval_rows)}")

    model_name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(model_name)

    def encode(ex):
        return tok(ex["text"], truncation=True, max_length=64)

    train_ds = Dataset.from_list(train_rows).map(encode, batched=True)
    eval_ds = Dataset.from_list(eval_rows).map(encode, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS),
        id2label=ID_TO_LABEL, label2id=LABEL_TO_ID,
    )

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "_runs"),
        num_train_epochs=3,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        logging_steps=10,
    )

    def metrics(pred):
        import numpy as np
        preds = pred.predictions.argmax(axis=-1)
        acc = (preds == pred.label_ids).mean()
        return {"accuracy": float(acc)}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics,
    )

    section("Training")
    trainer.train()

    section("Final eval")
    final = trainer.evaluate()
    console.print(f"eval_accuracy={final.get('eval_accuracy'):.3f}")

    section("Saving")
    trainer.save_model(str(MODEL_DIR))
    tok.save_pretrained(str(MODEL_DIR))
    console.print(f"Saved → {MODEL_DIR}")
    return MODEL_DIR


def demo_inference(path: Path) -> None:
    from transformers import pipeline

    section("Inference on 5 held-out prompts")
    clf = pipeline("text-classification", model=str(path), tokenizer=str(path))
    examples = [
        "Refund my order #1234, the product was defective",
        "How many days off do I get per year?",
        "I want to talk to a real person, your bot is useless",
        "Where can I find the company holiday calendar?",
        "Order #999 arrived broken, money back please",
    ]
    for s in examples:
        out = clf(s)[0]
        console.print(f"  {s[:55]:55}  → [bold]{out['label']}[/]  ({out['score']:.2f})")


def main() -> None:
    path = train_and_save()
    demo_inference(path)


if __name__ == "__main__":
    main()
