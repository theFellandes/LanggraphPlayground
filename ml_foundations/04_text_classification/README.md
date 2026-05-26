# ml_foundations · 04 · Text classification with a fine-tuned encoder

> When the LLM is overkill.

Sentiment, intent, topic, language detection, spam filtering — these
are **closed-set** classification tasks. A 65M-parameter encoder like
DistilBERT fine-tuned for 10 minutes on a Mac will outclassify (and
out-pace, and out-cost) a 70B-parameter chat LLM. By a lot.

This lesson fine-tunes `distilbert-base-uncased` on a small intent
dataset and gives you the numbers to make this call in your own work.

## What you'll build

A 3-class intent classifier — `refund`, `policy_question`,
`escalation` — trained on ~120 synthetic examples generated in the
script. Train, eval, save, then run inference inline.

## When to reach for this vs an LLM

| Question | Encoder | LLM |
|---|---|---|
| Latency per request | 10ms CPU, 1ms GPU | 200-1000ms |
| Cost per 1M classifications | ~$0 (it runs on your laptop) | $100-$5000 depending on model |
| Output | Closed set | Open-ended (you need parsing) |
| Few-shot updates | Re-train (~10 min) | Just edit the prompt |
| Hard reasoning, multi-step | Bad | Good |
| Multilingual transfer | Need explicit training | Often works zero-shot |

**Rule of thumb:** if your output is one of <50 fixed labels and you
have 100+ labeled examples per class, **fine-tune an encoder**.
Otherwise prompt the LLM.

## The pretrain → fine-tune trick

Step 1 (someone else's bill): pretrain `distilbert-base-uncased` on
Wikipedia + BooksCorpus, predicting masked words. Learns English
grammar + facts. **65M parameters.**

Step 2 (yours): freeze most weights, swap a 3-class head on top,
train for 3 epochs on 120 examples. Takes ~10 minutes on a laptop CPU.

The model now classifies your intent with ~95% accuracy. The
heavy-lifting (language understanding) was already done; you just
taught it your label set.

## Run it

```bash
uv sync --extra ml
uv run python -m ml_foundations.04_text_classification.train
```

The script:

1. Generates a synthetic dataset (intentional — keeps the lesson
   self-contained; replace with your real data when you adopt the
   pattern).
2. Trains DistilBERT for 3 epochs with a typical setup (AdamW, linear
   warmup, eval after each epoch).
3. Saves to `data/models/intent_classifier/`.
4. Runs inference on five held-out examples and prints predictions +
   confidences.

## Try it yourself

1. Replace the synthetic data with your real conversations. Even 50 labelled examples per class teaches well.
2. Swap `distilbert-base-uncased` for `xlm-roberta-base` for multilingual coverage. Slightly slower, much better cross-language transfer.
3. Wrap the trained model in a LangChain `@tool` so your **agent** can call it for fast intent detection before deciding to route.
4. Compare your encoder's accuracy to GPT-4o on the same 30 examples. Plot accuracy vs cost-per-1000-classifications.

## Pairs with

- [Lesson 04 · Structured output](../../lessons/04_structured_output/README.md) — the "LLM for classification" version this lesson is the alternative to
- [Lesson 25 · Tool design](../../lessons/25_tool_design/README.md) — wrap the trained encoder as a `@tool`

## References

- [DistilBERT paper · Sanh et al. 2019](https://arxiv.org/abs/1910.01108)
- [Hugging Face NLP course · ch 3](https://huggingface.co/learn/nlp-course/chapter3/1) — the canonical tutorial
- [`transformers` `Trainer` API docs](https://huggingface.co/docs/transformers/main_classes/trainer)
