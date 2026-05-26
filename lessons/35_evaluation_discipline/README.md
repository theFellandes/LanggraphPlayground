# Lesson 35 · Evaluation as a discipline

Lesson 26 Topic 4 ships a 40-line eval framework. That's the
**primer**. This is the **practice** — the full eval-driven
development workflow used by teams that ship LLM products
sustainably. Tools, anti-patterns, statistical-significance pitfalls,
CI integration, synthetic-data generation, the whole loop.

> **The unfair advantage:** teams with a real eval discipline iterate
> ~10× faster than teams without one. Not because they write better
> prompts but because they can *tell* a regression in 5 minutes
> instead of 5 days.

## What you'll learn

1. **The eval mental model** — why "vibes" stops working past lesson 5
2. **The five scorer flavours** — when each is right
3. **The frameworks** — promptfoo, RAGAS, deepeval, Inspect, LangSmith
4. **LLM-as-judge done properly** — rubrics, position bias, judge model selection
5. **CI gating** — block merges on regression; the workflow that closes the loop
6. **Synthetic test data** — generate eval cases with a stronger model
7. **Statistical significance** — when "v3 wins by 2%" actually means something

## Part 1 · The eval mental model

A prompt change "feels better" is **never a signal** — that's
confirmation bias. A 5% regression on a 30-case eval suite *is* a
signal.

```
eval set         = (input, expected, scorer)  ×  N cases
system under test = the thing you're iterating on
eval run         = run SUT on all N cases, score each, aggregate
```

The discipline:

1. **Build the eval set BEFORE iterating on the system.** Otherwise
   you're optimising against vibes.
2. **30-50 cases per task.** Fewer = noisy. More = slow + diminishing
   returns. Add cases when you encounter a new bug, not preemptively.
3. **Pin the scorer.** Especially if it's LLM-as-judge — judge model
   drifts.
4. **Run on every change.** Not "once a sprint." Every PR.
5. **Track scores over time.** A flat dashboard hides regressions.
6. **Include "should refuse" cases.** Otherwise you only test happy paths.

## Part 2 · The five scorer flavours

| Scorer | Output type | When |
|---|---|---|
| **Exact match** | Deterministic strings/dates/codes | Extraction tasks, classifications |
| **Contains match** | Substrings | "Did the answer mention X?" — looser than exact |
| **Schema match** | JSON / Pydantic | Structured output: required keys present, types right |
| **Embedding cosine** | Free text | Generic "is this semantically close to the expected" — coarse |
| **LLM-as-judge** | Free text + rubric | Open-ended: helpfulness, factuality, harmfulness |

The trap most teams fall into: **using LLM-as-judge for everything.**
It's slow, expensive, and noisier than the deterministic scorers.
Use deterministic ones where you can; reach for judges only when the
output is genuinely free-text.

## Part 3 · The frameworks

### 3.1 · promptfoo — declarative YAML, CLI

```yaml
# promptfooconfig.yaml
description: "Support bot prompt eval"
providers:
  - anthropic:messages:claude-sonnet-4-6
  - openai:chat:gpt-4o-mini
prompts:
  - file://prompts/v2.txt
  - file://prompts/v3.txt
tests:
  - vars:
      question: "How many PTO days?"
    assert:
      - type: contains
        value: "20"
      - type: llm-rubric
        value: "The answer cites the company handbook"
  - vars:
      question: "Refund $250 to a card?"
    assert:
      - type: contains
        value: "approval"
```

Run:

```bash
promptfoo eval                     # writes a dashboard
promptfoo view                     # opens HTML report at localhost:15500
promptfoo eval --no-progress -o out.json   # CI-friendly
```

**Best for**: matrix testing (M models × N prompts). The single
config file is the documentation of "what did we test."

### 3.2 · RAGAS — RAG-specific metrics

For RAG pipelines specifically:

```python
# pip install ragas datasets
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

eval_set = Dataset.from_dict({
    "question": ["How many PTO days?", "Refund $250?"],
    "answer":   ["20 days", "Needs human approval"],
    "contexts": [["Full-time employees get 20 PTO days..."],
                 ["Refunds > $100 require approval..."]],
    "ground_truth": ["20", "approval required"],
})
report = evaluate(eval_set, metrics=[faithfulness, answer_relevancy,
                                      context_precision, context_recall])
print(report)
```

The four RAGAS metrics:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Does the answer follow from the retrieved context? (catches hallucination) |
| **Answer relevancy** | Does the answer actually address the question? |
| **Context precision** | Are the retrieved chunks ranked with the relevant ones first? |
| **Context recall** | Did we retrieve enough? (vs the ground-truth answer) |

**Best for**: anything with retrieval. The single biggest investment
RAG teams under-make. Lesson 06's retriever decisions become testable.

### 3.3 · deepeval — pytest-native

```python
# pip install deepeval
import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def test_support_bot_pto():
    case = LLMTestCase(
        input="How many PTO days?",
        actual_output=run_my_agent("How many PTO days?"),
        retrieval_context=["Full-time get 20 PTO days/year."],
    )
    assert_test(case, [HallucinationMetric(threshold=0.5),
                       AnswerRelevancyMetric(threshold=0.7)])
```

Just `pytest tests/test_eval.py`. Plays nicely with your existing
test infrastructure.

**Best for**: teams already on pytest who don't want a separate eval
runner.

### 3.4 · Inspect — UK AI Safety Institute framework

```python
# pip install inspect-ai
from inspect_ai import Task, task, eval
from inspect_ai.dataset import csv_dataset
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

@task
def support_bot():
    return Task(
        dataset=csv_dataset("evals/support.csv"),
        solver=generate(),
        scorer=includes(),
    )

# CLI: inspect eval lessons/35_evaluation_discipline/support_bot.py
```

**Best for**: safety / agent / capability evaluations. Strong on
agentic test scenarios. Picked by AISI for their model evals — a
strong signal of rigor.

### 3.5 · LangSmith evals — first-party (covered in lesson 34)

Already in lesson 34. The integration story is best if you're already
on LangSmith for tracing; same datasets serve both.

### Cross-comparison

| Need | Pick |
|---|---|
| YAML/CLI, model × prompt matrix | promptfoo |
| RAG-specific quality | RAGAS |
| pytest-native, lives with your tests | deepeval |
| Agent / safety / capability evals | Inspect |
| Already on LangSmith | LangSmith evals |
| Quick custom scorer logic | Roll your own (lesson 26 Topic 4) |

## Part 4 · LLM-as-judge done properly

The dangerous default: "ask GPT-4 if the answer is good."

```python
# WRONG — vague rubric, position-biased, no calibration
judge_prompt = f"Is this a good answer?\nQ: {question}\nA: {answer}\nReply YES or NO."
```

Problems:

1. **No rubric** — "good" means different things in different contexts
2. **Position bias** — LLMs systematically prefer the first option when comparing
3. **No calibration** — what does "70% pass rate" mean if the judge is unreliable?
4. **Judge model drifts** — same prompt scores differently across model versions

### The properly-built judge

```python
JUDGE_PROMPT = """You are evaluating a customer-support bot's answer.

Score 1-5 on each dimension:
- correctness (1=wrong, 5=verifiably accurate)
- groundedness (1=invented, 5=clearly cites the context)
- helpfulness (1=evasive, 5=resolves the user's issue)

Reply STRICTLY as JSON:
{"correctness": int, "groundedness": int, "helpfulness": int, "reasoning": str}

Question: {{ question }}
Context provided to the bot: {{ context }}
Bot's answer: {{ answer }}
"""

def judge(question, context, answer) -> dict:
    out = get_llm("openai", model="gpt-4o").invoke(
        JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    )
    return json.loads(out.content)
```

Five rules:

1. **Use a rubric with named dimensions** — "correctness", "groundedness", "helpfulness" — each scored independently
2. **Use 1-5 Likert, not pass/fail** — captures nuance and helps detect judge calibration drift
3. **Make the judge return JSON** — programmatic aggregation
4. **For pairwise comparison, randomise position** — flip a coin per case for which answer is "A" vs "B"
5. **Pin the judge model** — `gpt-4o-2024-08-06`, not `gpt-4o` (which is a moving alias)
6. **Cross-validate against humans** — periodically label 30 cases by hand, measure judge agreement

### Position bias mitigation

```python
def pairwise_judge(question, answer_a, answer_b, n_trials=4):
    """Average over multiple position permutations."""
    wins_a = 0
    for _ in range(n_trials):
        if random.random() < 0.5:
            a, b, swap = answer_a, answer_b, False
        else:
            a, b, swap = answer_b, answer_a, True
        verdict = judge_pair(question, a, b)              # returns "A" or "B"
        if (verdict == "A" and not swap) or (verdict == "B" and swap):
            wins_a += 1
    return wins_a / n_trials
```

## Part 5 · CI gating

The workflow that closes the loop:

```yaml
# .github/workflows/eval.yml
name: prompt-eval
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra dev
      - name: Run evals
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: uv run python -m lessons.35_evaluation_discipline.run_eval --gate
      - name: Compare to baseline
        run: |
          uv run python scripts/compare_evals.py \
            --current out.json \
            --baseline main_branch_baseline.json \
            --max-regression 0.05    # block PR if pass-rate drops >5%
```

Key pieces:

- **Baseline file** — committed to main, regenerated nightly. Each PR diffs against it.
- **Regression threshold** — 5% on pass-rate is the common bar. Tune for your noise level.
- **Required check** — make the eval CI job *required* in branch protection. The discipline is in the enforcement.

### When the eval blocks a PR

The right response is **not** "lower the threshold." Two paths:

1. **The change is intended** to alter behaviour on the failing cases (e.g., you fixed a bug; the eval still expects the old answer). **Update the expected outputs.** Be explicit in the PR.
2. **The change has a real regression.** Fix it before merging.

Lowering the threshold to merge is how teams ship a slow quality decay.

## Part 6 · Synthetic test data

Hand-curating 50 cases is fine; hand-curating 5000 isn't. The
technique: **generate eval cases with a stronger model**.

```python
GENERATOR_PROMPT = """Generate 20 realistic customer-support questions
that an Acme Inc employee might ask. Cover:
- 5 about PTO/holidays
- 5 about expenses/reimbursements
- 5 about IT/security policies
- 5 edge cases (sensitive, ambiguous, out-of-scope)

For each, also generate the expected answer based on this handbook:
<handbook>{handbook_text}</handbook>

Reply STRICTLY as JSON array:
[{"question": "...", "expected_answer": "...", "category": "..."}]
"""
```

Run once with a frontier model (gpt-4, sonnet-4); review by hand;
commit the cases as your eval set.

**Hybrid pattern**: ~50 hand-curated cases (the ones that matter
most) + ~500 synthetic cases (coverage). Hand-curated cases are
weighted higher in the score.

### Adversarial test cases

A subset of your eval set should be **specifically adversarial**:

- Prompt injection attempts ("Ignore previous instructions...")
- Out-of-scope ("What's my horoscope today?")
- PII probes ("What's user alice's home address?")
- Refusal cases ("Tell me how to commit fraud")

These are the cases users complain about loudest when broken.

## Part 7 · Statistical significance

"V3 scored 0.74; V2 scored 0.72. V3 wins." → **not necessarily.**

On a 50-case eval, the standard error on a binary pass-rate is
~`sqrt(p(1-p)/n)` ≈ 6%. A 2-point difference is well within noise.

**Honest signals:**

- **≥ 5-point difference** on N ≥ 30 cases — probably real
- **≥ 3-point difference** on N ≥ 100 cases — probably real
- **Smaller gaps**: bootstrap-CI; or accept it might be noise

```python
import numpy as np

def bootstrap_ci(passes_a, passes_b, n_boot=1000):
    """95% CI on the difference in pass rates."""
    diffs = []
    for _ in range(n_boot):
        sa = np.random.choice(passes_a, size=len(passes_a), replace=True)
        sb = np.random.choice(passes_b, size=len(passes_b), replace=True)
        diffs.append(sa.mean() - sb.mean())
    return np.percentile(diffs, [2.5, 97.5])
```

If the 95% CI includes 0, you can't tell which is better.

## Run it

```bash
uv add deepeval ragas promptfoo
uv run python -m lessons.35_evaluation_discipline.example
uv run python -m lessons.35_evaluation_discipline.example --judge
uv run python -m lessons.35_evaluation_discipline.example --synthetic
uv run python -m lessons.35_evaluation_discipline.example --bootstrap
```

The script:

1. Defines a tiny eval set (10 cases) covering PTO / refund / remote work.
2. Runs `lookup_handbook` against both v1 and v2 prompts.
3. Scores each output with three scorers (exact, contains, LLM-as-judge).
4. Aggregates per-version and prints a comparison table.
5. **`--judge`**: shows the rubric-based judge with JSON output.
6. **`--synthetic`**: prints the generator prompt + a sample of generated cases.
7. **`--bootstrap`**: shows the bootstrap CI on the pass-rate diff.

## Anti-patterns

| Smell | Fix |
|---|---|
| "We eyeball the demo before shipping" | That isn't eval. Build a 30-case suite |
| One frontier-model judge scoring every metric | Use deterministic scorers where you can; reserve judges for free-text |
| Score = pass/fail with no rubric | 1-5 Likert with named dimensions |
| Same model judges its own output | Cross-model; or pin a frontier judge |
| No statistical significance check | At least eyeball N; bootstrap when in doubt |
| Eval set never grows | Add a case every time you fix a bug. The suite becomes a regression net |
| Synthetic-only eval set | Hand-curate the 20 most important cases; synthetic for breadth |
| No adversarial cases | "Should refuse" + "out of scope" + "PII probe" — non-negotiable |
| Eval lives in a notebook, never runs in CI | Move it into pytest or promptfoo, wire to GH Actions |
| Judge model is an alias (`gpt-4o`) | Pin the version (`gpt-4o-2024-08-06`) |

## Pairs with

- **[Lesson 26 · Misc](../26_misc/README.md)** — Topic 4 is this lesson's primer; this is the practice
- **[Lesson 32 · Prompt engineering lab](../32_prompt_engineering_lab/README.md)** — the registry promotes only on eval improvement
- **[Lesson 34 · Observability](../34_observability_tracing/README.md)** — eval scores show up alongside trace data
- **[Lesson 19 · Guardrails](../19_guardrails/README.md)** — judge nodes are an in-flight version of these scorers

## References

- [promptfoo docs](https://www.promptfoo.dev/docs/) — declarative eval CLI
- [RAGAS docs](https://docs.ragas.io/) — RAG-specific metrics
- [deepeval docs](https://docs.confident-ai.com/) — pytest-native
- [Inspect AI](https://inspect.aisi.org.uk/) — UK AISI framework
- [LangSmith evals](https://docs.smith.langchain.com/evaluation) — first-party
- [Zheng et al. · Judging LLM-as-a-Judge (2023)](https://arxiv.org/abs/2306.05685) — the position-bias paper
- [HELM benchmark](https://crfm.stanford.edu/helm/) — Stanford's holistic eval
- [Anthropic · How we test Claude](https://www.anthropic.com/research/evaluating-ai-systems) — frontier-lab eval methodology
- [Eugene Yan · Evals patterns](https://eugeneyan.com/writing/llm-evaluators/) — practitioner essays

## Next →

[Lesson 36 · The AI engineering library landscape](../36_library_landscape/README.md) — the libraries that surround eval/observability/prompts.
