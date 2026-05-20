---
name: deep-thinking
description: Structured analytical reasoning toolkit — first-principles decomposition, Bayesian updating, premortem/postmortem, steelmanning, devil's advocate, falsifiability check, decision matrices (cost/value/risk), mental models (Hanlon's razor, Goodhart's law, second-order effects, base rates), system-1 vs system-2 toggle, "what would have to be true" reverse reasoning, and Bayes-shaped note-taking. Use when the user asks for an honest assessment, a decision under uncertainty, an evaluation of an argument or plan, "think hard about this", or any task that benefits from slowing down and being explicit about the reasoning. Pairs with scientific-paper-researcher and llm-expert.
---

# Deep Thinking

A toolkit for **slowing the brain down on purpose**. Use the
framework that fits the question — not all of them — and **state
which one you're using** so the reader can check your work.

## When to invoke

- "Is this a good idea?" / "Help me decide between A and B."
- "Why might this fail?" / "What am I missing?"
- "Think through this carefully."
- "Steelman the opposite view."
- "Is this claim actually supported?"
- Strategic / architectural decisions with real consequences.

## Operating principles

1. **Pick one framework per question.** Mixing all of them at once dilutes everything.
2. **Show your work.** The framework name + the inputs + the conclusion. Not just the conclusion.
3. **State your priors.** "I'm walking in with a prior of 70% that X works because …" Then update them.
4. **Distinguish what you know from what you assume.** Two different colours of ink.
5. **Numbers > vibes** for anything that can be quantified, even loosely. "20% confident" beats "I'm not sure."
6. **Premortem before you commit.** Imagine the failure first; the upside takes care of itself.
7. **One conclusion at a time.** A reasoning trace that ends in "but on the other hand…" hasn't concluded.

## The frameworks

Pick by question shape:

| Question shape | Framework |
|---|---|
| "Why does this work?" / "What's the root cause?" | **First-principles decomposition** |
| "Should we do A or B?" | **Decision matrix** + **premortem** |
| "Is this claim true?" | **Steelman + falsifiability + base rates** |
| "What's the probability of X?" | **Bayesian update** with explicit prior |
| "What could go wrong?" | **Premortem** + **second-order effects** |
| "Why did this fail?" | **Postmortem** + 5 Whys |
| "Is this the best argument for the other side?" | **Steelman** |
| "Are we measuring the right thing?" | **Goodhart check** |

---

### 1 · First-principles decomposition

Strip the question to its **physical / mathematical / definitional**
foundations, then rebuild.

```
Question:    Why is our agent slow?
Assumed:     "Agents are slow" (vague)
Decompose:   total latency = prefill + decode + tool_time + network
             prefill  = f(prompt_length, model_size, GPU)
             decode   = output_tokens × time-per-token
             tool_time = sum of external calls
Re-aggregate: which term dominates? Measure each. Optimise the biggest.
```

**Trigger phrase to use:** *"Let me decompose this …"*

### 2 · Bayesian update

State a **prior probability**, list the evidence, update **explicitly**.

```
Prior:        P(this RAG change improves accuracy) = 30%
Evidence 1:   Similar change in paper X moved metric +5pp.    ↑ → 50%
Evidence 2:   Their data is more structured than ours.        ↓ → 40%
Evidence 3:   We've tried a related change before — no win.   ↓ → 30%
Posterior:    ~30%. Worth a small experiment, not a big commitment.
```

Even rough numbers beat verbal hedging. Be specific about *how much*
each piece of evidence moves you, not just the direction.

### 3 · Premortem ("imagine it's six months from now and this failed")

Best before any irreversible commitment.

```
We shipped X six months ago. It failed. Why?

Mode A · The model degraded under real traffic patterns we didn't test.
Mode B · The tool we depended on changed its API.
Mode C · Our eval was contaminated; production showed something else.
Mode D · A cheaper competitor shipped the same thing better.
Mode E · The bottleneck wasn't the model — it was {data, latency, UX}.

For each: how likely (1-5)? how bad (1-5)? what would prevent it?
```

**The output is the prevention list**, not the failure list. Fix the
top 2–3 risks before launch.

### 4 · Postmortem (blameless, structural)

After a failure. Distinguish:

- **Trigger** — the immediate event ("API rate-limited at 3pm").
- **Cause** — the structural reason it could happen ("no circuit breaker").
- **Detection** — how we noticed ("user complaint, not monitoring").
- **Response** — what we did ("rolled back in 12 minutes").
- **Lesson** — the change that prevents the *class* of failure.

Specifically *not* "Alice forgot the timeout." Always **what change
to the system would have caught this**.

### 5 · Steelman (the strongest version of the opposing view)

Before disagreeing with X, write the **best possible** version of X
in their own terms. If the steelman is weak, you can disagree freely.
If the steelman is strong, your disagreement needs to engage with it,
not the weak version.

```
Claim:           "Fine-tuning is rarely the right answer."
Steelman:        "If your task has a fixed format, distinct domain
                  vocabulary, and millions of consistent examples,
                  a fine-tuned 3B model can outperform a prompted 70B
                  at a fraction of the inference cost. RAG can't fix
                  format compliance; prompting can't internalise
                  domain syntax that's never appeared on the open web."
Engagement:      "Agreed for those constraints. My claim assumes the
                  user is in the early-iteration regime — fewer than
                  1000 examples, evolving format. Once the format is
                  frozen and data is plentiful, fine-tune."
```

### 6 · Falsifiability check

"What evidence would change my mind?"

If the answer is "nothing," the belief is not a position — it's a
posture. Rephrase until you can answer the question concretely.

```
Belief: "Our agent is going to be popular."
Q:      What concrete evidence would change my mind in 6 months?
A:      <100 weekly active users, churn >70%, NPS <0.

Now the belief is falsifiable. Track those numbers.
```

### 7 · Base-rate sanity check

Most predictions ignore base rates. Always ask:

> "Of all things in this reference class, what fraction succeed?"

```
"Most SaaS startups fail in the first 5 years (~80%)."
"Most ML research papers don't replicate at deployed scale (~50–80%)."
"Most refactors run 2–3× longer than estimated."

Now: is there a specific reason this case beats the base rate?
```

### 8 · Goodhart check ("the measure is not the territory")

Whenever a metric becomes a target, it stops being a good measure.

```
Goal:        Better customer-support agent.
Metric:      Resolution time.
Goodhart:    Agents close tickets faster by giving worse answers.
Fix:         Track resolution time + reopen rate + customer-rated quality.
             No single metric controls anything important.
```

### 9 · Second-order effects

First-order: "If we add X, then Y."
Second-order: "And then *because of Y*, Z."

```
1st:   Lower the price → more sales.
2nd:   More sales → more support tickets → support overwhelmed → quality drops.
3rd:   Quality drops → churn → net negative.

So: cap inflow, or staff up support, or don't lower the price.
```

### 10 · Decision matrix (when comparing options)

```
                  Option A   Option B   Option C
Quality (0-5)     4          3          5
Speed (0-5)       2          5          1
Cost (0-5)        3          5          2
Risk (0-5, low=hi) 4          5          2
Sum                13         18         10
```

Tag any column that's **load-bearing** — if Option B fails on quality
above the threshold, no amount of speed compensates. Sums are a
heuristic, not an answer; eyeball the cells.

### 11 · "What would have to be true?"

Reverse-engineer the world in which your plan succeeds.

```
For our agent to hit 80% task success by Q3, what has to be true?
1.  Our eval set is representative of production traffic.
2.  The chosen model meets 80% on our eval today.
3.  No major regression from prompt drift.
4.  Tool latency stays under 500ms.

Now: how confident are we in each? Which are we most likely wrong about?
```

### 12 · Mental models worth memorising

- **Hanlon's razor** — don't attribute to malice what's adequately explained by stupidity / accident / process gaps.
- **Chesterton's fence** — before removing a thing whose purpose you don't understand, find out *why it's there*.
- **Goodhart's law** — when a measure becomes a target, it ceases to be a good measure.
- **Sturgeon's law** — 90% of everything is mediocre. Calibrate expectations.
- **Survivorship bias** — you only see the survivors. The 100 failed competitors with the same plan are invisible.
- **Conservation of complexity** — complexity moves around; it rarely disappears. Beware "simple" solutions that just push the mess somewhere else.
- **The 80/20 rule (Pareto)** — applies to bugs, costs, users, and almost everything else. Find the 20%.
- **Reversible vs irreversible decisions** — reversible: decide fast. Irreversible: think longer.
- **The Lindy effect** — for things that survive on usefulness (libraries, ideas), the longer they've existed, the longer they're likely to keep existing.
- **Brandolini's law** — refuting bullshit takes 10× the energy of producing it. Pick your battles.

---

## The deep-thinking session structure

When asked to "think hard about X":

```
1. RESTATE the question in your own words.
2. ENUMERATE assumptions you're making.
3. PICK 1–2 frameworks from above (name them).
4. APPLY them — show the work, including the numbers.
5. CONCLUDE with one sentence + one confidence number ("60%").
6. NAME the strongest counter-argument to your conclusion.
7. STATE the cheapest experiment that would resolve the uncertainty.
```

If steps 5 and 6 sound the same, your "conclusion" is a hedge.
Pick a side or admit you can't yet.

## Output format

Default to **structured, scannable prose**:

```markdown
**Question.** <restated>

**Framework.** Bayesian update with explicit prior.

**Prior.** I walk in at 30%, because <reasons>.

**Evidence.**
- Evidence 1 (+12pp): …
- Evidence 2 (-8pp): …
- Evidence 3 (+5pp): …

**Posterior.** ~39%.

**Conclusion.** Don't commit. Run the cheaper experiment first.

**Counter.** The strongest case for committing now is <…>. The
reason I don't find it persuasive is <…>.

**Cheapest test.** Spend half a day on <X>. If <metric> > <threshold>,
my prior should jump to 60%+ and we revisit.
```

For decisions: a **decision matrix** table + a 3-sentence verdict.

For arguments: a **steelman → engagement → conclusion** pattern.

## Anti-patterns

| Smell | Fix |
|---|---|
| "On the one hand … on the other hand …" with no verdict | Force a number / probability. Pick a side. |
| "It depends" without enumerating the variables | List the variables and give the answer per branch. |
| Hedging with "potentially could possibly" | Say what you actually mean, or say "I don't know." |
| Throwing 5 frameworks at one question | Pick one. They aren't a checklist. |
| Conclusion ≈ restatement of the question | Make a *commitment*, even a probabilistic one. |
| Ignoring base rates | Ask "of all <reference class>, what fraction succeed?" first. |
| "We tried X and it didn't work" without specifics | What was tried, what was measured, what was the threshold? |
| Critiquing a strawman | Steelman first; then critique. |

## Pairs with

- `scientific-paper-researcher` — for the "is this claim actually supported by evidence" step
- `llm-expert` — for technical decisions about models / training / inference
- `python-design-patterns-applied` — for the "what's the right shape of code" decision
- Existing skills: `scientific-critical-thinking`, `scholar-evaluation`, `peer-review` (for formal review work)
