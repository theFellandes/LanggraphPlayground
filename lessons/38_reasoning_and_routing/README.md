# Lesson 38 · Reasoning models + routing

The 2024-2025 paradigm shift: a new class of LLMs that **think before
answering**. OpenAI's o1/o3/o4, Anthropic's Claude extended thinking,
DeepSeek R1. They trade latency and cost for *substantially* better
performance on math, code, and multi-step reasoning.

The downstream architecture question: **which model for which query?**
Easy queries don't need a reasoner — that's like running a finite-element
analysis to add 2 + 2. Hard queries shouldn't be sent to a cheap model
that will confidently get it wrong. This lesson is about **routing**:
the discipline of dispatching each query to the right tier of model.

## What you'll learn

1. **What reasoning models actually do** — internal chain-of-thought, why it works
2. **The 2026 lineup** — o3, Claude extended thinking, R1, when each wins
3. **The test-time compute paradigm** — "spend more on inference to skip training" mental model
4. **Routing strategies** — semantic, LLM-as-router, RouteLLM's learned classifier
5. **Cost cascading** — try cheap first; escalate on failure or low confidence
6. **The unified routing recipe in LangGraph** — one router node that dispatches across tiers

## Part 1 · What reasoning models do

The standard LLM samples one token at a time, fast. A reasoning model
samples **a long internal scratchpad first**, hidden from the user,
then produces the final answer. The scratchpad can be 1k-50k tokens —
the model is "thinking out loud" to itself.

This is **test-time compute**: spending more inference budget on the
hard problem at hand, rather than spending it during training. Two
benefits:

1. **Better answers on hard problems** — math olympiad, complex codebases, multi-hop reasoning
2. **Scaling without retraining** — you can give it 10× the thinking time and get measurably better results

The downside: **latency and cost**. A reasoning model on a hard
problem can take 30 seconds and 50,000 tokens to answer. The same
problem on Claude Sonnet 4.6 takes 3 seconds and 2,000 tokens — and
might be just as right on easy questions.

## Part 2 · The 2026 lineup

| Model | Provider | Strength | Cost (rough, output tokens) |
|---|---|---|---|
| **o3 / o3-mini** | OpenAI | Math, code, agentic planning | $60 / $4 per 1M out |
| **o4 (when available)** | OpenAI | Next-gen frontier | TBD (higher) |
| **Claude Sonnet 4.6 (extended thinking)** | Anthropic | Code, agentic tool use, careful reasoning | $15 per 1M out (thinking tokens billed separately) |
| **DeepSeek R1** | DeepSeek (open weights) | Open-source reasoning; near-frontier math | ~$2 per 1M out (DeepSeek API) |
| **Gemini 2.5 Pro thinking** | Google | Strong on long-context reasoning | $10 per 1M out |
| **QwQ 32B** | Alibaba (open weights) | OSS reasoning at the 32B scale | Free if self-hosted |

### When reasoning helps and when it doesn't

| Task | Reasoning model? |
|---|---|
| "What's the capital of Belgium?" | No — Haiku/Flash is fine |
| "Refactor this 400-line file" | Yes — code with multi-step dependencies |
| "Find the bug causing this stack trace" | Yes |
| "Summarise this email" | No — straightforward |
| "Solve this leetcode-hard problem" | Yes |
| "Write a 5-paragraph essay" | Marginal — slightly better, much more expensive |
| "Plan a 10-step agentic workflow" | Yes |
| "Translate to French" | No |

The rule of thumb: **reasoning models earn their cost when the answer
quality bottlenecks the workflow** — code that must be correct, math
that's the deliverable, plans where mistakes compound. They are
overkill (and slow) for "ordinary" LLM tasks.

## Part 3 · Using Claude extended thinking

Anthropic exposes the budget directly:

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    thinking={"type": "enabled", "budget_tokens": 8000},
    max_tokens=16000,
)
reply = llm.invoke("Solve: How many distinct primes divide 30030?")
print(reply.content)            # the answer
# Thinking tokens appear in reply.additional_kwargs["thinking"] (when exposed)
```

`budget_tokens` is how many tokens the model is *allowed* to think.
Higher budget → more careful reasoning → more cost. Typical settings:

- **2k** — light reasoning, ~1-2× cost of normal Sonnet
- **8k** — solid reasoning for code / planning
- **32k-64k** — frontier math, proofs, deep research

The thinking tokens are returned but **separated** from the final
answer — your code reads `reply.content` as before; the thinking is
an internal trace.

## Part 4 · Using OpenAI o3 / o4

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="o3-mini", reasoning_effort="high")
# reasoning_effort: "low" | "medium" | "high"
reply = llm.invoke("Solve: ...")
```

OpenAI exposes a single `reasoning_effort` knob instead of a token
budget. Same idea — more effort → more thinking tokens → better
answers + more cost.

### OpenAI reasoning API quirk

o3 doesn't support `temperature` or `top_p` — it's deterministic
sampling. It also doesn't accept `system` messages by default; use
"developer" messages or just put the system prompt at the top of the
user message.

## Part 5 · Routing — the discipline

You don't want to send every query to a reasoning model. You want a
**router node** that picks the right tier.

### Routing strategy 1 — semantic-router (sub-ms, no LLM)

For a small known set of query types:

```python
from semantic_router import Route, RouteLayer
from semantic_router.encoders import FastEmbedEncoder

routes = [
    Route(name="reasoning", utterances=[
        "Solve this math problem",
        "Refactor this code",
        "Debug this stack trace",
        "Plan a 10-step workflow",
    ]),
    Route(name="cheap", utterances=[
        "What's the capital of",
        "Summarise this",
        "Translate to",
        "How do I",
    ]),
]
rl = RouteLayer(encoder=FastEmbedEncoder(), routes=routes)
print(rl("Refactor my Python code").name)        # → "reasoning"
```

Sub-millisecond classification via embedding cosine. **Zero LLM cost
on the routing decision itself.**

### Routing strategy 2 — LLM-as-router (one cheap call)

For broader query distributions:

```python
ROUTER_PROMPT = """You are a query router. Classify the user's request
as one of:
- "reasoning" — needs careful multi-step thinking (math, code, planning, complex analysis)
- "knowledge"  — factual lookup, summarisation, simple Q&A
- "generation" — creative writing, brainstorming, drafting

Reply with ONE word: reasoning, knowledge, or generation.

Query: {query}
"""

def route(query: str) -> str:
    out = get_llm("anthropic", model="claude-haiku-4-5").invoke(
        ROUTER_PROMPT.format(query=query),
    )
    return out.content.strip().lower()
```

One cheap call → one of N tiers. Slower than semantic-router but
generalises beyond a fixed set.

### Routing strategy 3 — RouteLLM (learned classifier)

```python
from routellm.controller import Controller

client = Controller(
    routers=["mf"],              # matrix-factorisation router
    strong_model="anthropic/claude-sonnet-4-6",
    weak_model="anthropic/claude-haiku-4-5",
)
result = client.chat.completions.create(
    model="router-mf-0.11643",    # threshold tunes cost-quality tradeoff
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
```

Pre-trained on a public quality dataset; you just pick a threshold.
Empirically saves 30-70% cost at ~98% of strong-model quality.

### Cascading — try cheap first, escalate

```python
async def cascading_invoke(query: str) -> tuple[str, str]:
    """Returns (answer, tier_used)."""
    # 1. Try cheap model first.
    cheap_out = await get_llm("anthropic", model="claude-haiku-4-5").ainvoke(query)
    cheap_text = cheap_out.content

    # 2. Score confidence — heuristic: did it answer or punt?
    if _looks_uncertain(cheap_text):
        # 3. Escalate to reasoning model.
        strong_out = await reasoning_llm.ainvoke(query)
        return strong_out.content, "reasoning"
    return cheap_text, "cheap"

def _looks_uncertain(text: str) -> bool:
    markers = ["I'm not sure", "I don't have enough", "can't determine", "unclear"]
    return any(m.lower() in text.lower() for m in markers)
```

A cleaner alternative: ask the cheap model to **score its own
confidence**, and use that to route. Empirically, cheap models *can*
self-grade with reasonable accuracy.

## Part 6 · Putting it together in LangGraph

```python
from typing import Literal, TypedDict
from langgraph.graph import END, START, StateGraph

class State(TypedDict):
    query: str
    tier: Literal["reasoning", "cheap"]
    answer: str

def route_node(state):
    tier = "reasoning" if _needs_reasoning(state["query"]) else "cheap"
    return {"tier": tier}

def cheap_node(state):
    out = get_llm("anthropic", model="claude-haiku-4-5").invoke(state["query"])
    return {"answer": out.content}

def reasoning_node(state):
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        thinking={"type": "enabled", "budget_tokens": 8000},
        max_tokens=16000,
    )
    out = llm.invoke(state["query"])
    return {"answer": out.content}

def pick(state):
    return state["tier"]

g = StateGraph(State)
g.add_node("route", route_node)
g.add_node("cheap", cheap_node)
g.add_node("reasoning", reasoning_node)
g.add_edge(START, "route")
g.add_conditional_edges("route", pick, {"cheap": "cheap", "reasoning": "reasoning"})
g.add_edge("cheap", END)
g.add_edge("reasoning", END)

graph = g.compile()
```

Drop a router in front of any existing graph. The blast radius is
small (one node + one conditional edge) and the cost savings are
typically 50-80% on mixed traffic.

## Part 7 · Cost math, made concrete

Hypothetical workload: 1M queries / month, mixed difficulty.

| Strategy | Average cost / 1k queries | Monthly cost |
|---|---|---|
| All Sonnet | ~$15 | $15,000 |
| All Haiku | ~$1 | $1,000 |
| **70% Haiku + 30% Sonnet** (router) | ~$5 | $5,000 |
| **All o3-mini** | ~$50 | $50,000 |
| **70% Haiku + 25% Sonnet + 5% o3 (reasoning only when needed)** | ~$7 | $7,000 |

The routing tier is **easily the best ROI investment** in a production
LLM app. A 4-hour effort can save tens of thousands of dollars/month
without measurably hurting quality, if your eval (lesson 35) backs it
up.

## Run it

```bash
uv run python -m lessons.38_reasoning_and_routing.example
uv run python -m lessons.38_reasoning_and_routing.example --thinking
uv run python -m lessons.38_reasoning_and_routing.example --route
uv run python -m lessons.38_reasoning_and_routing.example --cascade
```

The script:

1. **`--thinking`** demonstrates Claude extended thinking on a math problem with two budget levels and compares the answers.
2. **`--route`** runs three queries through a heuristic router and shows which tier each picked.
3. **`--cascade`** demonstrates the cheap → reasoning escalation pattern.

## Anti-patterns

| Smell | Fix |
|---|---|
| Routing all queries to the reasoning model | Cost burn; latency burn; quality on easy queries is unchanged |
| Routing with `random.choice` | Use semantic-router or a learned router (RouteLLM) |
| Hard-coded model name in 200 places | One factory: `get_llm(tier="reasoning")`. Refactor early |
| Thinking budget too low | Worse than no thinking — model gets confused. 2k minimum |
| No eval gating router changes | Lesson 35 says always. Don't skip |
| Reasoning model used in interactive chat without streaming | Users wait 30s with no feedback. Stream the answer or show a "thinking..." spinner |
| Treating o3 as a drop-in for GPT-4o | Different API quirks: no temperature, different message shape |

## Pairs with

- **[Lesson 26 · Misc](../26_misc/README.md)** — Topic 2's cost math is the input to the router decision
- **[Lesson 35 · Evaluation](../35_evaluation_discipline/README.md)** — measure router quality before claiming it works
- **[Lesson 36 · Library landscape](../36_library_landscape/README.md)** — semantic-router and RouteLLM details
- **[Lesson 09 · Conditional edges](../09_conditional_edges/README.md)** — the routing primitive in LangGraph
- **[Lesson 22 · Architecture](../22_architecture/README.md)** — "Routing" is layer 5 of the 8 layers

## References

- [Anthropic · Claude extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) — the budget knob
- [OpenAI · Reasoning models](https://platform.openai.com/docs/guides/reasoning) — `reasoning_effort`
- [DeepSeek R1 paper · 2025](https://arxiv.org/abs/2501.12948) — open-weights reasoning
- [RouteLLM paper · Ong et al. 2024](https://arxiv.org/abs/2406.18665) — the learned router
- [Snell et al. · Scaling test-time compute (2024)](https://arxiv.org/abs/2408.03314) — the test-time-compute paradigm explained
- [Hugging Face · QwQ 32B](https://huggingface.co/Qwen/QwQ-32B-Preview) — open-weights reasoning model
- [semantic-router GitHub](https://github.com/aurelio-labs/semantic-router) — the lightweight router

## Next →

[`ml_foundations/06 · Fine-tuning LLMs`](../../ml_foundations/06_finetuning_llms/README.md) — when prompting + RAG + routing aren't enough.
