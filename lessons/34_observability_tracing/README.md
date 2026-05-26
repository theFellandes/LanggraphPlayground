# Lesson 34 · LLM observability + tracing

> "You can't fix what you can't see."

You ship a LangGraph app. Latency spikes. A user complains the bot
hallucinated. The cost dashboard shows yesterday was 3× normal. You
have **no idea** which prompt, which model, which retrieval, which
tool call is responsible. This lesson is the fix.

LLM observability is **not the same** as regular APM. Three reasons:

1. **Non-deterministic outputs.** The same input produces different
   outputs across calls. You can't reason from "what changed" without
   trace-level fidelity.
2. **Expensive primitives.** A single LLM call is $0.01-$5. A
   single bad prompt change × 1M users = $50k mistake. Cost belongs
   in the trace, not just the metrics layer.
3. **Multi-hop opacity.** "The agent called a tool that ran a
   retrieval that hit a vector DB" — each layer has its own latency
   and failure mode. You need a tree, not a flat log.

## What you'll learn

| # | Topic |
|---|---|
| 1 | **LangSmith** — the first-party tracer (runs, datasets, evaluators, prompts) |
| 2 | **Langfuse** — the self-hosted alternative |
| 3 | **OpenTelemetry for LLMs** — the vendor-neutral path |
| 4 | **Arize Phoenix** — local-first / OSS evaluation + observability |
| 5 | **What to log per call** — the canonical metadata set |
| 6 | **Cost dashboards** — turning token counts into per-user / per-feature dollars |
| 7 | **Sampling at scale** — when 100% tracing isn't affordable |

## Topic 1 · LangSmith — first-party, hosted

The default for LangChain/LangGraph apps. Set two env vars; every
chain/agent run lights up in the LangSmith UI.

```bash
# .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=my-app
```

That's the entire setup. Every `llm.invoke(...)`, every node in a
graph, every tool call is now a span in a tree.

### What you get

| Feature | What it does |
|---|---|
| **Runs** | One per invocation. Tree of nested spans (chain → llm → tool → ...) |
| **Datasets** | Curated `(input, expected)` pairs. Reusable across eval runs |
| **Evaluators** | LLM-as-judge or programmatic scorers attached to runs |
| **Comparisons** | A/B two prompt versions, see deltas in any column |
| **Prompts (Hub)** | Versioned, sharable prompt templates |
| **Annotations** | Manual ratings + comments by humans |

### The metadata pattern

You can attach arbitrary metadata to any invocation. **Use it
ruthlessly** — every field becomes a filterable column:

```python
from shared import get_llm

reply = get_llm().invoke(
    prompt,
    config={
        "metadata": {
            "prompt_version": "v3",
            "prompt_sha": "a7b2c9d1",
            "user_segment": "enterprise",
            "tenant_id": "acme",
            "ab_variant": "treatment",
            "request_id": "req_abc123",
        },
        "tags": ["customer-support", "refund-flow"],
        "run_name": "support-bot-turn",
    },
)
```

In LangSmith you can then filter "show all refund-flow runs from
enterprise users that took > 5s." Without this metadata, you're
flying blind.

### Eval with LangSmith

Datasets + evaluators run in CI:

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 1. Curate a dataset (one-time)
examples = [
    {"input": "How many PTO days?", "expected": "20"},
    {"input": "Refund a $250 order?", "expected": "requires human approval"},
]
client.create_dataset(dataset_name="support-bot-v1")
client.create_examples(
    inputs=[{"question": e["input"]} for e in examples],
    outputs=[{"answer": e["expected"]} for e in examples],
    dataset_name="support-bot-v1",
)

# 2. Run evaluation
def llm_under_test(inputs):
    return {"answer": get_llm().invoke(inputs["question"]).content}

def correctness(run, example):
    pred = run.outputs.get("answer", "").lower()
    gold = example.outputs.get("answer", "").lower()
    return {"key": "correctness", "score": int(gold in pred)}

evaluate(
    llm_under_test,
    data="support-bot-v1",
    evaluators=[correctness],
    experiment_prefix="prompt-v3",
)
```

The score history is now a chart in the UI. Regressions are visible
at a glance.

## Topic 2 · Langfuse — self-hosted, OSS

Same shape as LangSmith but you run it. Critical when:

- Data sovereignty rules forbid sending traces to a US-hosted SaaS
- You want unlimited retention without paying per-trace
- You're already on a Postgres + ClickHouse stack and adding another vendor is friction

Setup:

```bash
docker compose -f langfuse-docker-compose.yml up -d
```

```python
# pip install langfuse
from langfuse.callback import CallbackHandler

handler = CallbackHandler(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="http://localhost:3000",
)

reply = get_llm().invoke(prompt, config={"callbacks": [handler]})
```

LangChain calls `handler.on_*` for each span; Langfuse builds the
trace. UI is comparable to LangSmith.

**Where Langfuse pulls ahead**: integrates with **non-LangChain** code
(OpenAI SDK direct, Anthropic SDK direct) via decorators — `@observe`.
Useful if you have services that don't use LangChain.

## Topic 3 · OpenTelemetry — the vendor-neutral path

OTel is the **CNCF standard** for distributed tracing. The OTel
**semantic conventions for generative AI** (stable as of mid-2025)
define common attribute names: `gen_ai.system`, `gen_ai.prompt`,
`gen_ai.response.id`, `gen_ai.usage.prompt_tokens`, etc.

Why bother with OTel when LangSmith works?

- **Vendor portability.** Same instrumentation feeds LangSmith, Datadog, Honeycomb, Tempo, Jaeger — without code changes.
- **Unified observability.** Your LLM spans sit inside the same trace as your FastAPI request span, your Postgres query span, your Redis call. One UI shows the whole request.
- **No vendor lock-in.** If LangSmith doubles its pricing tomorrow, you flip an env var.

### Setup with Traceloop's `openllmetry`

```python
# pip install traceloop-sdk
from traceloop.sdk import Traceloop

Traceloop.init(
    app_name="langgraph-playground",
    api_endpoint="http://otel-collector:4318",   # or any OTLP endpoint
    disable_batch=False,
)
```

That auto-instruments LangChain, OpenAI, Anthropic, vector stores,
and emits OTel spans. Then point the OTel collector at any backend:

```yaml
# otel-collector-config.yaml
exporters:
  otlp/langsmith:
    endpoint: "api.smith.langchain.com:443"
  jaeger:
    endpoint: "jaeger:14250"
  datadog:
    api: { key: "${DD_API_KEY}" }
service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [otlp/langsmith, jaeger, datadog]   # fan out
```

Same trace, three backends. The OTel collector is what makes this
work without code changes.

### What the OTel GenAI spec defines

| Attribute | Example |
|---|---|
| `gen_ai.system` | `"anthropic"`, `"openai"` |
| `gen_ai.request.model` | `"claude-sonnet-4-6"` |
| `gen_ai.request.temperature` | `0.7` |
| `gen_ai.response.id` | `"msg_01ABC..."` |
| `gen_ai.usage.input_tokens` | `1234` |
| `gen_ai.usage.output_tokens` | `567` |
| `gen_ai.usage.total_cost_usd` | `0.0123` (your derived field) |
| `gen_ai.prompt.{i}.role` | `"system"` / `"user"` / `"assistant"` |
| `gen_ai.prompt.{i}.content` | the actual text (gate behind a flag for PII) |

Use these names. If you invent your own, every downstream tool
breaks on the next vendor switch.

## Topic 4 · Arize Phoenix — local-first OSS

Phoenix runs **in-process** as a local UI:

```python
# pip install arize-phoenix
import phoenix as px
px.launch_app()                # opens http://localhost:6006

from phoenix.otel import register
tracer_provider = register(
    project_name="my-app",
    endpoint="http://localhost:6006/v1/traces",
)
```

Then any OTel-instrumented call (Traceloop, OpenInference, your own)
appears in Phoenix's UI. Strongest at **eval introspection** — built-in
visualisers for embedding drift, retrieval quality, hallucination
detection.

Use when: you want zero infra and your laptop is sufficient. Switch
to Langfuse/LangSmith when you need shared dashboards.

## Topic 5 · What to log per call (the canonical set)

Every LLM call should emit:

| Field | Why |
|---|---|
| `request_id` | Correlate to your service logs |
| `user_id` (hashed) | Per-user cost / quality analysis |
| `tenant_id` | Multi-tenant cost attribution |
| `prompt_name` + `prompt_version` + `prompt_sha` | What ran; replayable |
| `model` + `provider` | Which vendor + which tier |
| `temperature` | Reproducibility |
| `input_tokens` / `output_tokens` / `cached_tokens` | Cost math |
| `cost_usd` | Pre-computed by your code; faster to query than recomputing |
| `latency_ms` (total + per-step) | The slow ones cluster |
| `tools_called` (list of names) | Tool-use patterns |
| `error` (if any) | Failure analysis |
| `ab_variant` | Required for experiment analysis |
| `user_segment` | Cohort analysis ("enterprise" vs "free") |
| `feature` / `route` | Which product feature called this |

This is ~15 fields. Stick them on every call. The query patterns you
unlock later are worth 10× the noise now.

## Topic 6 · Cost dashboards

The CFO question: "what does our AI bill look like by feature, by
user segment, by week?"

The pipeline:

```
LLM call → trace (LangSmith / Langfuse / OTel) → BigQuery / ClickHouse / Snowflake
                                                  ↓
                                            Looker / Metabase / Grafana
```

Three dashboards every team builds in the first year:

1. **Cost by feature** — which product flow burns money. Drives prompt-shortening + caching investments.
2. **Cost by user cohort** — the 1% of "power users" who cost 50% of the bill. Drives rate-limiting + tier-pricing decisions.
3. **Latency P95 by route** — which flow is slow. Drives streaming / caching / smaller-model decisions.

LangSmith has rudimentary versions of all three. For real
business-intelligence, export to your warehouse.

## Topic 7 · Sampling at scale

At 100k requests/day, 100% trace retention costs ~$5k/month on
hosted services. Mitigations:

| Strategy | Trade-off |
|---|---|
| **Head sampling** — random 10% | Loses rare bugs; misses the slow tail |
| **Tail sampling** — keep all errors + slow (P95+) traces + 1% of the rest | Best signal-to-noise; needs OTel tail-sampling processor |
| **Per-user sticky sampling** | Same user always sampled → coherent traces; harder to implement |
| **Importance sampling** — keep paid users / VIPs at 100%, free at 1% | Aligns cost with revenue |

Head sampling is the default mistake. Use OTel's tail-sampling
processor instead — it's a config knob, not a code change.

## Run it

```bash
uv add langsmith langfuse traceloop-sdk arize-phoenix

# Set env vars in .env, then:
uv run python -m lessons.34_observability_tracing.example
uv run python -m lessons.34_observability_tracing.example --langsmith
uv run python -m lessons.34_observability_tracing.example --langfuse
uv run python -m lessons.34_observability_tracing.example --otel
```

The script:

1. Runs a tiny LangGraph (retrieve → generate) with rich metadata.
2. Emits to LangSmith if `LANGSMITH_API_KEY` set.
3. Emits OTel spans to a local collector (or stdout if none).
4. Prints the canonical metadata set so you can see what gets logged.

## Anti-patterns

| Smell | Fix |
|---|---|
| Tracing only failures | Trace successes too — that's how you measure quality and cost |
| Logging the full prompt at INFO | PII risk. Gate behind a flag; log a sha + variable bindings instead |
| `print()` debugging in production | Use structured logging (`structlog`) so traces and logs cross-reference |
| Per-call instrumentation, no centralised metadata | Build a `with_metadata(...)` helper your whole codebase uses |
| One trace per request | A graph run is one trace; nested calls are spans. Tree, not list |
| Vendor-locked attribute names | Use OTel's `gen_ai.*` namespace from day one |
| Sampling at the head | Errors get sampled out. Use tail sampling |
| No cost attribution per feature | "AI cost too high" → "which feature?" should be one click |

## Pairs with

- **[Lesson 22 · Architecture](../22_architecture/README.md)** — observability is layer 7 of the 8 layers
- **[Lesson 26 · Misc](../26_misc/README.md)** — Topic 2 cost estimation; this lesson is how you measure it in production
- **[Lesson 32 · Prompt engineering lab](../32_prompt_engineering_lab/README.md)** — `prompt_version` / `prompt_sha` belong in the trace
- **[Lesson 35 · Evaluation](../35_evaluation_discipline/README.md)** — eval scores feed the same datasets
- **[Lesson 19 · Guardrails](../19_guardrails/README.md)** — safety violations are a metric

## References

- [LangSmith docs](https://docs.smith.langchain.com/) — first-party
- [Langfuse docs](https://langfuse.com/docs) — self-hosted
- [OpenTelemetry GenAI Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the canonical attribute names
- [Traceloop / openllmetry](https://github.com/traceloop/openllmetry) — auto-instrumentation
- [Arize Phoenix](https://docs.arize.com/phoenix) — OSS local-first
- [Datadog LLM Observability](https://docs.datadoghq.com/llm_observability/) — if you're already on Datadog
- [Honeycomb LLM tracing](https://www.honeycomb.io/blog/instrument-application-llm-models-opentelemetry) — OTel-native
- [Helicone](https://www.helicone.ai/) — proxy-based observability, no code changes

## Next →

[Lesson 35 · Evaluation as a discipline](../35_evaluation_discipline/README.md) — the eval workflows that read these traces.
