# Lesson 28 · Dynamic prompting with Jinja2

> "Prompts are code. Treat them like code."

Until now, every lesson has built prompts with `ChatPromptTemplate.from_messages` and `{variable}` placeholders. That works for simple substitution, but breaks when you need:

- **Conditional sections** ("if the user is on the enterprise plan, include section X")
- **Loops** ("for each retrieved doc, render `<doc id=1>...</doc>`")
- **Inheritance** (a base "company persona" that 12 different agents extend)
- **Includes** (the same "safety rules" snippet shared between every prompt)
- **Filters** (truncate, lowercase, json-encode, escape XML tags)
- **Version control** (`prompts/v3/critic.j2` vs `prompts/v4/critic.j2`)

For all of those, you reach for **Jinja2** — the same templating
engine Django, Flask, and Ansible use. LangChain has first-class
support: pass `template_format="jinja2"` to `PromptTemplate` and you
get the full Jinja syntax, with the bonus that the prompt is still a
LangChain Runnable you can pipe.

## What you'll learn

| Concept | Jinja syntax | Why it matters |
|---|---|---|
| Variable | `{{ user_name }}` | Same as f-string but inside templates |
| Conditional | `{% if plan == "enterprise" %} ... {% endif %}` | Branch by user tier, locale, A/B group |
| Loop | `{% for doc in docs %} ... {% endfor %}` | Render retrieved chunks, tool descriptions, examples |
| Include | `{% include "safety_rules.j2" %}` | Reuse the same snippet in 10 prompts |
| Inheritance | `{% extends "agent_base.j2" %}` + `{% block role %}{% endblock %}` | Base persona + per-agent overrides |
| Filter | `{{ context \| truncate(2000) }}` | Cap context length, escape, format dates |
| Macro | `{% macro doc_block(d) %}<doc id="{{ d.id }}">{{ d.text }}</doc>{% endmacro %}` | Reusable rendering helpers |

## The three-tier prompt structure that scales

Your `prompts/` folder should look like this:

```
prompts/
├── _base/
│   ├── persona.j2          ← "You are <name>, an assistant for <company>..."
│   ├── safety_rules.j2     ← shared refusal policy
│   └── output_format.j2    ← "Reply in <tag>...</tag>"
├── agents/
│   ├── researcher.j2       ← extends persona, fills in role + tools section
│   ├── writer.j2
│   └── critic.j2
└── tasks/
    ├── summarise.j2        ← one-shot prompts, no agent loop
    └── classify.j2
```

The base layer never changes. Agent prompts inherit and override. Task
prompts are flat. This three-tier shape mirrors how production codebases
end up after their first refactor — start here and you save yourself
the refactor.

## Pattern A — Inline Jinja inside `PromptTemplate`

When the template is short and lives next to its caller:

```python
from langchain_core.prompts import PromptTemplate

tpl = PromptTemplate.from_template(
    """You are answering on behalf of {{ company }}.
    {% if vip %}This user is a VIP — be extra-attentive.{% endif %}

    Retrieved context:
    {% for doc in docs %}
    [{{ loop.index }}] {{ doc.title }} — {{ doc.snippet | truncate(200) }}
    {% endfor %}

    Question: {{ question }}
    """,
    template_format="jinja2",
)

prompt = tpl.invoke({"company": "Acme", "vip": True, "docs": [...], "question": "..."})
```

Note: **`template_format="jinja2"` is the magic flag.** Without it you
get the default `f-string`-style parser, which doesn't understand `{%
%}` blocks.

## Pattern B — File-based templates with `jinja2.Environment`

When the template is long, reused across files, or needs `{% include
%}` / `{% extends %}`:

```python
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

PROMPTS_DIR = Path(__file__).parent / "prompts"

env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    autoescape=select_autoescape(disabled_extensions=("j2",)),
    trim_blocks=True,           # strip the trailing \n after {% %} blocks
    lstrip_blocks=True,         # strip leading whitespace before {% %}
    keep_trailing_newline=False,
)

def render(template_name: str, **kw) -> str:
    return env.get_template(template_name).render(**kw)

system_prompt = render("agents/researcher.j2", tools=tools, locale="tr-TR")
```

`trim_blocks` + `lstrip_blocks` keep the rendered output clean — without
them you get a spaghetti of blank lines from every `{% if %}` boundary.

## Pattern C — Jinja inside a LangChain agent's `system_prompt`

`create_agent` takes a `system_prompt: str | Callable`. The callable
form receives the runtime state, so you can render *per-call*:

```python
from langchain.agents import create_agent
from langgraph.runtime import Runtime

def build_system_prompt(state, runtime: Runtime) -> str:
    user = runtime.context.get("user", {})
    return env.get_template("agents/support.j2").render(
        user=user,
        tier=user.get("tier", "free"),
        history_len=len(state["messages"]),
    )

agent = create_agent(
    model=get_llm(),
    tools=[...],
    system_prompt=build_system_prompt,
)
```

This is the **dynamic** in dynamic prompting: the system prompt
changes *every turn* based on state. Plain string prompts can't do
that.

## Production patterns

### 1 · Versioned templates

```
prompts/
├── researcher.v1.j2
├── researcher.v2.j2     ← current
└── researcher.v3.j2     ← canary, A/B-rolling
```

In code, pick the version by env var or feature flag:

```python
template_version = settings.prompt_version or "v2"
tpl = env.get_template(f"researcher.{template_version}.j2")
```

This is what production teams do instead of editing prompts in place
— makes rollback a one-line config change.

### 2 · A/B testing with hashing

```python
import hashlib

def pick_variant(user_id: str, variants: list[str]) -> str:
    h = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
    return variants[h % len(variants)]

variant = pick_variant(user_id, ["v2", "v3"])
tpl = env.get_template(f"researcher.{variant}.j2")
```

Same user → same variant (sticky). Track which variant served which
response in your observability layer so you can compare quality.

### 3 · Prompt registry pattern

Instead of file paths, build a small registry:

```python
class PromptRegistry:
    def __init__(self, env: Environment):
        self.env = env
        self._cache: dict[tuple[str, str], Template] = {}

    def get(self, name: str, version: str = "current") -> Template:
        key = (name, version)
        if key not in self._cache:
            self._cache[key] = self.env.get_template(f"{name}.{version}.j2")
        return self._cache[key]

    def render(self, name: str, *, version: str = "current", **vars) -> str:
        return self.get(name, version).render(**vars)
```

Wrap it in a class so swapping for a remote registry (LangSmith Hub,
S3, your own DB-backed store) is a one-file change.

### 4 · Lock-protected hot reload (ties into lesson 27)

If you reload templates from disk while requests are in flight, you
need a lock:

```python
import asyncio
_reload_lock = asyncio.Lock()

async def reload_templates():
    async with _reload_lock:
        global env
        env = Environment(loader=FileSystemLoader(PROMPTS_DIR), ...)
```

Without the lock, a half-loaded `Environment` can serve a partial
template. This is exactly the pattern from lesson 27 — *shared mutable
state, multiple coroutines, lock around the mutation*.

## Common Jinja idioms for LLM prompts

```jinja
{# Render the tool list with descriptions #}
{% for tool in tools %}
- **{{ tool.name }}**: {{ tool.description }}
{% endfor %}

{# Conditional context #}
{% if conversation_summary %}
Earlier in this conversation:
{{ conversation_summary }}
{% endif %}

{# Truncate user content #}
User message: {{ user_msg | truncate(4000, killwords=True) }}

{# JSON-encode for structured prompts #}
Available actions: {{ actions | tojson }}

{# Inheritance — agent_base.j2 #}
You are {{ agent_name }}, an assistant for {{ company }}.
{% block role %}{% endblock %}
{% block rules %}{% include "_base/safety_rules.j2" %}{% endblock %}

{# Inheritance — researcher.j2 #}
{% extends "agent_base.j2" %}
{% block role %}
You search the web and record citations using the `cite` tool.
{% endblock %}
```

## Dynamic-prompting anti-patterns

| Smell | Fix |
|---|---|
| Building prompts with `+` concatenation across 50 lines of `if/else` | Move to a `.j2` file, use `{% if %}` blocks |
| Using `.format()` and forgetting one slot | Jinja raises `UndefinedError` at render time — make it loud (`StrictUndefined`) |
| HTML-autoescaping a prompt | Disable it. `autoescape` is for HTML output — prompts are XML/markdown |
| Embedding user input without escaping | Use a custom filter or strip `</` sequences to prevent prompt-injection via "close my tag" tricks |
| Reloading `Environment()` on every request | Build it once at process start; templates are cached internally |
| Storing prompts in code as multi-line strings forever | Move to files the moment you have > 2 prompts or > 30 lines |
| One mega-template with 20 `{% if %}` branches | Refactor into a base + 4 children using `{% extends %}` — much easier to diff |

## Run it

```bash
uv run python -m lessons.28_dynamic_prompting.example
```

The script runs four demos:

1. **Inline** — `PromptTemplate.from_template(..., template_format="jinja2")`
2. **File-based** — loads `prompts/researcher.j2` and `prompts/critic.j2`
3. **Inheritance** — `agent_base.j2` + a child override
4. **Dynamic per-call** — `system_prompt=callable` for `create_agent`

No API key needed for demos 1-3. Demo 4 uses a fake LLM stub.

## Try it yourself

1. Add a `{% if locale == "tr-TR" %}` block to `researcher.j2` that inserts Turkish-specific instructions (e.g. "Türkçe cevap ver") — this is how the lesson 23 localisation pattern composes with prompts.
2. Build a prompt registry that picks `researcher.{version}.j2` and falls back to `researcher.j2` if the version is missing.
3. Wire **sandbox mode** on the `Environment` (`jinja2.sandbox.SandboxedEnvironment`) — when the template variables are user-controllable, you do NOT want them to execute arbitrary Python.

## Pairs with

- **[Lesson 11 · Agent middleware](../11_agent_middleware/README.md)** — `before_model` middleware can mutate the prompt; Jinja is the right tool to render the mutation.
- **[Lesson 22 · Architecture](../22_architecture/README.md)** — the prompt registry sits in the "Prompt / Behaviour" layer.
- **[Lesson 27 · Locks](../27_locks_and_concurrency/README.md)** — hot-reloading templates needs a lock.
- **[Lesson 32 · Prompt engineering lab](../32_prompt_engineering_lab/README.md)** — A/B testing, evals, and the registry pattern at scale.

## References

- [Jinja2 docs](https://jinja.palletsprojects.com/) — the official manual.
- [`PromptTemplate.from_template(..., template_format="jinja2")`](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html#langchain_core.prompts.prompt.PromptTemplate.from_template) — LangChain's Jinja flag.
- [`SandboxedEnvironment`](https://jinja.palletsprojects.com/en/stable/sandbox/) — for user-supplied templates.
- [LangSmith Hub](https://smith.langchain.com/hub) — first-party prompt registry; great target for the registry pattern in step 2 of the exercises.

## Next →

[Lesson 29 · Vector databases](../29_vector_databases/README.md) — pgvector, Qdrant, hybrid search, when to pick which.
