# Lesson 32 · Prompt engineering lab

The curriculum closer for prompts. You have:

- Jinja templating (lesson 28)
- An eval framework (lesson 26 Topic 4)
- Locks for hot-reload (lesson 27)

This lesson **wires them together** into a working prompt-engineering
lab: a versioned registry, A/B routing, automated evals, and a
"promote to current" workflow that production teams reinvent at every
company. Tier 6 deep dive — by the end, you have the spine of
LangSmith Hub running locally.

## What you'll learn

1. **Prompt as artefact** — version, tag, hash, store metadata.
2. **The registry pattern** — `registry.get("researcher", version="v3")` instead of file paths.
3. **A/B routing** — sticky-per-user variant assignment.
4. **Eval-driven promotion** — only promote v3 to "current" if its eval score beats v2.
5. **Observability** — emit `prompt_name`, `prompt_version`, `prompt_hash` on every call.
6. **Hot reload** — pick up new template files without restart (with the lock from lesson 27).
7. **Sandboxing** — when prompts are user-supplied, lock them in `SandboxedEnvironment`.

## The artefact view

A prompt is a tuple, not a string:

```python
@dataclass(frozen=True)
class PromptArtefact:
    name: str                    # "researcher"
    version: str                 # "v3"
    template_source: str         # the .j2 content
    sha256: str                  # hash of template_source — change-detection
    metadata: dict[str, str]     # author, created_at, eval scores, etc.
    tags: list[str]              # ["canary"], ["production"], ["experimental"]
```

Once you have artefacts:

- **Reproducibility** — pinning a version pins the *exact* template bytes
- **Observability** — log the sha256 with every call; you can replay later
- **Audit** — "which prompt was used for this customer interaction?" → trace → sha → diff

## The registry pattern

```python
class PromptRegistry:
    def __init__(self, prompts_dir: Path):
        self.dir = prompts_dir
        self._env = Environment(loader=FileSystemLoader(prompts_dir), ...)
        self._cache: dict[tuple[str, str], PromptArtefact] = {}
        self._reload_lock = asyncio.Lock()       # ← from lesson 27

    def list(self, name: str) -> list[str]:
        """All available versions for a prompt name."""
        return sorted(p.stem.split(".")[-1] for p in self.dir.glob(f"{name}.*.j2"))

    def get(self, name: str, version: str = "current") -> PromptArtefact:
        version = self._resolve_version(name, version)
        key = (name, version)
        if key not in self._cache:
            src = (self.dir / f"{name}.{version}.j2").read_text()
            sha = hashlib.sha256(src.encode()).hexdigest()[:16]
            self._cache[key] = PromptArtefact(name, version, src, sha, {}, [])
        return self._cache[key]

    def render(self, name: str, *, version: str = "current", **vars) -> str:
        artefact = self.get(name, version)
        return self._env.from_string(artefact.template_source).render(**vars)

    async def reload(self):
        async with self._reload_lock:
            self._cache.clear()
            self._env = Environment(loader=FileSystemLoader(self.dir), ...)
```

Notes:

- **Cache keyed on `(name, version)`** — the cache is the hot path.
- **`reload()` clears and rebuilds** — guarded by an `asyncio.Lock` so a half-built env doesn't serve a partial template (lesson 27 pattern 1).
- **Versions are filename suffixes** — `researcher.v1.j2`, `researcher.v2.j2`. "current" is a symlink (Linux/macOS) or a tiny pointer file (Windows).

## A/B routing — sticky per-user

Same user → same variant, every time, until you change the assignment.

```python
import hashlib

def assign_variant(user_id: str, variants: list[str]) -> str:
    """Sticky assignment by hash; idempotent."""
    h = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
    return variants[h % len(variants)]

# In the agent's system-prompt callable:
def system_for_turn(state, runtime):
    user_id = runtime.context["user_id"]
    variant = assign_variant(user_id, ["v2", "v3"])
    return registry.render("researcher", version=variant, ...)
```

Why hashing instead of `random.choice`: the same user gets the same
variant on retry, so you don't double-count and you can replay
deterministically.

For **weighted** A/B (90% v2, 10% v3 canary):

```python
def weighted_variant(user_id: str, weights: dict[str, int]) -> str:
    h = int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 100
    cum = 0
    for variant, w in weights.items():
        cum += w
        if h < cum:
            return variant
    return list(weights)[-1]

weighted_variant("alice@acme.com", {"v2": 90, "v3": 10})
```

## Eval-driven promotion

The full loop:

```
1. Author writes prompts/researcher.v4.j2
2. Eval suite runs against v2 (current) and v4 (candidate)
3. If v4 score >= v2 score + epsilon, promote
4. "Promote" = repoint the `current` symlink/pointer
5. Next request gets v4
```

In CI:

```python
def promote_if_better(registry, eval_suite, name, candidate, current, epsilon=0.02):
    cur_score = eval_suite.score(registry.get(name, current))
    new_score = eval_suite.score(registry.get(name, candidate))
    if new_score >= cur_score + epsilon:
        registry.promote(name, candidate)
        return f"promoted {name}@{candidate}: {new_score:.3f} > {cur_score:.3f}"
    return f"holding {name}@{current}: {new_score:.3f} < {cur_score:.3f} + {epsilon}"
```

This is the **single most powerful discipline** in production
LLM-ops. Without it: "we changed a prompt and quality got worse but
nobody noticed for two weeks." With it: every prompt change is gated
on a measured improvement.

## Observability hooks

Every LLM call should emit:

| Field | Why |
|---|---|
| `prompt_name` | "researcher" |
| `prompt_version` | "v3" — pinned at request time |
| `prompt_sha` | First 16 chars of sha256 — survives renames |
| `variant` | If A/B is on |
| `user_segment` | Cohort identifier |
| `latency_ms` | The slow ones cluster around bad prompts |
| `tokens_in / tokens_out` | Cost attribution |

In a LangChain stack, attach these as `metadata={...}` on the model
call. LangSmith picks them up automatically; for self-hosted stacks,
they show up in your traces / logs.

## Hot reload with locks

The classic bug: you edit `researcher.v3.j2` while requests are in
flight; the next request reads a half-saved file and crashes. Fix
with file-watcher + lock:

```python
async def watch_and_reload(registry, interval=2.0):
    last_mtimes = {}
    while True:
        await asyncio.sleep(interval)
        changed = False
        for p in registry.dir.glob("*.j2"):
            mtime = p.stat().st_mtime
            if last_mtimes.get(p) != mtime:
                last_mtimes[p] = mtime
                changed = True
        if changed:
            await registry.reload()        # lock-guarded
```

Production-grade: use `watchfiles` or `watchdog` for an actual inotify
subscription instead of polling.

## Sandboxing user-supplied templates

If your product exposes "let users write their own prompts" (custom
agent builders), **never** let user templates run in a regular `jinja2.Environment`:

```python
{{ self._TemplateReference__context.environment.from_string(...).render() }}
```

This kind of construction can escape and execute arbitrary Python.
The fix: `SandboxedEnvironment`, which blocks attribute access to
dunders and a list of dangerous names.

```python
from jinja2.sandbox import SandboxedEnvironment

sandbox = SandboxedEnvironment(
    loader=FileSystemLoader("user_templates"),
    autoescape=False,
)
```

For higher assurance, use **immutable sandboxes** + a process boundary
(rendering in a subprocess with no filesystem access). For most
SaaS-y use cases, `SandboxedEnvironment` is the bar.

## Run it

```bash
uv run python -m lessons.32_prompt_engineering_lab.example                  # full demo
uv run python -m lessons.32_prompt_engineering_lab.example --registry       # registry mechanics
uv run python -m lessons.32_prompt_engineering_lab.example --ab             # A/B routing
uv run python -m lessons.32_prompt_engineering_lab.example --promote        # eval-driven promotion
```

The lab uses a stubbed LLM that returns a length-based score, so the
eval-driven promotion demo is deterministic — the longer prompt wins,
imagining longer prompts produce more grounded answers (yes, this is
oversimplified; in real life you'd use lesson 26's eval scorer).

## Anti-patterns

| Smell | Fix |
|---|---|
| Editing prompts in production and crossing fingers | Eval-driven promotion in CI |
| `git diff` is the only prompt audit trail | Hash + version + tag, stored alongside traces |
| One mega-template with 30 if/else | Inherit (`{% extends %}`); register as multiple artefacts if needed |
| User-supplied template in `Environment(...)` | `SandboxedEnvironment` minimum |
| Reloading prompts on every request | Cache + watcher; reload only on change |
| A/B with `random.choice` | Hash by user id — sticky assignment is non-negotiable |
| "We A/B'd, v3 wins" with no statistical significance | At least 30-50 cases per variant; ideally bootstrap CI |

## Pairs with

- **[Lesson 28 · Dynamic prompting](../28_dynamic_prompting/README.md)** — the Jinja primitives this lesson registries on top of
- **[Lesson 26 · Miscellaneous](../26_misc/README.md)** — Topic 4's eval framework is what gates promotion
- **[Lesson 27 · Locks](../27_locks_and_concurrency/README.md)** — hot reload needs the mutex
- **[Lesson 19 · Guardrails](../19_guardrails/README.md)** — judge-node evals are another scoring source for promotion

## References

- [LangSmith Hub](https://smith.langchain.com/hub) — the first-party prompt registry
- [Jinja2 SandboxedEnvironment](https://jinja.palletsprojects.com/en/stable/sandbox/)
- [promptfoo eval CI integration](https://www.promptfoo.dev/docs/integrations/ci-cd/)
- [Anthropic prompt cookbook](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) — patterns to encode as templates
- [OpenAI · Prompt engineering guide](https://platform.openai.com/docs/guides/prompt-engineering)

## Next →

[Lesson 33 · Vector database internals](../33_vector_database_internals/README.md) — the under-the-hood companion to lesson 29.

Or pick a capstone — the **pro** variants ([`research_assistant_pro`](../../projects/research_assistant_pro/README.md), [`customer_support_bot_pro`](../../projects/customer_support_bot_pro/README.md), [`rag_qa_api_pro`](../../projects/rag_qa_api_pro/README.md)) wire together everything from Tier 6.

Or hop tracks to [`ml_foundations`](../../ml_foundations/README.md)
to learn what's *inside* the language models you've been calling.
