# Claude Code skills

Five focused, opinionated Claude Code skills derived from the
patterns in this repo. Drop them into `~/.claude/skills/` (or any
subdirectory of it) and they become available to every Claude
Code session on your machine.

## What's inside

| Folder | Skill name (triggered by) | One-liner |
|---|---|---|
| [`python-design-patterns-applied/`](python-design-patterns-applied/SKILL.md) | `python-design-patterns-applied` | Practical Python patterns — Adapter, Factory, Strategy, Decorator, Context Manager, Observer, Registry, Repository — with worked code from real multi-provider systems |
| [`python-clean-code/`](python-clean-code/SKILL.md) | `python-clean-code` | Naming, type hints, error handling, dataclasses vs Pydantic vs TypedDict, docstrings, module structure, code smells |
| [`fastapi-pytest-functional/`](fastapi-pytest-functional/SKILL.md) | `fastapi-pytest-functional` | Functional-style (NOT class-based) pytest for FastAPI + general HTTP API testing. TestClient, AsyncClient, fixtures, parametrize, respx, schema assertions, WebSockets |
| [`langgraph-1x-engineering/`](langgraph-1x-engineering/SKILL.md) | `langgraph-1x-engineering` | LangGraph 1.x patterns — StateGraph, checkpointers, HITL, streaming modes, subgraphs, supervisor, swarm, Store, guardrails |
| [`langchain-1x-engineering/`](langchain-1x-engineering/SKILL.md) | `langchain-1x-engineering` | LangChain 1.x patterns — LCEL, switchable provider adapter, structured output, tools, `create_agent` + middleware, RAG with Chroma + FastEmbed |

All five reflect the **May 2026 stack** (LangChain 1.3.x,
LangGraph 1.2.x, FastAPI 0.115+, Pydantic 2.x, `uv` package manager)
and the patterns the rest of this repo demonstrates.

## Install

### Option A · Copy (simple, works everywhere)

```bash
# from the repo root
cp -r skills ~/.claude/skills/langgraph-playground
```

Restart Claude Code (or run `/reload-skills` if your install supports
it). Skills appear in the available-skills list under their own
names — see the table above.

On Windows PowerShell:

```powershell
Copy-Item -Recurse skills $env:USERPROFILE\.claude\skills\langgraph-playground
```

### Option B · Symlink (single source of truth)

If you want edits in this repo to take effect immediately on your
machine, symlink instead of copying.

macOS / Linux:

```bash
ln -s "$(pwd)/skills" ~/.claude/skills/langgraph-playground
```

Windows (PowerShell, needs admin **or** Developer Mode on):

```powershell
New-Item -ItemType SymbolicLink `
  -Path "$env:USERPROFILE\.claude\skills\langgraph-playground" `
  -Target "$(Resolve-Path .\skills)"
```

## How a skill triggers

Claude Code reads the `description:` field from each `SKILL.md`'s
YAML frontmatter and decides whether to invoke the skill based on
the user's request. Every description here ends with a clear
*"Use when …"* clause — keep that pattern when you edit them.

## Edit / fork freely

These are plain Markdown files. No build, no toolchain. Edit them
to taste — change a default, swap an example, add a section. Reload
Claude Code (or run `/reload-skills`) to pick up changes.

If you want one of these to **override** an identically-named skill
from another source, change the `name:` field in its frontmatter to
exactly match the one you want to replace.

## Naming convention

Names use `-applied`, `-1x`, `-functional` suffixes so they don't
collide with similarly-purposed skills shipped by other plugins.
Each skill includes a *"Pairs with …"* note pointing at sibling
skills it complements.

## License

MIT — same as the rest of the repo. Use them, fork them, share them.

## Contributing back

Found a bug, want to add a pattern, or noticed a v1 API has shifted
again? Open a PR — each skill is a single file, easy to review.
