# Capstone · `research_assistant`

A multi-agent research tool built with the **supervisor pattern**. The
supervisor orchestrates three specialists:

- **researcher** — searches the web via Tavily, records cited claims.
- **writer** — turns cited claims into a structured Markdown report.
- **critic** — reads the draft and either approves it or sends it back to the writer with missing-citation feedback.

## Concepts exercised

- **Lesson 02 · LCEL** — the prompt-template chains inside each agent
- **Lesson 05 · Tools** — `cite` is a custom tool; the researcher uses TavilySearchResults
- **Lesson 10 · `create_agent`** — every worker is a `create_agent`
- **Lesson 16 · Supervisor** — `langgraph_supervisor.create_supervisor` routes between workers

## Prerequisites

You need a Tavily key for the search tool:

```bash
# in .env
TAVILY_API_KEY=tvly-…
```

Get one free at <https://tavily.com>.

## Run it

```bash
uv run python -m projects.research_assistant.graph "What are the latest advances in fusion energy?"
```

If you don't pass a question it defaults to one about fusion energy.

## What you should see

A trace showing the supervisor delegating to the researcher (multiple
tool calls), then to the writer (one big reply), then to the critic
(approval or revision). The final assistant message is the polished
Markdown report.

## Try it yourself

- Add a `fact_checker` worker that re-runs Tavily on every claim and verifies the source.
- Replace the supervisor's `model` with a cheaper Haiku-tier model while keeping the workers on Sonnet — a common cost-tuning pattern.
- Persist the run with a `MemorySaver` checkpointer so you can `get_state_history(cfg)` and replay the workflow.
