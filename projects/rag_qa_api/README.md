# Capstone · `rag_qa_api`

A production-shaped **RAG Q&A service**. A two-node LangGraph
(`retrieve → generate`) wrapped in FastAPI, persisted to Postgres via
`AsyncPostgresSaver`, dockerised, and discoverable by LangGraph Studio
through `langgraph.json`.

## Architecture

```
client ──HTTP──▶ FastAPI ──invoke──▶ LangGraph
                                       │
                                       ├── retrieve  (Chroma + FastEmbed)
                                       └── generate  (Anthropic / OpenAI via shared.llm)

state checkpointed to: Postgres
```

## Concepts exercised

| Lesson | Used for |
|---|---|
| 06 · RAG basics | Chroma index over `data/sample_docs/` |
| 08 · LangGraph basics | `StateGraph(QAState)` with two nodes |
| 12 · persistence | `AsyncPostgresSaver` keyed by `thread_id` |
| 14 · streaming | `stream_mode="messages"` for SSE token streaming |

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/healthz` | — | `{"status": "ok"}` |
| POST | `/chat` | `{thread_id, message}` | `{reply}` |
| POST | `/stream` | `{thread_id, message}` | SSE stream of `token` events, ending with `done` |
| GET | `/docs` | — | Interactive OpenAPI (FastAPI default) |

## Run it (Docker — recommended)

```bash
cd projects/rag_qa_api
docker compose up --build
```

Then in another shell:

```bash
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"thread_id": "t1", "message": "How many PTO days do I get?"}'
```

Stream tokens with SSE:

```bash
curl -N -X POST localhost:8000/stream \
  -H 'content-type: application/json' \
  -d '{"thread_id": "t1", "message": "Summarise the refund policy in 3 bullets."}'
```

Open <http://localhost:8000/docs> for the interactive OpenAPI explorer.

## Run it (local, without Docker)

You need a running Postgres reachable at the URL in your `.env`'s
`POSTGRES_URL`. Then:

```bash
uv sync --extra api
uv run uvicorn projects.rag_qa_api.app:app --reload
```

## LangGraph Studio

The `langgraph.json` exposes the graph as `qa`. With the LangGraph
CLI installed (`uv tool install langgraph-cli[inmem]`), run from this
directory:

```bash
langgraph dev
```

The Studio UI then visualises the graph and lets you step through runs
node-by-node.

## Try it yourself

- Add an `auth` middleware in FastAPI that requires a header `X-API-Key`.
- Replace the `retrieve_node` retriever with the `MultiQueryRetriever` from lesson 07.
- Add a third node `cite_sources` that appends a `Sources:` footer to the reply using the `metadata.source` of the retrieved chunks.
- Use `stream_mode="updates"` instead of `"messages"` so the client receives per-node JSON updates rather than tokens.
