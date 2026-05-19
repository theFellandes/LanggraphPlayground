"""FastAPI front for the RAG Q&A graph.

Endpoints:
  GET  /healthz                       liveness probe
  POST /chat   {thread_id, message}   non-streaming reply
  POST /stream {thread_id, message}   SSE token stream
  GET  /docs                          OpenAPI explorer (FastAPI default)

Run locally with:
    uv run --extra api uvicorn projects.rag_qa_api.app:app --reload

Or in Docker (recommended — gets you a Postgres alongside):
    cd projects/rag_qa_api && docker compose up
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from projects.rag_qa_api.graph import compiled_graph


class ChatRequest(BaseModel):
    thread_id: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Set up one compiled graph per process; reuse for every request."""
    async with compiled_graph() as graph:
        app.state.graph = graph
        yield


app = FastAPI(title="LangGraph RAG QA API", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    graph = app.state.graph
    cfg = {"configurable": {"thread_id": req.thread_id}}
    try:
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": req.message}]},
            cfg,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"reply": result["messages"][-1].content}


@app.post("/stream")
async def stream(req: ChatRequest) -> EventSourceResponse:
    graph = app.state.graph
    cfg = {"configurable": {"thread_id": req.thread_id}}

    async def event_iter():
        async for chunk, _meta in graph.astream(
            {"messages": [{"role": "user", "content": req.message}]},
            cfg,
            stream_mode="messages",
        ):
            if chunk.content:
                yield {"event": "token", "data": chunk.content}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_iter())
