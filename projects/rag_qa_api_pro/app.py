"""FastAPI front for the rag_qa_api_pro graph.

Endpoints:
  GET  /healthz                       liveness probe
  POST /chat   {thread_id, message,
                tenant_id, tier}      non-streaming reply
  POST /stream {…}                    SSE token stream
  GET  /docs                          OpenAPI explorer (FastAPI default)

Auth: every /chat and /stream request must carry `X-API-Key: <API_KEY>`.

Run:
    uv run uvicorn projects.rag_qa_api_pro.app:app --reload

Or in Docker (recommended — uses the lesson 29 compose stack):
    cd projects/rag_qa_api_pro && docker compose up
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from projects.rag_qa_api_pro.graph import compiled_graph

API_KEY = os.environ.get("API_KEY", "dev-key")


class ChatRequest(BaseModel):
    thread_id: str
    message: str
    tenant_id: str = "default"
    tier: str = "free"


def check_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="bad or missing X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with compiled_graph() as graph:
        app.state.graph = graph
        yield


app = FastAPI(title="LangGraph RAG QA API (pro)", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/chat", dependencies=[Depends(check_api_key)])
async def chat(req: ChatRequest) -> dict:
    graph = app.state.graph
    cfg = {"configurable": {"thread_id": req.thread_id}}
    try:
        result = await graph.ainvoke(
            {
                "messages": [{"role": "user", "content": req.message}],
                "tenant_id": req.tenant_id,
                "tier": req.tier,
                "rewrites": [],
                "candidates": [],
                "relevant": [],
                "cited_answer": "",
            },
            cfg,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"reply": result["messages"][-1].content}


@app.post("/stream", dependencies=[Depends(check_api_key)])
async def stream(req: ChatRequest) -> EventSourceResponse:
    graph = app.state.graph
    cfg = {"configurable": {"thread_id": req.thread_id}}

    async def event_iter():
        async for chunk, _meta in graph.astream(
            {
                "messages": [{"role": "user", "content": req.message}],
                "tenant_id": req.tenant_id,
                "tier": req.tier,
                "rewrites": [],
                "candidates": [],
                "relevant": [],
                "cited_answer": "",
            },
            cfg,
            stream_mode="messages",
        ):
            if getattr(chunk, "content", None):
                yield {"event": "token", "data": chunk.content}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_iter())
