"""FastAPI"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from nanning_travel_agent.agent.graph import build_graph
from nanning_travel_agent.agent.llm import build_llm
from nanning_travel_agent.agent.memory import STORE_DB
from nanning_travel_agent.metro_mcp.metro_client import load_metro_tools
from nanning_travel_agent.rag.tools import search_nanning_docs

RECURSION_LIMIT = 20
CHECKPOINT_DB = ROOT / "data" / "memory" / "checkpoints.sqlite"
STATIC_DIR = Path(__file__).resolve().parent / "static"
HITL = os.environ.get("HITL", "0") == "1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    async with (
        AsyncSqliteStore.from_conn_string(str(STORE_DB)) as store,
        AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver,
    ):
        tools = [search_nanning_docs, *(await load_metro_tools())]
        app.state.graph = build_graph(build_llm(), tools, saver, store)
        app.state.tool_names = [tool.name for tool in tools]
        yield


app = FastAPI(title="南宁出行助手 API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str = "api-local"
    user_id: str = "local"
    resume: bool = False  # True 时 message 是补充给 htil 反问的回答


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["ok", "need_input"]
    answer: str = ""
    question: str = ""
    tools: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0


def _config(req: ChatRequest) -> dict:
    return {
        "configurable": {"thread_id": req.thread_id, "user_id": req.user_id},
        "recursion_limit": RECURSION_LIMIT,
    }


def _graph_input(req: ChatRequest):
    if req.resume:
        return Command(resume=req.message)
    return {"messages": [HumanMessage(req.message)]}


def _turn_tools(messages: list) -> list[str]:
    """调用过的工具名。"""
    names: list[str] = []
    for message in reversed(messages):
        if message.type == "human":
            break
        names.extend(call["name"] for call in (getattr(message, "tool_calls", None) or []))
    return list(reversed(names))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "hitl": HITL, "tools": app.state.tool_names}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    started = time.perf_counter()
    result = await app.state.graph.ainvoke(_graph_input(req), config=_config(req))
    elapsed = int((time.perf_counter() - started) * 1000)
    tools = _turn_tools(result["messages"])
    pending = result.get("__interrupt__") or []
    if pending:
        return ChatResponse(thread_id=req.thread_id, status="need_input",
                            question=str(pending[0].value.get("question", "")),
                            tools=tools, elapsed_ms=elapsed)
    return ChatResponse(thread_id=req.thread_id, status="ok",
                        answer=str(result["messages"][-1].content),
                        tools=tools, elapsed_ms=elapsed)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    graph = app.state.graph
    config = _config(req)

    async def events():
        started = time.perf_counter()
        async for chunk, meta in graph.astream(_graph_input(req), config=config, stream_mode="messages"):
            # 只转发 agent 节点的最终回答；摘要、planner 的 token 不能漏给前端
            if meta.get("langgraph_node") != "agent":
                continue
            if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str) and chunk.content:
                payload = {"type": "token", "text": chunk.content}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        state = await graph.aget_state(config)
        pending = [item for task in state.tasks for item in getattr(task, "interrupts", ())]
        if pending:
            payload = {"type": "need_input", "question": pending[0].value.get("question", "")}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        payload = {"type": "done", "elapsed_ms": int((time.perf_counter() - started) * 1000)}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
