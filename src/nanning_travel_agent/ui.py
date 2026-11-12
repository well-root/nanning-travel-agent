#ui界面


from __future__ import annotations

import asyncio
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from nanning_travel_agent.agent.graph import build_graph
from nanning_travel_agent.metro_mcp.metro_client import load_metro_tools
from nanning_travel_agent.rag.tools import search_nanning_docs

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CHECKPOINT_DIR = ROOT / "data" / "memory"
CHECKPOINT_DB = CHECKPOINT_DIR / "checkpoints.sqlite"
RECURSION_LIMIT = 20


def build_llm(model: str, temperature: float) -> ChatOpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model=model,
        temperature=temperature,
    )


async def run_turn(question: str, thread_id: str, model: str, temperature: float):
    """一次问答在自己的事件循环里建库连接、加载工具、跑图，避免跨 loop 复用对象。"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tools = [search_nanning_docs, *(await load_metro_tools())]

    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        graph = build_graph(build_llm(model, temperature), tools, saver)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
        }
        before = len((await graph.aget_state(config)).values.get("messages", []))
        result = await graph.ainvoke({"messages": [HumanMessage(question)]}, config=config)

    new_messages = result["messages"][before:]
    calls = [
        (tc["name"], tc["args"])
        for m in new_messages
        for tc in (getattr(m, "tool_calls", None) or [])
    ]
    outputs = [str(m.content) for m in new_messages if m.type == "tool"]
    return str(result["messages"][-1].content), calls, outputs


st.set_page_config(page_title="南宁出行助手", page_icon="🚇", layout="wide")
st.title("🚇 南宁出行助手")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("会话")
    thread_id = st.text_input("thread_id", value="ui-local")
    model = st.selectbox("模型", ["deepseek-flash", "deepseek-chat"])
    temperature = st.slider("temperature", 0.0, 1.0, 0.1, 0.05)
    if st.button("清屏（服务端历史保留）"):
        st.session_state.messages = []
        st.rerun()
    st.caption("换 thread_id = 换一条会话；历史存在 data/memory/checkpoints.sqlite")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("calls"):
            with st.expander(f"工具调用（{len(msg['calls'])} 次）"):
                for (name, args), output in zip(msg["calls"], msg.get("outputs", [])):
                    st.code(f"{name}({args})", language="json")
                    st.caption(output[:300])

question = st.chat_input("比如：从朝阳广场到火车东站怎么走")
if question:
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("查资料 / 查换乘中…"):
            answer, calls, outputs = asyncio.run(
                run_turn(question, thread_id, model, temperature)
            )
        st.markdown(answer)
        if calls:
            with st.expander(f"工具调用（{len(calls)} 次）"):
                for (name, args), output in zip(calls, outputs):
                    st.code(f"{name}({args})", language="json")
                    st.caption(output[:300])

    st.session_state.messages.append(
        {"role": "assistant", "text": answer, "calls": calls, "outputs": outputs}
    )