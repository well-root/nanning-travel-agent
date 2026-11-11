from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from nanning_travel_agent.agent.graph import SYSTEM_PROMPT, build_graph
from nanning_travel_agent.mcp.client import load_metro_tools
from nanning_travel_agent.rag.tools import search_nanning_docs

RECURSION_LIMIT = 20


def build_llm() -> ChatOpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
        temperature=0.1,
    )


def print_trace(messages: list) -> None:
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            print(f"  [调用] {tc['name']}({tc['args']})")
        if m.type == "tool":
            print(f"  [返回] {str(m.content)[:60]}…")


async def main() -> None:
    llm = build_llm()
    tools = [search_nanning_docs, *(await load_metro_tools())]
    graph = build_graph(llm, tools)
    config = {"configurable": {"thread_id": "cli"}, "recursion_limit": RECURSION_LIMIT}

    print("南宁出行助手（输入 exit 退出）")
    first_turn = True
    seen = 0

    while True:
        question = input("\n你: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "/exit"}:
            break

        payload = [HumanMessage(question)]
        if first_turn:
            payload.insert(0, SystemMessage(SYSTEM_PROMPT))
            first_turn = False

        result = await graph.ainvoke({"messages": payload}, config=config)

        print_trace(result["messages"][seen:])
        seen = len(result["messages"])
        print(f"助手: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())