"""并行取数子图冒烟：看它一次规划几个调用、耗时多少、合并结果全不全。"""

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from nanning_travel_agent.agent.gather import build_gather_graph, make_gather_tool
from nanning_travel_agent.agent.llm import build_llm
from nanning_travel_agent.metro_mcp.metro_client import load_metro_tools
from nanning_travel_agent.rag.tools import search_nanning_docs

QUESTION = "从朝阳广场到火车东站怎么走，大概要多长时间，票价是多少？"


async def main() -> int:
    llm = build_llm()
    tools = [search_nanning_docs, *(await load_metro_tools())]
    print(f"工具池 {len(tools)} 个: {[t.name for t in tools]}\n")

    graph = build_gather_graph(llm, tools)
    started = time.perf_counter()
    state = await graph.ainvoke({"query": QUESTION})
    elapsed = time.perf_counter() - started

    calls = getattr(state.get("plan_message"), "tool_calls", None) or []
    final = state.get("final") or ""
    names = [call["name"] for call in calls]

    print(f"问题: {QUESTION}")
    print(f"计划调用 {len(calls)} 个: {[(c['name'], c['args']) for c in calls]}")
    print(f"子图耗时: {elapsed:.2f}s")
    print(f"合并结果 {len(final)} 字，带标记的工具: {[n for n in set(names) if f'[{n}]' in final]}")
    print("-" * 60)
    print(final[:800])

    gather_tool = make_gather_tool(llm, tools)
    text = await gather_tool.ainvoke({"query": QUESTION})
    print(f"\n工具入口返回 {len(text)} 字: {text[:200]}")

    ok = len(calls) >= 2 and len(final) > 0
    print(f"\n{'PASS' if ok else 'FAIL'}: 计划调用 {len(calls)} 个（要求 >= 2）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

