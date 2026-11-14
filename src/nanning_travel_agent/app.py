from __future__ import annotations

import asyncio
import os
from pathlib import Path
from langgraph.types import Command
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

from nanning_travel_agent.agent.graph import build_graph
from nanning_travel_agent.agent.llm import build_llm
from nanning_travel_agent.agent.memory import STORE_DB, load_memories
from nanning_travel_agent.metro_mcp.metro_client import load_metro_tools
from nanning_travel_agent.rag.tools import search_nanning_docs

RECURSION_LIMIT = 20
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "memory"
CHECKPOINT_DB = CHECKPOINT_DIR / "checkpoints.sqlite"
USER_ID = os.environ.get("USER_ID", "local")


def print_trace(messages: list) -> None:
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            print(f"  [调用] {tc['name']}({tc['args']})")
        if m.type == "tool":
            print(f"  [返回] {str(m.content)[:60]}…")


async def main() -> None:
    llm = build_llm()
    tools = [search_nanning_docs, *(await load_metro_tools())]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    async with (
        AsyncSqliteStore.from_conn_string(str(STORE_DB)) as store,
        AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver,
    ):
        graph = build_graph(llm, tools, saver, store)
        config = {
            "configurable": {"thread_id": f"cli-{USER_ID}", "user_id": USER_ID},
            "recursion_limit": RECURSION_LIMIT,
        }

        memories = await load_memories(store, USER_ID)
        print(f"（长期记忆 {len(memories)} 条，user={USER_ID}）")

        restored = (await graph.aget_state(config)).values.get("messages", [])
        seen = len(restored)
        if restored:
            print(f"（已恢复 {len(restored)} 条历史，thread=cli-{USER_ID}）")
        print("南宁出行助手（输入 exit 退出）")

        while True:
            question = input("\n你: ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit", "/exit"}:
                break

            result = await graph.ainvoke({"messages": [HumanMessage(question)]}, config=config)

            while True:
                interrupts = result.get("__interrupt__") or []
                if not interrupts:
                    break
                print(f"助手: {interrupts[0].value['question']}")
                answer = input("你: ").strip()
                if not answer:
                    # 空输入不能直接 break，否则图一直挂在断点上，下一轮没法正常开新对话
                    answer = "不记得了，你按查得到的回答"
                result = await graph.ainvoke(Command(resume=answer), config=config)

            print_trace(result["messages"][seen:])
            seen = len(result["messages"])
            print(f"助手: {result['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())