"""跨会话长期记忆：Store 存用户偏好，每轮开始注入提示，模型也可主动要求记住。

与会话记忆的分工：
- checkpoints.sqlite 存"这次聊了什么"（按 thread_id 隔离）
- store.sqlite 存"这个人是谁、习惯什么"（按 user_id 隔离，跨会话共享）
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.base import BaseStore

MEMORY_DIR = Path(__file__).resolve().parents[3] / "data" / "memory"
STORE_DB = MEMORY_DIR / "store.sqlite"

DEFAULT_USER = "local"
MAX_INJECT = 8


def _namespace(user_id: str) -> tuple[str, str]:
    return ("user", user_id or DEFAULT_USER)


def user_id_of(config: RunnableConfig | None) -> str:
    """从 config 取 user_id，没配就用 local。"""
    if not config:
        return DEFAULT_USER
    return (config.get("configurable") or {}).get("user_id") or DEFAULT_USER


async def load_memories(store: BaseStore, user_id: str, limit: int = MAX_INJECT) -> list[str]:
    items = await store.asearch(_namespace(user_id), limit=limit)
    facts = [str(item.value.get("fact", "")).strip() for item in items]
    return [fact for fact in facts if fact]


def memory_prompt(memories: list[str]) -> str:
    lines = "\n".join(f"- {fact}" for fact in memories)
    return f"以下是用户以前告诉过你的长期信息，回答时直接用上，不要再问一遍：\n{lines}"


def make_remember_tool(store: BaseStore):
    """生成 remember_fact 工具；store 用闭包带进去，工具签名里就不会出现注入参数。"""

    @tool
    async def remember_fact(fact: str, config: RunnableConfig) -> str:
        """记住用户的长期信息，例如常住地、常走的路线、出行偏好（"我住广西大学""每周五去朝阳广场"）。

        只在用户明确说了关于自己的、以后还用得上的信息时调用；一次一句，不要存临时的查询条件。
        """
        clean = fact.strip()
        if not clean:
            return "没听到要记的内容"
        await store.aput(
            _namespace(user_id_of(config)),
            f"fact-{time.time_ns()}",
            {"fact": clean, "created_at": datetime.now().isoformat(timespec="seconds")},
        )
        return f"已记住：{clean}"

    return remember_fact
