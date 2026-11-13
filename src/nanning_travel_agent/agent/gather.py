"""并行取数子图：模型一次规划所有工具调用，扇出并行执行，再合并成一段资料。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

PLANNER_PROMPT = """你是南宁地铁出行助手的取数调度器。
读用户问题，决定要调用哪些工具、各传什么参数；需要的信息一次发全，它们会被并行执行。
同一份信息不要重复查，只查问题里确实需要的。"""

MAX_RESULT_CHARS = 1500


def _to_text(output) -> str:
    """MCP 工具返回的是 content 块列表，拼成纯文本；普通字符串原样放行。"""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(output)


class GatherState(TypedDict, total=False):
    query: str
    plan_message: object
    results: Annotated[list[str], operator.add]
    final: str


def build_gather_graph(llm: BaseChatModel, tools: list[BaseTool]):
    """返回编译好的子图：ainvoke({"query": ...}) 之后取 state["final"]。"""
    llm_with_tools = llm.bind_tools(tools)
    by_name = {tool_item.name: tool_item for tool_item in tools}

    async def plan(state: GatherState):
        response = await llm_with_tools.ainvoke(
            [SystemMessage(PLANNER_PROMPT), HumanMessage(state["query"])]
        )
        return {"plan_message": response}

    async def execute(payload: dict):
        call = payload["call"]
        target = by_name.get(call["name"])
        if target is None:
            return {"results": [f"[{call['name']}] 未知工具"]}
        try:
            output = await target.ainvoke(call.get("args") or {})
        except Exception as exc:  # 单路失败不影响其他分支
            return {"results": [f"[{call['name']}] 调用失败：{exc}"]}
        text = _to_text(output)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + "…"
        return {"results": [f"[{call['name']}]\n{text}"]}

    def fan_out(state: GatherState):
        calls = getattr(state.get("plan_message"), "tool_calls", None) or []
        if not calls:
            return [Send("merge", {})]
        return [Send("execute", {"call": call}) for call in calls]

    async def merge(state: GatherState):
        return {"final": "\n\n".join(state.get("results") or [])}

    builder = StateGraph(GatherState)
    builder.add_node("plan", plan)
    builder.add_node("execute", execute)
    builder.add_node("merge", merge)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", fan_out)
    builder.add_edge("execute", "merge")
    builder.add_edge("merge", END)
    return builder.compile()


def make_gather_tool(llm: BaseChatModel, tools: list[BaseTool]):
    """把并行取数子图包成一个复合工具，供主图 ToolNode 调用。"""
    subgraph = build_gather_graph(llm, tools)

    @tool
    async def gather_parallel(query: str) -> str:
        """一次并行查询多个来源（换乘路线 + 票价/运营时间/优惠等文档）。

        当问题同时涉及"怎么走"和"多少钱/几点/怎么优惠"时使用，调用一次胜过分别调多个工具。
        query 写成完整的一句话，包含起点、终点和所有想知道的点。
        """
        state = await subgraph.ainvoke({"query": query})
        return state.get("final") or "（未取到可用信息）"

    return gather_parallel