"""双工具路由图：模型自己决定查文档（RAG）还是查换乘（MCP），工具调用过多时强制收口。"""

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage,RemoveMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from nanning_travel_agent.agent.hitl import ask_user
from nanning_travel_agent.agent.memory import (
    load_memories, make_remember_tool, memory_prompt, user_id_of,
)

SYSTEM_PROMPT = """你是南宁地铁出行助手。
票价、优惠政策、运营时间、车票使用这类问题，用 search_nanning_docs 查文档。
两个车站之间的路线、站数、耗时，用 find_metro_route 查。
线路覆盖、某条线经过哪些站、某站在哪条线上，用 list_lines / line_stations 查。
站名不确定时先用 search_station 确认准确站名，不要凭印象拼。
工具返回 NO_RELEVANT_DOCS、站名不存在或查不到时，直说查不到，不要编。
用户说了自己的常住地、常走的路线、出行偏好（"我住广西大学""每周五去朝阳广场"）时，用 remember_fact 记下来，一次一句。
路线问题（怎么走、坐几号线、几站、多久）缺起点或终点时，先调 ask_user 问清楚，不要自己猜一个站。
站名查不到（比如"南宁东站"不是地铁站）时，用 ask_user 确认用户是不是指别的站。
问过一次就不要再问，按已有信息回答，查不到的直接说查不到。"""

MAX_TOOL_ROUNDS = int(os.environ.get("MAX_TOOL_ROUNDS", "6"))
NO_MORE_TOOLS_HINT = """工具调用次数已到上限。现在只能用已经拿到的信息回答：
不要再说“我去查一下”，没查到的部分直接说明查不到，然后给出已知的部分。"""
ALREADY_ASKED_HINT = """已经问过用户一次了。现在只能用已经拿到的信息回答，
不要再反问，也不要猜站名；确实缺的部分直接说明查不到。"""

SUMMARY_THRESHOLD = int(os.environ.get("SUMMARY_THRESHOLD", "10"))
KEEP_RECENT_TURNS = 4
SUMMARY_PROMPT = """把下面的对话压成不超过三句话的摘要，保留：用户是谁、常从哪出发、有什么偏好，以及已经查到的关键结论。
不要寒暄，不要编造没出现过的信息。"""


def _tool_rounds(messages: list) -> int:
    """数本轮（最后一条人类消息之后）模型请求过几次工具。"""
    rounds = 0
    for message in reversed(messages):
        if message.type == "human":
            break
        if getattr(message, "tool_calls", None):
            rounds += 1
    return rounds


def _asked_before(messages: list) -> bool:
    """本轮是否已经反错过。ask_user 返回的是 ToolMessage，靠 name 认。"""
    for message in reversed(messages):
        if message.type == "human":
            return False
        if message.type == "tool" and getattr(message, "name", "") == "ask_user":
            return True
    return False


def _summary_split(messages: list, keep_recent: int):
    """按照每一轮来切分内容"""
    humans = [i for i, m in enumerate(messages) if m.type == "human"]
    if len(humans) <= keep_recent:
        return None
    cut = humans[-keep_recent]
    return messages[:cut], messages[cut:]

def build_graph(llm, tools, checkpointer=None, store: BaseStore | None = None,
                hitl: bool | None = None):
    hitl = (os.environ.get("HITL", "0") == "1") if hitl is None else hitl

    base_tools = [*tools, make_remember_tool(store)] if store is not None else list(tools)
    tool_set = [*base_tools, ask_user] if hitl else base_tools
    # 两个绑定：问之前挂 ask_user，问过之后不挂，模型想再问也调不到
    llm_tools = llm.bind_tools(base_tools)
    llm_ask = llm.bind_tools(tool_set) if hitl else llm_tools

    async def agent(state: MessagesState, config: RunnableConfig):
        messages = state["messages"]
        updates: list = []

        if len(messages) > SUMMARY_THRESHOLD:
            split = _summary_split(messages, KEEP_RECENT_TURNS)
            if split:
                old, messages = split
                digest = await llm.ainvoke([SystemMessage(SUMMARY_PROMPT), *old])
                summary = SystemMessage(f"此前对话摘要：{digest.content}")
                # 删掉旧消息、用一条摘要替代，再继续回答本轮
                updates = [RemoveMessage(id=m.id) for m in old] + [summary]
                messages = [summary, *messages]

        asked = _asked_before(messages)

        prompt = [SystemMessage(SYSTEM_PROMPT)]
        if store is not None:
            remembered = await load_memories(store, user_id_of(config))
            if remembered:
                prompt.append(SystemMessage(memory_prompt(remembered)))
        prompt.extend(messages)
        if _tool_rounds(messages) >= MAX_TOOL_ROUNDS:
            prompt.append(SystemMessage(NO_MORE_TOOLS_HINT))
            response = await llm.ainvoke(prompt)
        else:
            if asked:
                prompt.append(SystemMessage(ALREADY_ASKED_HINT))
            response = await (llm_tools if asked else llm_ask).ainvoke(prompt)

        return {"messages": [*updates, response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tool_set))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # 有 tool_calls → tools，没有 → END
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer or InMemorySaver(), store=store)