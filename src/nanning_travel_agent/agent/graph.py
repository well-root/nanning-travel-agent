"""双工具路由图：模型自己决定查文档（RAG）还是查换乘（MCP），工具调用过多时强制收口。"""

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage,RemoveMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


SYSTEM_PROMPT = """你是南宁地铁出行助手。
票价、优惠政策、线路站点、运营时间、车票使用这类问题，用 search_nanning_docs 查文档。
两个车站之间的换乘路线，用 find_metro_route 查。
工具返回 NO_RELEVANT_DOCS 或「站名不存在」时，直说查不到，不要编。"""

MAX_TOOL_ROUNDS = int(os.environ.get("MAX_TOOL_ROUNDS", "6"))
NO_MORE_TOOLS_HINT = """工具调用次数已到上限。现在只能用已经拿到的信息回答：
不要再说“我去查一下”，没查到的部分直接说明查不到，然后给出已知的部分。"""

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


def _summary_split(messages: list, keep_recent: int):
    """按照每一轮来切分内容"""
    humans = [i for i, m in enumerate(messages) if m.type == "human"]
    if len(humans) <= keep_recent:
        return None
    cut = humans[-keep_recent]
    return messages[:cut], messages[cut:]

def build_graph(llm: BaseChatModel, tools: list[BaseTool],checkpointer: BaseCheckpointSaver|None=None):
    llm_with_tools = llm.bind_tools(tools)

    async def agent(state: MessagesState):
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

        prompt = [SystemMessage(SYSTEM_PROMPT), *messages]
        if _tool_rounds(messages) >= MAX_TOOL_ROUNDS:
            prompt.append(SystemMessage(NO_MORE_TOOLS_HINT))
            response = await llm.ainvoke(prompt)
        else:
            response = await llm_with_tools.ainvoke(prompt)

        return {"messages": [*updates, response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # 有 tool_calls → tools，没有 → END
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer or InMemorySaver())



