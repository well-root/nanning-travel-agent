"""双工具路由图：模型自己决定查文档（RAG）还是查换乘（MCP）。"""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

SYSTEM_PROMPT = """你是南宁地铁出行助手。
票价、优惠政策、线路站点、运营时间、车票使用这类问题，用 search_nanning_docs 查文档。
两个车站之间的换乘路线，用 find_metro_route 查。
工具返回 NO_RELEVANT_DOCS 或「站名不存在」时，直说查不到，不要编。"""


def build_graph(llm: BaseChatModel, tools: list[BaseTool]):
    llm_with_tools = llm.bind_tools(tools)

    async def agent(state: MessagesState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # 有 tool_calls → tools，没有 → END
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=InMemorySaver())

