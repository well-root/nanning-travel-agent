from langchain_core.tools import tool

from nanning_travel_agent.rag.retrieval import get_retriever


@tool
def search_nanning_docs(query:str)->str:
    """查询南宁地铁的文档知识：票价规则、优惠政策、线路站点、运营时间、车票使用。

    不负责两个车站之间的换乘路线，那要用 find_metro_route。

    Args:
        query: 带具体关键词的检索语句，比如"起步价"、"学生卡优惠"、"末班车"。

    Returns:
        若干条 "[文档标题] 正文"，用 --- 分隔。查不到时返回 NO_RELEVANT_DOCS。
    """
    try:
        docs = get_retriever().search(query)
    except Exception as exc:
        return f"RETRIEVAL_ERROR: {exc}"

    if not docs:
        return "NO_RELEVANT_DOCS"

    return "\n\n---\n\n".join(
        f"[{d.metadata.get('title', '未知文档')}] {d.page_content}" for d in docs
    )



