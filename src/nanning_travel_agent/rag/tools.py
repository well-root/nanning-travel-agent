from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore

from nanning_travel_agent.rag import config
from nanning_travel_agent.rag.embedding import get_embeddings

_store:QdrantVectorStore|None=None

def _get_store() -> QdrantVectorStore:
    global _store
    if _store is None:
        _store = QdrantVectorStore.from_existing_collection(
            embedding=get_embeddings(),
            collection_name=config.COLLECTION,
            url=config.QDRANT_URL,
        )
    return _store


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
        docs=_get_store().similarity_search(
            query=query,
            k=config.RETRIEVAL_K,
            score_threshold=config.SCORE_THRESHOLD,
        )
    except Exception as exc:
        return f"RETRIEVAL_ERROR: {exc}"

    if not docs:
        return "NO_RELEVANT_DOCS"

    return "\n\n---\n\n".join(
        f"[{d.metadata.get('title', '未知文档')}] {d.page_content}" for d in docs
    )



