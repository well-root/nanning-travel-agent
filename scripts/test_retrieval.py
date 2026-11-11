"""ai测试的召回脚本不经过模型，直接测 Qdrant 召回。"""
from langchain_qdrant import QdrantVectorStore
from nanning_travel_agent.rag import config as cfg
from nanning_travel_agent.rag.embedding import get_embeddings

PASS_SCORE = 0.60

CASES = [
    ("南宁地铁起步价多少，能坐多远", "南宁地铁票价计价规则"),
    ("南宁市民卡坐地铁打几折", "南宁地铁票种与优惠政策"),
    ("宁仔计次票在哪些车站能买到", "南宁地铁计次票"),
    ("南宁地铁周五、周六运营到几点", "南宁地铁运营服务时间"),
    ("坐完公交 90 分钟内换乘地铁有优惠吗", "南宁地铁票种与优惠政策"),
]

def main() -> None:
    store = QdrantVectorStore.from_existing_collection(
        url=cfg.QDRANT_URL, collection_name=cfg.COLLECTION, embedding=get_embeddings()
    )

    bad = 0
    for q, want in CASES:
        hits = store.similarity_search_with_score(q, k=cfg.RETRIEVAL_K)
        if not hits:
            print(f"FAIL 无结果 | {q}")
            bad += 1
            continue
        doc, score = hits[0]
        title = doc.metadata.get("title", "")
        ok = title == want and score >= PASS_SCORE
        print(f"{'PASS' if ok else 'FAIL'} {score:.3f} {title} (期望 {want}) | {q}")
        bad += 0 if ok else 1
    print(f"{len(CASES) - bad}/{len(CASES)}")
    raise SystemExit(1 if bad else 0)

if __name__ == "__main__":
    main()
