"""检索验收：不经过模型，直接测召回。vector / hybrid / hybrid+rerank 三组对比。"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from nanning_travel_agent.rag.retrieval import HybridRetriever

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CASES = [
    ("南宁地铁起步价多少，能坐多远", "南宁地铁票价计价规则"),
    ("南宁市民卡坐地铁打几折", "南宁地铁票种与优惠政策"),
    ("宁仔计次票在哪些车站能买到", "南宁地铁计次票"),
    ("南宁地铁周五、周六运营到几点", "南宁地铁运营服务时间"),
    ("坐完公交 90 分钟内换乘地铁有优惠吗", "南宁地铁票种与优惠政策"),
]

CONFIGS = [
    ("vector", "vector", False),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=[label for label, _, _ in CONFIGS], default=None)
    args = parser.parse_args()

    failed = 0
    for label, mode, use_rerank in CONFIGS:
        if args.only and args.only != label:
            continue
        retriever = HybridRetriever(mode=mode, rerank=use_rerank)
        print(f"\n=== {label} ===")
        bad = 0
        for query, want in CASES:
            hits = retriever.search_with_score(query)
            if not hits:
                print(f"FAIL 无结果 | {query}")
                bad += 1
                continue
            doc, score = hits[0]
            title = doc.metadata.get("title", "")
            ok = title == want
            print(f"{'PASS' if ok else 'FAIL'} {score:.3f} {title} (期望 {want}) | {query}")
            bad += 0 if ok else 1
        print(f"{len(CASES) - bad}/{len(CASES)}")
        failed += bad

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()