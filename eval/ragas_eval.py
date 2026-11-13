from __future__ import annotations

import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path

try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    _vertexai = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover - 仅占位
        pass

    _vertexai.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

from nanning_travel_agent.rag.embedding import get_embeddings

RESULTS_DIR = ROOT / "eval" / "results"


def main() -> None:
    latest = sorted(RESULTS_DIR.glob("*_dual.jsonl"))[-1]
    raw = [json.loads(line) for line in latest.read_text(encoding="utf-8").splitlines() if line.strip()]

    # 有文档片段或有工具输出的题都参与评分：纯换乘题的路线事实来自 MCP 工具，
    # 只用 rag_contexts 会让混合题/换乘题的 faithfulness 被低估
    rows = [
        {
            "user_input": item["question"],
            "response": item["answer"],
            "retrieved_contexts": item.get("rag_contexts", []) + item.get("tool_contexts", []),
        }
        for item in raw
        if item.get("answer") and (item.get("rag_contexts") or item.get("tool_contexts"))
    ]
    print(f"输入: {latest.name}    参与评分: {len(rows)} 条")

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            model="deepseek-flash",
            temperature=0,
        ),
        # DeepSeek 只支持 n=1，ragas 默认会给 ChatOpenAI 传 n>1，必须关掉
        bypass_n=True,
    )
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    result = evaluate(
        EvaluationDataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )
    print(result)

    frame = result.to_pandas()
    out = RESULTS_DIR / f"ragas_{datetime.now():%Y%m%d_%H%M%S}.json"
    frame.to_json(out, orient="records", force_ascii=False, indent=2)

    print("\n均值:")
    for name in ("faithfulness", "answer_relevancy"):
        if name in frame.columns:
            print(f"  {name}: {frame[name].mean():.3f}")
    print(f"明细: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()