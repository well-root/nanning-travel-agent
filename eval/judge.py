"""跑题库、规则判分、落 results。dual 与 mcp_only 两组共用同一套 15 条题。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
load_dotenv(ROOT / ".env")

from nanning_travel_agent.agent.graph import  build_graph
from nanning_travel_agent.mcp.client import load_metro_tools
from nanning_travel_agent.rag.tools import search_nanning_docs

GROUPS = ("dual", "mcp_only")
RECURSION_LIMIT = 20


def build_llm() -> ChatOpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 没读到，检查根目录 .env")
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model=os.environ.get("EVAL_MODEL", "deepseek-flash"),
        temperature=float(os.environ.get("EVAL_TEMPERATURE", "0.1")),
    )


def load_cases(limit: int | None, ids: set[str] | None = None) -> list[dict]:
    path = EVAL_DIR / "queries.jsonl"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cases = [json.loads(ln) for ln in lines]
    if ids:
        cases = [case for case in cases if case["id"] in ids]
    return cases[:limit] if limit else cases


def norm(text: str) -> str:
    """全半角归一 + 去掉所有空白，「2 元」和「2元」当成同一个。"""
    return "".join(unicodedata.normalize("NFKC", text).split())


def judge_case(case: dict, answer: str, called: list[str]) -> dict:
    body = norm(answer)
    expect = case["expect_tools"]
    tools_ok = (not called) if not expect else set(expect) <= set(called)
    missing = [k for k in case["must_contain"] if norm(k) not in body]
    forbidden = [k for k in case["must_not_contain"] if norm(k) in body]
    any_list = case.get("must_contain_any") or []
    missing_any = list(any_list) if any_list and not any(norm(k) in body for k in any_list) else []
    return {
        "tools_ok": tools_ok,
        "content_ok": not missing and not forbidden and not missing_any,
        "missing": missing,
        "missing_any": missing_any,
        "forbidden": forbidden,
        "passed": tools_ok and not missing and not forbidden and not missing_any,
    }


async def run_group(group: str, limit: int | None, ids: set[str] | None = None) -> list[dict]:
    llm = build_llm()
    tools = await load_metro_tools()
    if group == "dual":
        tools = [search_nanning_docs, *tools]

    cases = load_cases(limit, ids)
    names = [t.name for t in tools]
    print(f"组: {group}    模型: {llm.model_name}    temperature: {llm.temperature}")
    print(f"工具: {names}")
    print(f"题数: {len(cases)}\n")

    graph = build_graph(llm, tools)
    rows: list[dict] = []

    for idx, case in enumerate(cases, 1):
        config = {
            "configurable": {"thread_id": f"eval-{group}-{case['id']}-{time.time_ns()}"},
            "recursion_limit": RECURSION_LIMIT,
        }
        started = time.perf_counter()
        messages: list = []
        error = ""
        try:
            result = await graph.ainvoke(
                {"messages": [ HumanMessage(case["question"])]},
                config=config,
            )
            messages = result["messages"]
            answer = str(messages[-1].content)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            answer = ""

        elapsed = int((time.perf_counter() - started) * 1000)
        called = [tc["name"] for m in messages for tc in (getattr(m, "tool_calls", None) or [])]
        verdict = judge_case(case, answer, called)

        mark = "PASS" if verdict["passed"] else "FAIL"
        seen = ",".join(dict.fromkeys(called)) or "-"
        print(f"[{idx:2d}/{len(cases)}] {case['id']:<3} {mark} {elapsed:>6}ms  tools={seen}")
        if not verdict["passed"]:
            if error:
                print(f"        error: {error}")
            if not verdict["tools_ok"]:
                print(f"        工具: 期望 {case['expect_tools']}，实际 {called}")
            if verdict["missing"]:
                print(f"        缺关键词: {verdict['missing']}")
            if verdict["missing_any"]:
                print(f"        缺任一关键词: {verdict['missing_any']}")
            if verdict["forbidden"]:
                print(f"        出现禁用词: {verdict['forbidden']}")

        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "group": group,
                "question": case["question"],
                "model": llm.model_name,
                "temperature": llm.temperature,
                "called_tools": called,
                "expect_tools": case["expect_tools"],
                "tools_ok": verdict["tools_ok"],
                "content_ok": verdict["content_ok"],
                "missing": verdict["missing"],
                "missing_any": verdict["missing_any"],
                "forbidden": verdict["forbidden"],
                "passed": verdict["passed"],
                "ms": elapsed,
                "messages": len(messages),
                "answer": answer,
                "error": error,
            }
        )
    return rows


def summarize(rows: list[dict]) -> None:
    total = len(rows)
    if not total:
        return
    passed = sum(r["passed"] for r in rows)
    tools_ok = sum(r["tools_ok"] for r in rows)
    avg_ms = sum(r["ms"] for r in rows) // total
    print(f"\n通过 {passed}/{total} ({passed / total:.0%})   工具选择 {tools_ok}/{total}   平均 {avg_ms}ms")

    by_cat: dict[str, list[int]] = {}
    for row in rows:
        stat = by_cat.setdefault(row["category"], [0, 0])
        stat[0] += 1
        stat[1] += int(row["passed"])
    for cat, (n, ok) in by_cat.items():
        print(f"  {cat:<7} {ok}/{n}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="跑 eval/queries.jsonl 并判分")
    parser.add_argument("--group", choices=GROUPS, default="dual", help="dual=双工具，mcp_only=只有换乘工具")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条，调试用")
    parser.add_argument("--ids", default=None, help="只跑指定 id，逗号分隔，如 e1,e2")
    args = parser.parse_args()

    ids = {i.strip() for i in args.ids.split(",") if i.strip()} if args.ids else None
    rows = await run_group(args.group, args.limit, ids)
    summarize(rows)

    if rows:
        out = EVAL_DIR / "results" / f"{datetime.now():%Y%m%d_%H%M%S}_{args.group}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        print(f"结果写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
