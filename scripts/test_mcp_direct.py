"""MCP 直调验收：不经过 LangGraph，直接用 langchain-mcp-adapters 调自研的 nanning-metro server。

地位：CLAUDE.md 里的固定验证命令之一。任何改动 MCP 接入层之后都要先跑它。

运行：
    uv run python scripts/test_mcp_direct.py

验收三条（全部通过才退出码 0）：
  1. 工具列表里必须有 find_metro_route
  2. 朝阳广场 → 火车东站：1 号线，13 站，约 24 分钟（正常路径）
  3. 新民路 → 南宁剧场：在朝阳广场换乘 2 号线，约 10 分钟（含换乘路径）
  4. 故意传错站名（北京西站）：返回「站名不存在」文本，而不是抛异常（D4 建图要用到这个行为）

注意：地铁站名是「火车东站」；「南宁东站」是高铁站，不在数据里。
"""

from __future__ import annotations

import asyncio
import os
import re
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

# MCP server 的启动方式（从 ~/.claude.json 的 nanning-metro 条目抄来的）。
# 允许用环境变量覆盖，换机器/换虚拟环境时不用改代码。
METRO_PYTHON = os.environ.get(
    "NANNING_METRO_PYTHON",
    r"D:\agent\nanning-metro-mcp\.venv\Scripts\python.exe",
)
METRO_SERVER = os.environ.get(
    "NANNING_METRO_SERVER",
    r"D:\agent\nanning-metro-mcp\src\nanning_metro_mcp\server.py",
)

TOOL_NAME = "find_metro_route"


def build_client() -> MultiServerMCPClient:
    """构造指向 nanning-metro 的 MCP 客户端（只连接，不启动任何图）。"""
    return MultiServerMCPClient(
        {
            "nanning_metro": {
                "transport": "stdio",
                "command": METRO_PYTHON,
                "args": [METRO_SERVER],
                # Windows 下 stdio 传中文站名必须显式指定 UTF-8，否则乱码/报错
                "env": {"PYTHONIOENCODING": "utf-8"},
            }
        }
    )


def result_text(result: object) -> str:
    """把 MCP 工具返回值拍扁成纯文本。

    langchain-mcp-adapters 0.3.x 默认返回「内容块列表」：
        [{"type": "text", "text": "...", "id": "..."}]
    而不是字符串。D3/D4 里写工具逻辑时会反复用到这个转换。
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in result
        )
    return str(result)


def check(name: str, ok: bool, detail: str, problems: list[str]) -> None:
    """单条断言：打印结果，不通过则记入 problems。"""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" -> {detail}"))
    if not ok:
        problems.append(name)


async def main() -> int:
    problems: list[str] = []
    client = build_client()

    print("1) 加载工具…")
    tools: list[BaseTool] = await client.get_tools()
    names = [t.name for t in tools]
    print(f"   工具: {names}")
    check("工具列表含 find_metro_route", TOOL_NAME in names, f"实际={names}", problems)
    if TOOL_NAME not in names:
        return 1

    route_tool = next(t for t in tools if t.name == TOOL_NAME)

    print("\n2) 正常路径：朝阳广场 → 火车东站")
    r1 = result_text(await route_tool.ainvoke({"start": "朝阳广场", "end": "火车东站"}))
    print("   " + r1.replace("\n", "\n   "))
    check("走 1 号线", "1号线" in r1, r1, problems)
    check("终点是火车东站", "火车东站" in r1, r1, problems)
    check("13 站", re.search(r"13\s*站", r1) is not None, r1, problems)
    check("有总时长", re.search(r"\d+(?:\.\d+)?\s*分钟", r1) is not None, r1, problems)

    print("\n3) 换乘路径：新民路 → 南宁剧场")
    r2 = result_text(await route_tool.ainvoke({"start": "新民路", "end": "南宁剧场"}))
    print("   " + r2.replace("\n", "\n   "))
    check("含换乘说明", "换乘" in r2, r2, problems)
    check("换乘站是朝阳广场", "朝阳广场" in r2, r2, problems)
    check("走 2 号线", "2号线" in r2, r2, problems)
    check("有总时长", re.search(r"\d+(?:\.\d+)?\s*分钟", r2) is not None, r2, problems)

    print("\n4) 错误路径：不存在的站名（北京西站）")
    try:
        r3 = result_text(await route_tool.ainvoke({"start": "朝阳广场", "end": "北京西站"}))
        print("   " + r3.replace("\n", "\n   "))
        check("返回站名不存在文本", "站名不存在" in r3, r3, problems)
        check("没有把异常抛给调用方", True, "", problems)
    except Exception as exc:  # noqa: BLE001 - 这里就是要区分「抛异常」和「返回文本」
        print(f"   抛出异常: {type(exc).__name__}: {exc}")
        check("错误以文本返回而非抛异常", False, f"{type(exc).__name__}: {exc}", problems)

    print("\n" + "=" * 60)
    if problems:
        print(f"FAILED（{len(problems)} 项）: {problems}")
        return 1
    print("ALL PASSED：MCP 直调链路正常")
    return 0


if __name__ == "__main__":
    # Windows 控制台默认编码可能不是 UTF-8，中文站名会乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(asyncio.run(main()))
