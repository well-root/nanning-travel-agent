from __future__ import annotations

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
from pathlib import Path
import os
from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage

from nanning_travel_agent.rag.tools import search_nanning_docs

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

SYSTEM_PROMPT = """你是南宁地铁出行助手。
票价、优惠政策、线路站点、运营时间、车票使用这类问题，先调 search_nanning_docs 查文档，按文档回答，不要自己编。
工具返回 NO_RELEVANT_DOCS 就直说文档里没有，别猜。"""

MAX_STEP=10

def build_llm()->ChatOpenAI:
    api_key=os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY not set")
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
        temperature=0.1,
    )


def handle_turn(llm_with_tools,tools_by_name:dict,messages:list)->str:
    """完整的对话 可以调用工具"""
    for _ in range(MAX_STEP):
        ai=llm_with_tools.invoke(messages)
        messages.append(ai)

        if not ai.tool_calls:
            return ai.content

        for tc in ai.tool_calls:
            fn = tools_by_name.get(tc["name"])
            try:
                result = fn.invoke(tc["args"]) if fn else f"未知工具: {tc['name']}"
            except Exception as exc:
                result = f"工具执行失败: {exc}"
            if not isinstance(result, str):
                result = str(result)

            print(f"  [工具] {tc['name']}({tc['args']}) → {result[:70]}…")
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    return "（工具调用超过上限，已强制结束）"


def main() -> None:
    llm = build_llm()
    tools = [search_nanning_docs]
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages = [SystemMessage(SYSTEM_PROMPT)]
    print("南宁出行助手（输入 exit 退出）")

    while True:
        question = input("\n你: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "/exit"}:
            break

        messages.append(HumanMessage(question))
        answer = handle_turn(llm_with_tools, tools_by_name, messages)
        print(f"助手: {answer}")


if __name__ == "__main__":
    main()

