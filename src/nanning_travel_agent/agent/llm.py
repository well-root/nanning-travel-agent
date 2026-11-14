"""统一的模型构造入口。

"""

import os

from langchain_openai import ChatOpenAI

NO_THINKING = {"thinking": {"type": "disabled"}}


def build_llm(model: str | None = None, temperature: float = 0.1) -> ChatOpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 没读到，检查根目录 .env")
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model=model or "deepseek-flash",
        temperature=temperature,
        extra_body=NO_THINKING,
    )
