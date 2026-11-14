"""信息不全时让模型主动反问：把 interrupt 包成一个工具交给模型自己决定用不用。"""

from langchain_core.tools import tool
from langgraph.types import interrupt


@tool
def ask_user(question: str) -> str:
    """出行信息不全时向用户提问，一次只问一句。用户回答后继续原来的任务。"""
    return str(interrupt({"question": question}))