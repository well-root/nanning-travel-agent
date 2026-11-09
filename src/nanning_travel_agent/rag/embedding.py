import os
from dotenv import load_dotenv
from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from nanning_travel_agent.rag import config
# 指定位置不依赖当前的工作目录：rag -> 包 -> src -> 项目根
load_dotenv(Path(__file__).resolve().parents[3] / ".env")
@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，检查项目根目录 .env")
    return OpenAIEmbeddings(
        model=config.EMBED_MODEL,
        api_key=api_key,
        base_url=os.environ.get("DASHSCOPE_BASE_URL",
                                "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        dimensions=config.EMBED_DIM,
        chunk_size=config.EMBED_BATCH,
        check_embedding_ctx_length=False,
        max_retries=3,
        request_timeout=30,

    )
