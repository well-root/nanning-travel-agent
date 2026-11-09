#解析出来的向量然后存起来
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from nanning_travel_agent.rag import config
from nanning_travel_agent.rag.embedding import get_embeddings
class DocLoader:
    def load_one(self,path:Path)->dict:
        text=path.read_text(encoding="utf-8",)
        if not text.startswith("---"):
            raise ValueError(f"{path.name}:缺frontmatter")

        lines=text.splitlines()
        #找出第二个表头
        end=None
        for i in range(1,len(lines)):
            if lines[i].strip()=="---":
                end=i
                break
        if end is None:
            raise ValueError(f"{path.name}: frontmatter 没有闭合的 ---")
        #解析kv结果
        meta:dict[str,str]={}
        for line in lines[1:end]:
            if not line.strip():
                continue
            key,_,value=line.partition(":")
            key,value=key.strip(),value.strip()
            if key:
                meta[key]=value
        #先检验在干活
        missing = [k for k in config.REQUIRED_META if not meta.get(k)]
        if missing:
            raise ValueError(f"{path.name}: frontmatter 缺字段 {missing}")

        body = "\n".join(lines[end + 1:]).strip()
        return {"meta": meta, "body": body}

    def load_all(self) -> list[dict]:
        paths = sorted(config.DOCS_DIR.glob("*.md"))
        if not paths:
            raise FileNotFoundError(f"没有找到语料文件: {config.DOCS_DIR}")
        return [self.load_one(p) for p in paths]


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

class MarkdownChunk:
    def __init__(self):
            self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_MAX,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "；", "，", " "],
        )

    def chunk_document(self, item: dict) -> list[Document]:
        meta, body = item["meta"], item["body"]
        lines = body.splitlines()
        heading_stack: list[tuple[int, str]] = []  # [(level, text)]
        blocks: list[tuple[str, str]] = []  # [(section, content)]
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:  # 空行：跳过
                i += 1
                continue

            m = HEADING_RE.match(stripped)  # ① 标题：更新栈，不单独成块
            if m:
                level, text = len(m.group(1)), m.group(2).strip()
                heading_stack = [(lv, t) for lv, t in heading_stack if lv < level]
                heading_stack.append((level, text))
                i += 1
                continue






