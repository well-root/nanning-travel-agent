"""md →结构化切块 → 阿里云 embedding → Qdrant。"""



from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from nanning_travel_agent.rag import config
from nanning_travel_agent.rag.embedding import get_embeddings



class DocLoader:
    """读 data/docs/*.md，拆出 frontmatter 元数据和正文。"""

    def load_one(self, path: Path) -> dict:
        text = path.read_text(encoding="utf-8-sig")
        if not text.startswith("---"):
            raise ValueError(f"{path.name}:缺frontmatter")

        lines = text.splitlines()
        # 找出第二个 ---
        end = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end = i
                break
        if end is None:
            raise ValueError(f"{path.name}: frontmatter 没有闭合的 ---")

        # 解析 kv
        meta: dict[str, str] = {}
        for line in lines[1:end]:
            if not line.strip():
                continue
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key:
                meta[key] = value

        # 先校验再干活
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


class _Block(NamedTuple):
    """切块中间产物：kind 决定它能不能被二次切分。"""
    kind: str
    section: str    # 所属标题路径
    content: str


class MarkdownChunk:
    """按 Markdown 结构切块。"""

    def __init__(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_MAX,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "；", "，", " "],
        )

    @staticmethod
    def _section(heading_stack: list[tuple[int, str]]) -> str:
        #按照结构表头查找
        return " > ".join(text for _, text in heading_stack)

    def chunk_document(self, item: dict) -> list[Document]:
        meta, body = item["meta"], item["body"]
        lines = body.splitlines()
        heading_stack: list[tuple[int, str]] = []
        blocks: list[_Block] = []
        i = 0

        while i < len(lines):
            stripped = lines[i].strip()

            if not stripped:                        # 空行：跳过
                i += 1
                continue

            m = HEADING_RE.match(stripped)          #标题
            if m:
                level, text = len(m.group(1)), m.group(2).strip()
                # 同级或更高级标题出现时，把栈里更深的标题弹掉
                heading_stack = [(lv, t) for lv, t in heading_stack if lv < level]
                heading_stack.append((level, text))
                i += 1
                continue

            if stripped.startswith("|"):            # 表格
                start = i
                while i < len(lines) and lines[i].strip().startswith("|"):
                    i += 1
                blocks.append(_Block("table", self._section(heading_stack),
                                    "\n".join(lines[start:i]).strip()))
                continue

            if stripped.startswith("```"):          # 代码块
                start = i
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    i += 1
                i = min(i + 1, len(lines))
                blocks.append(_Block("code", self._section(heading_stack),
                                    "\n".join(lines[start:i]).strip()))
                continue

            start = i
            while i < len(lines):
                s = lines[i].strip()
                if not s or HEADING_RE.match(s) or s.startswith("|") or s.startswith("```"):
                    break
                i += 1
            blocks.append(_Block("para", self._section(heading_stack),
                                "\n".join(lines[start:i]).strip()))

        return self._to_documents(blocks, meta)

    def _to_documents(self, blocks: list[_Block], meta: dict) -> list[Document]:
        docs: list[Document] = []

        for block in blocks:
            content = block.content.strip()
            if len(content) < config.MIN_CHUNK:      # 纯标题、碎行：丢掉
                continue

            # 只有段落允许二次切分；表格/代码块整段保留
            if block.kind == "para" and len(content) > config.CHUNK_MAX:
                pieces = self.splitter.split_text(content)
            else:
                pieces = [content]

            for piece in pieces:
                piece = piece.strip()
                if len(piece) < config.MIN_CHUNK:
                    continue
                text = f"{block.section}\n\n{piece}" if block.section else piece
                docs.append(Document(
                    page_content=text,
                    # metadata 会被写进 Qdrant payload：只能放 str/int/bool，不能有 None
                    metadata={**meta, "section": block.section},
                ))

        return docs


#入库进入qdrant
class Indexer:

    def __init__(self) -> None:
        self.client = QdrantClient(url=config.QDRANT_URL)
        self.embeddings = get_embeddings()          # lru_cache，只建一次

    def ping(self) -> None:
        """先确认 Qdrant """
        try:
            self.client.get_collections()
        except Exception as exc:
            raise RuntimeError(
                f"连不上 Qdrant（{config.QDRANT_URL}）：先执行 docker start qdrant"
            ) from exc

    def _check_dim(self) -> None:
        """已存在的 collection 维度不符就抛错"""
        if not self.client.collection_exists(config.COLLECTION):
            return
        vectors = self.client.get_collection(config.COLLECTION).config.params.vectors
        size = getattr(vectors, "size", None)
        if size is not None and size != config.EMBED_DIM:
            raise ValueError(
                f"collection '{config.COLLECTION}' 维度是 {size}，"
                f"但当前 embedding 输出 {config.EMBED_DIM} 维——需要重建 collection"
            )

    def index(self, docs: list[Document], recreate: bool = True) -> int:
        if not docs:
            raise ValueError("没有可写入的块")
        self.ping()
        self._check_dim()
        QdrantVectorStore.from_documents(
            documents=docs,
            embedding=self.embeddings,
            url=config.QDRANT_URL,
            collection_name=config.COLLECTION,
            force_recreate=recreate,
        )
        return self.client.count(config.COLLECTION).count


def main() -> None:
    print(f"[1/4] 读取语料: {config.DOCS_DIR}")
    items = DocLoader().load_all()
    print(f"      共 {len(items)} 篇")

    print("[2/4] 切块")
    chunker = MarkdownChunk()
    chunks: list[Document] = []
    for item in items:
        doc_chunks = chunker.chunk_document(item)
        chunks.extend(doc_chunks)
        print(f"      {item['meta']['title']}: {len(doc_chunks)} 块")
    if not chunks:
        raise RuntimeError("切块结果为空，检查 MarkdownChunk")

    lengths = sorted(len(c.page_content) for c in chunks)
    print(
        f"      共 {len(chunks)} 块 | 长度 min/median/max = "
        f"{lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}"
    )
    for c in chunks[:2]:
        preview = c.page_content[:80].replace("\n", " / ")
        print(f"      示例: {preview}")

    print("[3/4] 写入 Qdrant")
    total = Indexer().index(chunks)
    print(f"      collection='{config.COLLECTION}' 维度={config.EMBED_DIM} points_count={total}")

    if total != len(chunks):
        raise RuntimeError(f"写入数量不一致：切块 {len(chunks)}，库里 {total}")
    print(f"[4/4] 完成：{config.QDRANT_URL}/collections/{config.COLLECTION}")


if __name__ == "__main__":
    main()
