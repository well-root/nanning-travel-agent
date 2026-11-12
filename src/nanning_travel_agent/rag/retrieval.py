"""混合检索"""

from __future__ import annotations

import os

import httpx
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from nanning_travel_agent.rag import config as cfg
from nanning_travel_agent.rag.embedding import get_embeddings

RRF_K = 60
DENSE_K = 8
SPARSE_K = 8

RERANK_MODEL = "gte-rerank-v2"
RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
RERANK_CANDIDATES = 12


def tokenize(text: str) -> list[str]:
    compact = "".join(text.split()).lower()
    if len(compact) < 2:
        return [compact] if compact else []
    return [compact[i : i + 2] for i in range(len(compact) - 1)]


class HybridRetriever:
    def __init__(
        self,
        mode: str | None = None,
        k: int = cfg.RETRIEVAL_K,
        rerank: bool | None = None,
    ):
        self.mode = (mode or os.environ.get("RETRIEVAL_MODE", "hybrid")).lower()
        self.rerank = os.environ.get("RERANK", "1") == "1" if rerank is None else rerank
        self.k = k
        self._store: QdrantVectorStore | None = None
        self._bm25: BM25Okapi | None = None
        self._corpus: list[Document] = []

    @property
    def store(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore.from_existing_collection(
                embedding=get_embeddings(),
                collection_name=cfg.COLLECTION,
                url=cfg.QDRANT_URL,
            )
        return self._store

    @property
    def bm25(self) -> BM25Okapi:
        if self._bm25 is None:
            self._corpus = self._load_corpus()
            self._bm25 = BM25Okapi([tokenize(d.page_content) for d in self._corpus])
        return self._bm25

    @staticmethod
    def _load_corpus() -> list[Document]:
        client = QdrantClient(url=cfg.QDRANT_URL)
        points, _ = client.scroll(
            collection_name=cfg.COLLECTION,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        docs = [
            Document(
                page_content=p.payload.get("page_content", ""),
                metadata=p.payload.get("metadata", {}),
            )
            for p in points
        ]
        return [d for d in docs if d.page_content.strip()]

    def _dense(self, query: str, k: int) -> list[tuple[Document, float]]:
        hits = self.store.similarity_search_with_score(
            query, k=k, score_threshold=cfg.SCORE_THRESHOLD
        )
        return [(doc, float(score)) for doc, score in hits]

    def _sparse(self, query: str, k: int) -> list[tuple[Document, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._corpus[i], float(scores[i])) for i in order if scores[i] > 0]

    def _rerank(self, query: str, candidates: list[Document]) -> list[tuple[Document, float]]:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 没读到")

        payload = {
            "model": RERANK_MODEL,
            "input": {"query": query, "documents": [doc.page_content for doc in candidates]},
            "parameters": {"top_n": self.k, "return_documents": False},
        }
        response = httpx.post(
            RERANK_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        results = (response.json().get("output") or {}).get("results") or []
        return [
            (candidates[item["index"]], float(item["relevance_score"]))
            for item in results
        ][: self.k]

    def search_with_score(self, query: str) -> list[tuple[Document, float]]:
        if self.mode == "vector":
            return self._dense(query, self.k)

        fused: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for rank, (doc, _) in enumerate(self._dense(query, DENSE_K)):
            key = doc.page_content
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            docs.setdefault(key, doc)
        for rank, (doc, _) in enumerate(self._sparse(query, SPARSE_K)):
            key = doc.page_content
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            docs.setdefault(key, doc)

        ordered = sorted(fused, key=lambda key: fused[key], reverse=True)
        candidates = [docs[key] for key in ordered[:RERANK_CANDIDATES]]

        if self.rerank:
            try:
                return self._rerank(query, candidates)
            except Exception:
                pass  # 重排失败就退回 RRF 顺序，不打断问答链路

        return [(doc, fused[doc.page_content]) for doc in candidates[: self.k]]

    def search(self, query: str) -> list[Document]:
        return [doc for doc, _ in self.search_with_score(query)]


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever

