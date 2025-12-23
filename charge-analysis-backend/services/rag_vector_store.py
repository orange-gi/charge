from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
from urllib.parse import urlparse
from typing import Any, Callable

import chromadb
from chromadb.api.models.Collection import Collection

from config import get_settings


@dataclass(frozen=True)
class VectorHit:
    id: str
    document: str
    metadata: dict[str, Any]
    distance: float | None


class ChromaVectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        # Windows 下嵌入式 PersistentClient 偶发 native 崩溃（Segmentation fault）。
        # 提供 HTTP 模式：使用独立 Chroma Server，把崩溃隔离到向量库进程。
        chroma_http_url = (getattr(settings, "chroma_http_url", "") or "").strip()
        if chroma_http_url:
            parsed = urlparse(chroma_http_url)
            if not parsed.hostname:
                raise ValueError(f"CHROMA_HTTP_URL 不合法：{chroma_http_url}")
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            ssl = parsed.scheme == "https"
            self._client = chromadb.HttpClient(host=host, port=int(port), ssl=ssl)
        else:
            self._client = chromadb.PersistentClient(path=str(settings.chroma_persist_directory))

    def get_or_create_collection(self, chroma_collection_id: str) -> Collection:
        if not chroma_collection_id or not isinstance(chroma_collection_id, str):
            raise ValueError("chroma_collection_id 不能为空")
        try:
            return self._client.get_collection(name=chroma_collection_id)
        except Exception:
            return self._client.create_collection(name=chroma_collection_id)

    def delete_by_document_id(self, chroma_collection_id: str, document_id: int) -> None:
        col = self.get_or_create_collection(chroma_collection_id)
        # where 过滤删除（避免残留 chunk）
        col.delete(where={"document_id": int(document_id)})

    def upsert(
        self,
        chroma_collection_id: str,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        col = self.get_or_create_collection(chroma_collection_id)
        # upsert 在 chromadb==0.4.18 可用
        col.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def get_by_where(
        self,
        chroma_collection_id: str,
        where: dict[str, Any],
        limit: int = 50,
    ) -> list[VectorHit]:
        col = self.get_or_create_collection(chroma_collection_id)
        # get 不返回 distance，因此 distance=None
        res = col.get(where=where, limit=limit, include=["documents", "metadatas"])
        ids = res.get("ids", []) or []
        docs = res.get("documents", []) or []
        metas = res.get("metadatas", []) or []
        hits: list[VectorHit] = []
        for _id, doc, meta in zip(ids, docs, metas):
            hits.append(VectorHit(id=_id, document=doc or "", metadata=meta or {}, distance=None))
        return hits

    def query(
        self,
        chroma_collection_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        col = self.get_or_create_collection(chroma_collection_id)
        res = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        hits: list[VectorHit] = []
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append(VectorHit(id=_id, document=doc or "", metadata=meta or {}, distance=float(dist) if dist is not None else None))
        return hits


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    return SentenceTransformer(settings.bge_model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return embed_texts_batched(texts)


def embed_texts_batched(
    texts: list[str],
    *,
    batch_size: int = 32,
    on_batch: Callable[[int, int, int], None] | None = None,
) -> list[list[float]]:
    """分批 embedding，便于上层输出进度日志。

    - on_batch(done, total, batch_elapsed_ms)
    """
    model = _get_embedder()
    total = len(texts)
    if total == 0:
        return []
    bs = int(batch_size) if batch_size and int(batch_size) > 0 else 32

    out: list[list[float]] = []
    done = 0
    for start in range(0, total, bs):
        batch = texts[start : start + bs]
        t0 = time.time()
        # normalize_embeddings=True 对检索更稳定，且便于距离解释
        vectors = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        out.extend([v.tolist() for v in vectors])
        done = min(total, start + len(batch))
        if on_batch:
            on_batch(done, total, int((time.time() - t0) * 1000))
    return out


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]

