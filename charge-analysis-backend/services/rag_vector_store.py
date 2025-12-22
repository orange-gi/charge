from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

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
        self._client = chromadb.PersistentClient(path=str(settings.chroma_persist_directory))

    def get_or_create_collection(self, chroma_collection_id: str) -> Collection:
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
    model = _get_embedder()
    # normalize_embeddings=True 对检索更稳定，且便于距离解释
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]

