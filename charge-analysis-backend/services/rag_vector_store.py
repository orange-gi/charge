from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from config import get_settings

logger = logging.getLogger(__name__)


class VectorStoreUnavailableError(RuntimeError):
    """向量库不可用（例如依赖冲突导致 chromadb 无法 import）。"""


@dataclass(frozen=True)
class VectorHit:
    id: str
    document: str
    metadata: dict[str, Any]
    distance: float | None


class ChromaVectorStore:
    def __init__(self) -> None:
        # 延迟 import，避免在应用启动阶段因为依赖冲突直接崩溃。
        # 典型冲突：chromadb==0.4.x 在 NumPy 2.x 下会触发 `np.float_ was removed`。
        try:
            # ---- NumPy 2.x 兼容补丁（不改依赖版本）----
            # chromadb==0.4.x 的部分代码仍引用 np.float_（在 NumPy 2.0 被移除）。
            # 这里在导入 chromadb 前补回别名，避免 import 阶段直接崩溃。
            import numpy as np  # type: ignore

            if not hasattr(np, "float_"):  # NumPy 2.x
                # 将 float_ 映射到 float64，满足旧依赖对名称的引用
                setattr(np, "float_", np.float64)
            # 某些旧代码可能引用 np.uint（确保存在）
            if not hasattr(np, "uint"):
                setattr(np, "uint", np.uint64)

            import chromadb  # type: ignore
        except Exception as e:
            msg = (
                "Chroma 向量库不可用：导入 chromadb 失败。"
                "如果你看到 `np.float_ was removed`，说明当前 chromadb 版本与 NumPy 2.x 不兼容。"
                "可选方案：升级 chromadb；或保持旧版本并在导入前做 numpy 兼容补丁；或将 RAG 功能放到独立环境。"
            )
            logger.error("%s 原始错误: %s", msg, e)
            raise VectorStoreUnavailableError(msg) from e

        settings = get_settings()
        self._chromadb = chromadb
        self._client = chromadb.PersistentClient(path=str(settings.chroma_persist_directory))

    def get_or_create_collection(self, chroma_collection_id: str):
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

