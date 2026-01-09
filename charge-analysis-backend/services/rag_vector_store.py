from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

# 在导入 chromadb 之前禁用遥测
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("DO_NOT_TRACK", "1")

import chromadb
from chromadb.api.models.Collection import Collection

from config import get_settings

logger = logging.getLogger(__name__)

# 抑制 ChromaDB 遥测日志
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


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

    def _looks_like_windows_path(s: str) -> bool:
        raw = (s or "").strip()
        # 典型形式：D:\xxx 或 D:/xxx
        return len(raw) >= 3 and raw[1] == ":" and (raw[2] == "\\" or raw[2] == "/")

    def _has_model_weights(p: Path) -> bool:
        # HuggingFace Transformers 常见权重文件（根目录）
        root_candidates = [
            p / "pytorch_model.bin",
            p / "model.safetensors",
            p / "tf_model.h5",
            p / "model.ckpt.index",
            p / "flax_model.msgpack",
        ]
        if any(c.exists() for c in root_candidates):
            return True
        # sentence-transformers 常见结构：0_Transformer/xxx
        st_candidates = [
            p / "0_Transformer" / "pytorch_model.bin",
            p / "0_Transformer" / "model.safetensors",
            p / "0_Transformer" / "tf_model.h5",
        ]
        if any(c.exists() for c in st_candidates):
            return True
        return False

    # 1) 若显式配置了 BGE_MODEL_PATH，则严格按该路径加载（并校验权重存在）
    env_bge_path = (os.environ.get("BGE_MODEL_PATH") or "").strip()
    if env_bge_path:
        p = Path(env_bge_path).expanduser()
        if not p.exists():
            raise RuntimeError(f"BGE_MODEL_PATH 指向的目录不存在：{p}")
        if not _has_model_weights(p):
            raise RuntimeError(
                f"BGE_MODEL_PATH 指向的目录缺少模型权重文件：{p}。"
                f"需要包含如 pytorch_model.bin / model.safetensors（或 sentence-transformers 的 0_Transformer/* 权重）。"
            )
        model_path = str(p.resolve())
        logger.info(f"使用本地 BGE 模型路径(BGE_MODEL_PATH): {model_path}")
        return SentenceTransformer(model_path)

    # 2) 未显式配置 BGE_MODEL_PATH：仅当默认路径“真的像一个可用模型目录”时才使用
    if settings.bge_model_path and settings.bge_model_path.exists():
        p = Path(settings.bge_model_path).expanduser()
        if _has_model_weights(p):
            model_path = str(p.resolve())
            logger.info(f"使用本地 BGE 模型路径(默认): {model_path}")
            return SentenceTransformer(model_path)
        logger.warning(
            f"检测到默认 BGE_MODEL_PATH 目录存在但不包含权重文件，将忽略并回退到 BGE_MODEL_NAME。path={p}"
        )

    # 3) 使用 BGE_MODEL_NAME（可为 HF 仓库名或本地路径）
    model_name_or_path = str(settings.bge_model_name or "").strip()
    if os.name != "nt" and _looks_like_windows_path(model_name_or_path):
        logger.warning(
            "当前运行环境为非 Windows，但 BGE_MODEL_NAME 看起来是 Windows 路径："
            f"{model_name_or_path}。若你在 Docker/Linux 里运行，需要把模型目录挂载到容器内，并用 BGE_MODEL_PATH 指向容器内路径。"
        )
    logger.info(f"使用 BGE 模型名称/路径(BGE_MODEL_NAME): {model_name_or_path}")
    return SentenceTransformer(model_name_or_path)


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

