"""知识库（RAG）相关服务。"""
from __future__ import annotations

import re
import time
from typing import List

from sqlalchemy.orm import Session

from database import session_scope
from models import KnowledgeCollection, KnowledgeDocument, RAGQuery


class RagService:
    """提供知识库增删查等操作。"""

    def create_collection(self, name: str, description: str | None, user_id: int) -> KnowledgeCollection:
        with session_scope() as session:
            collection = KnowledgeCollection(
                name=name,
                description=description or "",
                document_count=0,
                collection_type="document",
                is_active=True,
                created_by=user_id,
            )
            session.add(collection)
            session.flush()
            session.refresh(collection)
            return collection

    def add_document_from_text(
        self,
        collection_id: int,
        filename: str,
        content: str,
        file_size: int,
        file_type: str,
        user_id: int,
    ) -> KnowledgeDocument:
        chunks = max(1, len(content) // 500)
        with session_scope() as session:
            collection = session.get(KnowledgeCollection, collection_id)
            if collection is None:
                raise ValueError("知识库不存在")
            document = KnowledgeDocument(
                collection_id=collection_id,
                filename=filename,
                file_path=f"/knowledge-docs/{collection_id}/{filename}",
                file_size=file_size,
                file_type=file_type,
                content=content[:10000],
                chunk_count=chunks,
                upload_status="completed",
                uploaded_by=user_id,
            )
            session.add(document)

            collection.document_count = (collection.document_count or 0) + 1
            session.add(collection)

            session.flush()
            session.refresh(document)
            return document

    def query(self, collection_id: int, query: str, user_id: int | None = None, limit: int = 5) -> dict:
        start = time.time()
        with session_scope() as session:
            documents: List[KnowledgeDocument] = (
                session.query(KnowledgeDocument)
                    .filter(KnowledgeDocument.collection_id == collection_id)
                    .all()
            )

        pattern = re.compile(re.escape(query), re.IGNORECASE)
        matches = []
        for doc in documents:
            content = doc.content or ""
            score = len(pattern.findall(content)) + (1 if pattern.search(doc.filename or "") else 0)
            if score > 0:
                matches.append({
                    "id": doc.id,
                    "filename": doc.filename,
                    "snippet": content[:500],
                    "score": score,
                })

        matches.sort(key=lambda item: item["score"], reverse=True)
        matches = matches[:limit]

        response_text = (
            f"基于集合 {collection_id} 找到 {len(matches)} 个相关文档。" if matches else "未找到相关文档"
        )

        elapsed_ms = int((time.time() - start) * 1000)
        with session_scope() as session:
            session.add(
                RAGQuery(
                    collection_id=collection_id,
                    query_text=query,
                    result_count=len(matches),
                    response_text=response_text,
                    user_id=user_id,
                    query_time_ms=elapsed_ms,
                )
            )

        return {
            "response": response_text,
            "documents": matches,
            "query_time": elapsed_ms,
        }


_rag_service: RagService | None = None


def get_rag_service() -> RagService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService()
    return _rag_service
