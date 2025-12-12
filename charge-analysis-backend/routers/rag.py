"""知识库（RAG）API。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from io import BytesIO

from core.dependencies import get_current_user
from models import User, UserRole
from schemas import (
    KnowledgeDocumentRead,
    RagCollectionCreate,
    RagCollectionRead,
    RagQueryRecord,
    RagQueryRequest,
    RagQueryResponse,
)
from PyPDF2 import PdfReader

from services.rag_service import get_rag_service

router = APIRouter(prefix="/api/rag", tags=["rag"])
rag_service = get_rag_service()


@router.post("/collections")
def create_collection(
    payload: RagCollectionCreate,
    user: User = Depends(get_current_user),
) -> RagCollectionRead:
    collection = rag_service.create_collection(payload.name, payload.description, user.id)
    return RagCollectionRead.model_validate(collection)


@router.get("/collections", response_model=list[RagCollectionRead])
def list_collections(user: User = Depends(get_current_user)) -> list[RagCollectionRead]:
    records = rag_service.list_collections(user.id)
    return [RagCollectionRead.model_validate(item) for item in records]


@router.get("/collections/{collection_id}", response_model=RagCollectionRead)
def get_collection(collection_id: int, user: User = Depends(get_current_user)) -> RagCollectionRead:
    collection = rag_service.get_collection(collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    if collection.created_by not in {None, user.id} and user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问该知识库")
    return RagCollectionRead.model_validate(collection)


@router.post("/collections/{collection_id}/documents")
async def upload_document(
    collection_id: int,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
) -> KnowledgeDocumentRead:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件为空")
    try:
        content = _extract_text_from_upload(file, raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        document = rag_service.add_document_from_text(
            collection_id=collection_id,
            filename=file.filename,
            content=content,
            file_size=len(raw_bytes),
            file_type=file.content_type or "txt",
            user_id=user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return KnowledgeDocumentRead.model_validate(document)


def _extract_text_from_upload(file: UploadFile, raw_bytes: bytes) -> str:
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()

    if filename.endswith(".pdf") or content_type == "application/pdf":
        try:
            reader = PdfReader(BytesIO(raw_bytes))
        except Exception as exc:  # PyPDF2 内部会抛出多种异常
            raise ValueError("无法解析 PDF 文件") from exc
        pages_text: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                pages_text.append(text)
        if not pages_text:
            raise ValueError("PDF 中未提取到可用文本")
        return "\n".join(pages_text)

    if b"\x00" in raw_bytes:
        raise ValueError("文件包含不可处理的二进制内容，请提供文本格式")

    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("utf-8", errors="ignore")


@router.get("/collections/{collection_id}/documents", response_model=list[KnowledgeDocumentRead])
def list_documents(collection_id: int, user: User = Depends(get_current_user)) -> list[KnowledgeDocumentRead]:
    collection = rag_service.get_collection(collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    if collection.created_by not in {None, user.id} and user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问该知识库")
    records = rag_service.list_documents(collection_id)
    return [KnowledgeDocumentRead.model_validate(doc) for doc in records]


@router.get("/collections/{collection_id}/queries", response_model=list[RagQueryRecord])
def list_queries(
    collection_id: int,
    limit: int = 20,
    user: User = Depends(get_current_user),
) -> list[RagQueryRecord]:
    collection = rag_service.get_collection(collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    if collection.created_by not in {None, user.id} and user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问该知识库")
    records = rag_service.list_queries(collection_id, limit=limit)
    return [RagQueryRecord.model_validate(item) for item in records]


@router.post("/query", response_model=RagQueryResponse)
def query_knowledge(
    payload: RagQueryRequest,
    user: User = Depends(get_current_user),
) -> RagQueryResponse:
    result = rag_service.query(payload.collection_id, payload.query, user.id)
    return RagQueryResponse(**result)
