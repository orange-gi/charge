"""知识库（RAG）API。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from core.dependencies import get_current_user
from models import User
from schemas import RagCollectionCreate, RagQueryRequest, RagQueryResponse
from services.rag_service import get_rag_service

router = APIRouter(prefix="/api/rag", tags=["rag"])
rag_service = get_rag_service()


@router.post("/collections")
def create_collection(
    payload: RagCollectionCreate,
    user: User = Depends(get_current_user),
) -> dict:
    collection = rag_service.create_collection(payload.name, payload.description, user.id)
    return {
        "id": collection.id,
        "name": collection.name,
        "description": collection.description,
        "document_count": collection.document_count,
    }


@router.post("/collections/{collection_id}/documents")
async def upload_document(
    collection_id: int,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件为空")

    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = raw_bytes.decode("utf-8", errors="ignore")

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
    return {"document_id": document.id}


@router.post("/query", response_model=RagQueryResponse)
def query_knowledge(
    payload: RagQueryRequest,
    user: User = Depends(get_current_user),
) -> RagQueryResponse:
    result = rag_service.query(payload.collection_id, payload.query, user.id)
    return RagQueryResponse(**result)
