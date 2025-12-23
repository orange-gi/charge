"""知识库（RAG）API。"""
from __future__ import annotations

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile, status

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

from services.rag_service import DocumentAlreadyExistsError, get_rag_service

router = APIRouter(prefix="/api/rag", tags=["rag"])
rag_service = get_rag_service()


@router.post("/collections")
def create_collection(
    payload: RagCollectionCreate = Body(
        ...,
        example={"name": "充电协议FAQ", "description": "收录常见的国标充电问题解答"},
    ),
    user: User = Depends(get_current_user),
) -> RagCollectionRead:
    try:
        collection = rag_service.create_collection(payload.name, payload.description, user.id)
        return RagCollectionRead.model_validate(collection)
    except ValueError as exc:
        # 例如 chroma_collection_id 为空/非法
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"创建知识库失败：{exc}") from exc


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
    overwrite: bool = Form(False),
    user: User = Depends(get_current_user),
) -> KnowledgeDocumentRead:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件为空")
    filename = file.filename or "upload.xlsx"
    lower = filename.lower()
    if not (lower.endswith(".xlsx") or lower.endswith(".xlsm")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="当前仅支持上传 Excel（.xlsx/.xlsm）")

    try:
        document, _report = rag_service.add_excel_document(
            collection_id=collection_id,
            filename=filename,
            file_size=len(raw_bytes),
            file_type=file.content_type or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            user_id=user.id,
            raw_bytes=raw_bytes,
            overwrite=overwrite,
        )
    except DocumentAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        # collection 不存在 -> 404；其它解析错误 -> 400
        msg = str(exc)
        if "知识库不存在" in msg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg) from exc
    return KnowledgeDocumentRead.model_validate(document)


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
    payload: RagQueryRequest = Body(
        ...,
        example={"collection_id": 12, "query": "直流桩电压波动超标如何处理？"},
    ),
    user: User = Depends(get_current_user),
) -> RagQueryResponse:
    result = rag_service.query(
        payload.collection_id,
        payload.query,
        user_id=user.id,
        limit=getattr(payload, "top_k", 5) or 5,
        show_retrieval=getattr(payload, "show_retrieval", True),
    )
    return RagQueryResponse(**result)
