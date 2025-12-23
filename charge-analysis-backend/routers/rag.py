"""知识库（RAG）API。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Body, Depends, File, Form, HTTPException, UploadFile, status

from core.dependencies import get_current_user
from models import User, UserRole
from schemas import (
    KnowledgeDocumentRead,
    RagCollectionCreate,
    RagCollectionRead,
    RagDocumentLogRead,
    RagQueryRecord,
    RagQueryRequest,
    RagQueryResponse,
)

from services.rag_service import DocumentAlreadyExistsError, get_rag_service
from config import get_settings

router = APIRouter(prefix="/api/rag", tags=["rag"])
settings = get_settings()


@router.post("/collections")
def create_collection(
    payload: RagCollectionCreate = Body(
        ...,
        example={"name": "充电协议FAQ", "description": "收录常见的国标充电问题解答"},
    ),
    user: User = Depends(get_current_user),
) -> RagCollectionRead:
    rag_service = get_rag_service()
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
    rag_service = get_rag_service()
    records = rag_service.list_collections(user.id)
    return [RagCollectionRead.model_validate(item) for item in records]


@router.get("/collections/{collection_id}", response_model=RagCollectionRead)
def get_collection(collection_id: int, user: User = Depends(get_current_user)) -> RagCollectionRead:
    rag_service = get_rag_service()
    collection = rag_service.get_collection(collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    if collection.created_by not in {None, user.id} and user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问该知识库")
    return RagCollectionRead.model_validate(collection)


@router.post("/collections/{collection_id}/documents")
async def upload_document(
    collection_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    overwrite: bool = Form(False),
    user: User = Depends(get_current_user),
) -> KnowledgeDocumentRead:
    rag_service = get_rag_service()
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件为空")
    filename = file.filename or "upload.xlsx"
    lower = filename.lower()
    if not (lower.endswith(".xlsx") or lower.endswith(".xlsm")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="当前仅支持上传 Excel（.xlsx/.xlsm）")

    # 先把文件落盘（后台任务读取）
    store_dir = Path(settings.upload_path) / "knowledge-docs" / str(collection_id)
    store_dir.mkdir(parents=True, exist_ok=True)
    store_path = store_dir / filename
    try:
        store_path.write_bytes(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"保存文件失败：{exc}") from exc

    try:
        document = rag_service.prepare_excel_document_upload(
            collection_id=collection_id,
            filename=filename,
            file_path=str(store_path),
            file_size=len(raw_bytes),
            file_type=file.content_type or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            user_id=user.id,
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

    # 后台异步处理：构建索引并更新状态/日志
    if background_tasks is not None:
        background_tasks.add_task(rag_service.process_excel_document_by_id, document.id)
    else:
        # 兜底：理论上不会发生
        rag_service.process_excel_document_by_id(document.id)

    return KnowledgeDocumentRead.model_validate(document)


@router.get("/collections/{collection_id}/documents", response_model=list[KnowledgeDocumentRead])
def list_documents(collection_id: int, user: User = Depends(get_current_user)) -> list[KnowledgeDocumentRead]:
    rag_service = get_rag_service()
    collection = rag_service.get_collection(collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    if collection.created_by not in {None, user.id} and user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问该知识库")
    records = rag_service.list_documents(collection_id)
    return [KnowledgeDocumentRead.model_validate(doc) for doc in records]


@router.get("/collections/{collection_id}/documents/{document_id}", response_model=KnowledgeDocumentRead)
def get_document(collection_id: int, document_id: int, user: User = Depends(get_current_user)) -> KnowledgeDocumentRead:
    rag_service = get_rag_service()
    collection = rag_service.get_collection(collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    if collection.created_by not in {None, user.id} and user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问该知识库")

    doc = rag_service.get_document(document_id)
    if doc is None or int(doc.collection_id) != int(collection_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    return KnowledgeDocumentRead.model_validate(doc)


@router.get("/collections/{collection_id}/documents/{document_id}/logs", response_model=list[RagDocumentLogRead])
def list_document_logs(
    collection_id: int,
    document_id: int,
    limit: int = 200,
    user: User = Depends(get_current_user),
) -> list[RagDocumentLogRead]:
    rag_service = get_rag_service()
    collection = rag_service.get_collection(collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    if collection.created_by not in {None, user.id} and user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问该知识库")

    doc = rag_service.get_document(document_id)
    if doc is None or int(doc.collection_id) != int(collection_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")

    logs = rag_service.list_document_logs(document_id, limit=limit)
    return [RagDocumentLogRead.model_validate(item) for item in logs]


@router.get("/collections/{collection_id}/queries", response_model=list[RagQueryRecord])
def list_queries(
    collection_id: int,
    limit: int = 20,
    user: User = Depends(get_current_user),
) -> list[RagQueryRecord]:
    rag_service = get_rag_service()
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
    rag_service = get_rag_service()
    result = rag_service.query(
        payload.collection_id,
        payload.query,
        user_id=user.id,
        limit=getattr(payload, "top_k", 5) or 5,
        show_retrieval=getattr(payload, "show_retrieval", True),
    )
    return RagQueryResponse(**result)
