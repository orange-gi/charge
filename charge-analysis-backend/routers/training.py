"""训练管理 API。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from config import get_settings
from core.dependencies import get_current_user
from database import get_db
from models import ModelVersion, TrainingDataset, TrainingStatus, TrainingTask, User
from schemas import TrainingTaskCreateRequest
from services.training_service import get_training_service

router = APIRouter(prefix="/api/training", tags=["training"])
settings = get_settings()
training_service = get_training_service()


@router.post("/datasets")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str | None = Form(None),
    dataset_type: str = Form("standard"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件为空")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="ignore")

    lines = [line for line in text.splitlines() if line.strip()]
    sample_count = max(0, len(lines) - 1)

    dataset_dir = Path(settings.upload_path) / "datasets" / str(user.id)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(datetime.utcnow().timestamp())
    file_path = dataset_dir / f"{timestamp}_{file.filename}"
    file_path.write_bytes(raw)

    dataset = TrainingDataset(
        name=name,
        description=description,
        dataset_type=dataset_type,
        file_path=str(file_path),
        sample_count=sample_count,
        meta_info=json.dumps({"filename": file.filename, "uploaded_at": datetime.utcnow().isoformat()}),
        is_public=False,
        created_by=user.id,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return {"dataset_id": dataset.id, "sample_count": sample_count}


@router.post("/tasks", response_model=dict)
def create_task(
    payload: TrainingTaskCreateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == payload.dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="数据集不存在")

    task = TrainingTask(
        name=payload.name,
        description=payload.description,
        dataset_id=payload.dataset_id,
        model_type=payload.model_type,
        hyperparameters=json.dumps(payload.hyperparameters or {}),
        status=TrainingStatus.PENDING,
        progress=0.0,
        total_epochs=(payload.hyperparameters or {}).get("epochs", 10),
        created_by=user.id,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return {"task_id": task.id, "status": task.status.value}


@router.post("/tasks/{task_id}/start")
async def start_task(
    task_id: int,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id, TrainingTask.created_by == user.id).first()
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    if task.status not in {TrainingStatus.PENDING, TrainingStatus.FAILED}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务状态不可启动")

    task.status = TrainingStatus.QUEUED
    db.add(task)
    db.commit()

    background_tasks.add_task(training_service.start_training, task_id)
    return {"message": "训练已启动", "task_id": task_id}


@router.get("/tasks/{task_id}")
def get_task(task_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id, TrainingTask.created_by == user.id).first()
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    return {
        "id": task.id,
        "name": task.name,
        "status": task.status.value,
        "progress": task.progress,
        "metrics": json.loads(task.metrics) if task.metrics else None,
    }


@router.post("/models")
def create_model_version(
    name: str = Form(...),
    model_type: str = Form("flow_control"),
    version: str = Form(...),
    model_path: str = Form(...),
    config: str | None = Form(None),
    metrics: str | None = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    record = ModelVersion(
        name=name,
        model_type=model_type,
        version=version,
        model_path=model_path,
        config=config,
        metrics=metrics,
        created_by=user.id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"model_version_id": record.id}
