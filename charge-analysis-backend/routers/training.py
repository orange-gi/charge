"""训练管理 API。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from config import get_settings
from core.dependencies import get_current_user
from database import get_db
from models import (
    ModelVersion,
    TrainingConfig,
    TrainingDataset,
    TrainingEvaluation,
    TrainingLog,
    TrainingMetrics,
    TrainingStatus,
    TrainingTask,
    User,
)
from schemas import (
    ModelPublishRequest,
    TrainingConfigRequest,
    TrainingConfigResponse,
    TrainingEvaluationRequest,
    TrainingEvaluationResponse,
    TrainingLogResponse,
    TrainingMetricPoint,
    TrainingTaskCreateRequest,
    TrainingTaskDetailResponse,
    TrainingTaskListResponse,
)
from services.training_service import get_training_service

router = APIRouter(prefix="/api/training", tags=["training"])
settings = get_settings()
training_service = get_training_service()


def _loads_json(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _dumps_json(data: dict[str, Any] | None) -> str | None:
    if data is None:
        return None
    return json.dumps(data)


def _config_to_response(config: TrainingConfig) -> TrainingConfigResponse:
    return TrainingConfigResponse(
        id=config.id,
        name=config.name,
        base_model=config.base_model,
        model_path=config.model_path,
        adapter_type=config.adapter_type,
        model_size=config.model_size,
        dataset_strategy=config.dataset_strategy,
        hyperparameters=_loads_json(config.hyperparameters) or {},
        notes=config.notes,
        created_at=config.created_at,
        updated_at=config.updated_at,
    )


def _task_to_response(task: TrainingTask) -> TrainingTaskDetailResponse:
    return TrainingTaskDetailResponse(
        id=task.id,
        name=task.name,
        status=task.status.value,
        progress=task.progress,
        model_size=task.model_size,
        adapter_type=task.adapter_type or "lora",
        dataset_id=task.dataset_id,
        config_id=task.config_id,
        current_epoch=task.current_epoch,
        total_epochs=task.total_epochs,
        metrics=_loads_json(task.metrics) or {},
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


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


@router.get("/configs", response_model=list[TrainingConfigResponse])
def list_training_configs(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    configs = (
        db.query(TrainingConfig)
        .filter((TrainingConfig.created_by == user.id) | (TrainingConfig.created_by.is_(None)))
        .order_by(TrainingConfig.updated_at.desc())
        .all()
    )
    return [_config_to_response(config) for config in configs]


@router.post("/configs", response_model=TrainingConfigResponse)
def create_training_config(
    payload: TrainingConfigRequest = Body(
        ...,
        example={
            "name": "DC充电LoRA配置",
            "base_model": "Qwen2-1.5B",
            "model_path": "/models/qwen2",
            "adapter_type": "lora",
            "model_size": "1.5b",
            "dataset_strategy": "full",
            "hyperparameters": {"learning_rate": 0.0003, "batch_size": 4, "epochs": 5},
            "notes": "适用于国标充电日志微调",
        },
    ),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if payload.model_size not in {"1.5b", "7b"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="当前仅支持 1.5B 或 7B 模型")

    config = TrainingConfig(
        name=payload.name,
        base_model=payload.base_model,
        model_path=payload.model_path,
        adapter_type=payload.adapter_type,
        model_size=payload.model_size,
        dataset_strategy=payload.dataset_strategy,
        hyperparameters=_dumps_json(payload.hyperparameters or {}),
        notes=payload.notes,
        created_by=user.id,
    )
    db.add(config)
    db.commit()
    db.refresh(config)
    return _config_to_response(config)


@router.put("/configs/{config_id}", response_model=TrainingConfigResponse)
def update_training_config(
    config_id: int,
    payload: TrainingConfigRequest = Body(
        ...,
        example={
            "name": "DC充电LoRA配置-更新",
            "base_model": "Qwen2-1.5B",
            "model_path": "/models/qwen2-v2",
            "adapter_type": "lora",
            "model_size": "1.5b",
            "dataset_strategy": "filtered",
            "hyperparameters": {"learning_rate": 0.00025, "batch_size": 8, "epochs": 8},
            "notes": "增加了异常案例样本",
        },
    ),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    config = (
        db.query(TrainingConfig)
        .filter(TrainingConfig.id == config_id, TrainingConfig.created_by == user.id)
        .first()
    )
    if config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="配置不存在")

    for field in (
        "name",
        "base_model",
        "model_path",
        "adapter_type",
        "model_size",
        "dataset_strategy",
        "notes",
    ):
        setattr(config, field, getattr(payload, field))

    config.hyperparameters = _dumps_json(payload.hyperparameters or {})
    db.add(config)
    db.commit()
    db.refresh(config)
    return _config_to_response(config)


@router.post("/tasks", response_model=dict)
def create_task(
    payload: TrainingTaskCreateRequest = Body(
        ...,
        example={
            "name": "国标异常诊断-批次1",
            "description": "使用Dataset 3和Config 2训练直流桩诊断模型",
            "dataset_id": 3,
            "config_id": 2,
            "model_type": "flow_control",
            "model_size": "1.5b",
            "adapter_type": "lora",
            "hyperparameters": {"learning_rate": 0.0002, "epochs": 6, "batch_size": 8},
        },
    ),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if payload.model_size not in {"1.5b", "7b"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="当前仅支持训练 1.5B 或 7B LoRA 模型")
    if payload.adapter_type.lower() != "lora":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="当前仅开放 LoRA 适配器训练")

    dataset = (
        db.query(TrainingDataset)
        .filter(TrainingDataset.id == payload.dataset_id, TrainingDataset.created_by == user.id)
        .first()
    )
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="数据集不存在或无权限")

    config = None
    if payload.config_id:
        config = (
            db.query(TrainingConfig)
            .filter(TrainingConfig.id == payload.config_id, TrainingConfig.created_by == user.id)
            .first()
        )
        if config is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="训练配置不存在或无权限")
        if config.model_size != payload.model_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="训练配置与任务模型大小不一致",
            )

    base_hyper = _loads_json(config.hyperparameters if config else None) or {}
    merged_hyper = {**base_hyper, **(payload.hyperparameters or {})}
    total_epochs = int(merged_hyper.get("epochs", 10))

    task = TrainingTask(
        name=payload.name,
        description=payload.description,
        dataset_id=payload.dataset_id,
        config_id=payload.config_id,
        model_type=payload.model_type,
        adapter_type=payload.adapter_type,
        model_size=payload.model_size,
        hyperparameters=_dumps_json(merged_hyper),
        status=TrainingStatus.PENDING,
        progress=0.0,
        total_epochs=total_epochs,
        total_steps=total_epochs * training_service.steps_per_epoch,
        created_by=user.id,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return {"task_id": task.id, "status": task.status.value}


@router.get("/tasks", response_model=TrainingTaskListResponse)
def list_tasks(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    tasks = (
        db.query(TrainingTask)
        .filter(TrainingTask.created_by == user.id)
        .order_by(TrainingTask.created_at.desc())
        .all()
    )
    return {"items": [_task_to_response(task) for task in tasks]}


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
    return {"message": "训练已加入队列", "task_id": task_id}


@router.get("/tasks/{task_id}", response_model=TrainingTaskDetailResponse)
def get_task(task_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> TrainingTaskDetailResponse:
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id, TrainingTask.created_by == user.id).first()
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    return _task_to_response(task)


@router.get("/tasks/{task_id}/logs", response_model=list[TrainingLogResponse])
def get_task_logs(
    task_id: int,
    limit: int = 200,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id, TrainingTask.created_by == user.id).first()
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    logs = (
        db.query(TrainingLog)
        .filter(TrainingLog.task_id == task_id)
        .order_by(TrainingLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        TrainingLogResponse(
            id=log.id,
            log_level=log.log_level.value,
            message=log.message,
            created_at=log.created_at,
        )
        for log in logs
    ]


@router.get("/tasks/{task_id}/metrics", response_model=list[TrainingMetricPoint])
def get_task_metrics(
    task_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id, TrainingTask.created_by == user.id).first()
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    metrics = (
        db.query(TrainingMetrics)
        .filter(TrainingMetrics.task_id == task_id)
        .order_by(TrainingMetrics.created_at.asc())
        .all()
    )
    return [
        TrainingMetricPoint(
            epoch=metric.epoch,
            step=metric.step,
            loss=metric.loss,
            accuracy=metric.accuracy,
            learning_rate=metric.learning_rate,
            gpu_memory=metric.gpu_memory,
            created_at=metric.created_at,
        )
        for metric in metrics
    ]


@router.post("/tasks/{task_id}/evaluate", response_model=TrainingEvaluationResponse)
def evaluate_task(
    task_id: int,
    payload: TrainingEvaluationRequest = Body(
        ...,
        example={
            "evaluation_type": "automatic",
            "metrics": {"accuracy": 0.91, "loss": 0.34, "latency_ms": 120},
            "recommended_plan": "增加故障日志数据并降低学习率",
            "notes": "桩A样本表现最佳",
        },
    ),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id, TrainingTask.created_by == user.id).first()
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    evaluation = task.evaluation or TrainingEvaluation(task_id=task.id, created_by=user.id, evaluator=user.username or user.email)
    evaluation.evaluation_type = payload.evaluation_type
    evaluation.metrics = _dumps_json(payload.metrics) or "{}"
    evaluation.recommended_plan = payload.recommended_plan
    evaluation.notes = payload.notes

    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)

    return TrainingEvaluationResponse(
        id=evaluation.id,
        task_id=task.id,
        evaluator=evaluation.evaluator,
        evaluation_type=evaluation.evaluation_type,
        metrics=_loads_json(evaluation.metrics) or {},
        recommended_plan=evaluation.recommended_plan,
        notes=evaluation.notes,
        created_at=evaluation.created_at,
    )


@router.get("/tasks/{task_id}/evaluation", response_model=TrainingEvaluationResponse | None)
def get_task_evaluation(
    task_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id, TrainingTask.created_by == user.id).first()
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    evaluation = task.evaluation
    if evaluation is None:
        return None
    return TrainingEvaluationResponse(
        id=evaluation.id,
        task_id=task.id,
        evaluator=evaluation.evaluator,
        evaluation_type=evaluation.evaluation_type,
        metrics=_loads_json(evaluation.metrics) or {},
        recommended_plan=evaluation.recommended_plan,
        notes=evaluation.notes,
        created_at=evaluation.created_at,
    )


@router.post("/tasks/{task_id}/publish")
def publish_task_model(
    task_id: int,
    payload: ModelPublishRequest = Body(
        ...,
        example={
            "version": "v1.0.0",
            "target_environment": "prod-shanghai",
            "endpoint_url": "https://rag.example.com/models/dc-diagnosis",
            "notes": "首个外部可用版本",
            "set_default": True,
        },
    ),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id, TrainingTask.created_by == user.id).first()
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    if task.status != TrainingStatus.COMPLETED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="仅已完成的任务可以发布模型")

    if payload.set_default:
        db.query(ModelVersion).filter(ModelVersion.model_type == task.model_type).update(
            {"is_default": False},
            synchronize_session=False,
        )

    metadata = {
        "task_id": task.id,
        "dataset_id": task.dataset_id,
        "config_id": task.config_id,
        "target_environment": payload.target_environment,
        "endpoint_url": payload.endpoint_url,
        "notes": payload.notes,
    }

    model_version = ModelVersion(
        name=f"{task.name}-{payload.version}",
        model_type=task.model_type,
        version=payload.version,
        model_path=task.model_path or f"/models/task_{task.id}.bin",
        config=json.dumps(metadata),
        metrics=task.metrics,
        is_active=True,
        is_default=payload.set_default,
        created_by=user.id,
    )
    db.add(model_version)
    task.model_version_id = model_version.id
    db.add(task)
    db.commit()
    db.refresh(model_version)

    return {
        "model_version_id": model_version.id,
        "version": model_version.version,
        "endpoint_url": payload.endpoint_url,
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
