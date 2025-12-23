"""训练管理 API。"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
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
    DatasetUploadResponse,
    KeywordEvalResponse,
    ModelPublishRequest,
    ModelPublishResponse,
    ModelVersionCreateRequest,
    ModelVersionCreateResponse,
    SftLoraTaskCreateRequest,
    TaskStartResponse,
    TrainingConfigRequest,
    TrainingConfigResponse,
    TrainingEvaluationRequest,
    TrainingEvaluationResponse,
    TrainingLogResponse,
    TrainingMetricPoint,
    TrainingTaskCreateRequest,
    TrainingTaskCreateResponse,
    TrainingTaskDetailResponse,
    TrainingTaskListResponse,
    YamlParseRequest,
    YamlParseResponse,
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


def _normalize_llamafactory_yaml(cfg: dict[str, Any]) -> dict[str, Any]:
    """将用户提供的 YAML（LLaMAFactory 形式）收敛到后端训练 worker 需要的最小字段集合。"""

    def get(key: str, default: Any = None) -> Any:
        return cfg.get(key, default)

    normalized = {
        # model
        "model_name_or_path": get("model_name_or_path"),
        # method
        "stage": get("stage", "sft"),
        "do_train": bool(get("do_train", True)),
        "finetuning_type": get("finetuning_type", "lora"),
        "lora_target": get("lora_target", "all"),
        "lora_rank": int(get("lora_rank", 8)),
        "lora_alpha": int(get("lora_alpha", 16)),
        "lora_dropout": float(get("lora_dropout", 0.05)),
        # dataset
        "dataset": get("dataset"),
        "template": get("template", "qwen"),
        "cutoff_len": int(get("cutoff_len", 1024)),
        "max_samples": int(get("max_samples", 1000000)),
        "overwrite_cache": bool(get("overwrite_cache", True)),
        "preprocessing_num_workers": int(get("preprocessing_num_workers", 1)),
        # output
        "output_dir": get("output_dir"),
        "logging_steps": int(get("logging_steps", 10)),
        "save_steps": int(get("save_steps", 100)),
        "plot_loss": bool(get("plot_loss", False)),
        "overwrite_output_dir": bool(get("overwrite_output_dir", True)),
        # train
        "per_device_train_batch_size": int(get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(get("gradient_accumulation_steps", 1)),
        "learning_rate": float(get("learning_rate", 5e-4)),
        "num_train_epochs": float(get("num_train_epochs", 1.0)),
        "lr_scheduler_type": get("lr_scheduler_type", "cosine"),
        "warmup_ratio": float(get("warmup_ratio", 0.0)),
        "bf16": bool(get("bf16", False)),
        "ddp_timeout": int(get("ddp_timeout", 180000000)),
        # eval
        "val_size": float(get("val_size", 0.0)),
        "per_device_eval_batch_size": int(get("per_device_eval_batch_size", 1)),
        "eval_strategy": get("eval_strategy", "no"),
        "eval_steps": int(get("eval_steps", 50)),
        # misc
        "seed": int(get("seed", 42)),
    }

    if not normalized["model_name_or_path"]:
        raise ValueError("缺少 model_name_or_path")
    if not normalized["output_dir"]:
        raise ValueError("缺少 output_dir")
    if normalized["stage"] != "sft":
        raise ValueError("最小实现仅支持 stage: sft")
    if normalized["finetuning_type"] != "lora":
        raise ValueError("最小实现仅支持 finetuning_type: lora")
    if not normalized["do_train"]:
        raise ValueError("do_train=false 当前不支持")
    return normalized


def _parse_yaml_text(yaml_text: str) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(yaml_text) or {}
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"YAML 解析失败: {exc}")
    if not isinstance(loaded, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="YAML 顶层必须是 mapping（key-value）")
    try:
        return _normalize_llamafactory_yaml(loaded)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


def _count_json_samples(raw: bytes) -> int:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="ignore")
    s = text.strip()
    if not s:
        return 0
    # JSON / JSON array
    if s[0] in "{[":
        try:
            obj = json.loads(s)
        except Exception:
            return 0
        if isinstance(obj, list):
            return len([x for x in obj if isinstance(x, dict)])
        if isinstance(obj, dict):
            return 1
        return 0
    # JSONL
    count = 0
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            count += 1
    return count


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


@router.post("/datasets", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(..., example="直流桩异常样本-12月"),
    description: str | None = Form(None, example="来自上海实验室的现场调试日志"),
    dataset_type: str = Form("standard", example="standard"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DatasetUploadResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件为空")

    filename = file.filename or "dataset"
    suffix = Path(filename).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        sample_count = _count_json_samples(raw)
        dataset_type = dataset_type or "sft_json"
    else:
        # 兼容旧逻辑：将非空行数 - 1 作为样本数（适配 csv header）
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="ignore")
        lines = [line for line in text.splitlines() if line.strip()]
        sample_count = max(0, len(lines) - 1)

    dataset_dir = Path(settings.upload_path) / "datasets" / str(user.id)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(datetime.utcnow().timestamp())
    file_path = dataset_dir / f"{timestamp}_{filename}"
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
    return DatasetUploadResponse(dataset_id=dataset.id, sample_count=sample_count)


@router.post("/yaml/parse", response_model=YamlParseResponse)
def parse_training_yaml(
    payload: YamlParseRequest = Body(...),
    user: User = Depends(get_current_user),
) -> YamlParseResponse:
    config = _parse_yaml_text(payload.yaml_text)
    return YamlParseResponse(config=config)


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


@router.post("/tasks", response_model=TrainingTaskCreateResponse)
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
) -> TrainingTaskCreateResponse:
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
    return TrainingTaskCreateResponse(task_id=task.id, status=task.status.value)


@router.post("/tasks/sft_lora", response_model=TrainingTaskCreateResponse)
def create_sft_lora_task(
    payload: SftLoraTaskCreateRequest = Body(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TrainingTaskCreateResponse:
    dataset = (
        db.query(TrainingDataset)
        .filter(TrainingDataset.id == payload.dataset_id, TrainingDataset.created_by == user.id)
        .first()
    )
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="数据集不存在或无权限")

    cfg = _parse_yaml_text(payload.yaml_text)

    # 仅支持 1.5b / 7b 的 UI 约束：用 model_name_or_path 粗略推断
    m = str(cfg.get("model_name_or_path") or "").lower()
    model_size = "7b" if "7b" in m else "1.5b"

    total_epochs = int(math.ceil(float(cfg.get("num_train_epochs") or 1.0)))
    hyper = {
        "llamafactory_yaml": cfg,
        "llamafactory_yaml_raw": payload.yaml_text,
    }

    task = TrainingTask(
        name=payload.name,
        description=payload.description,
        dataset_id=payload.dataset_id,
        config_id=None,
        model_type=payload.model_type,
        adapter_type="lora",
        model_size=model_size,
        hyperparameters=json.dumps(hyper, ensure_ascii=False),
        status=TrainingStatus.PENDING,
        progress=0.0,
        total_epochs=total_epochs,
        total_steps=None,
        created_by=user.id,
        model_path=str(cfg.get("output_dir") or ""),
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return TrainingTaskCreateResponse(task_id=task.id, status=task.status.value)


@router.post("/tasks/{task_id}/evaluate_keywords", response_model=KeywordEvalResponse)
async def evaluate_keywords(
    task_id: int,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> KeywordEvalResponse:
    """基于用户上传的评估集（JSON）做关键词命中评估，并同步写入 TrainingEvaluation。"""
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id, TrainingTask.created_by == user.id).first()
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    if not task.model_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务缺少模型输出路径（output_dir）")

    try:
        hyper = json.loads(task.hyperparameters or "{}")
    except Exception:
        hyper = {}
    cfg = hyper.get("llamafactory_yaml") if isinstance(hyper, dict) else None
    if not isinstance(cfg, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务缺少 llamafactory_yaml 配置")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="ignore")
    try:
        items = json.loads(text)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"评估集 JSON 解析失败: {exc}")
    if not isinstance(items, list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="评估集 JSON 顶层必须是数组")

    # lazy import（避免无 GPU 环境启动时开销）
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = cfg.get("model_name_or_path")
    if not base_model:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="llamafactory_yaml 缺少 model_name_or_path")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, task.model_path)
    model.eval()

    details = []
    strict_pass = 0
    hit_rate_sum = 0.0

    for it in items:
        if not isinstance(it, dict):
            continue
        question = str(it.get("question") or "").strip()
        expected = it.get("expected_keywords") or []
        expected = [str(x) for x in expected if str(x).strip()]
        if not question:
            continue

        # Qwen-style chat prompt（最小实现）
        messages = [
            {"role": "system", "content": "新能源充放电专家"},
            {"role": "user", "content": question},
        ]
        try:
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        except Exception:
            prompt = f"[SYSTEM]新能源充放电专家\n[USER]{question}\n[ASSISTANT]"
            inputs = tokenizer(prompt, return_tensors="pt").input_ids

        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
        else:
            input_ids = inputs
            attention_mask = None

        if torch.cuda.is_available():
            input_ids = input_ids.to(model.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=False,
            )

        gen = out[0][input_ids.shape[-1] :]
        answer = tokenizer.decode(gen, skip_special_tokens=True).strip()

        hit_keywords = []
        for kw in expected:
            if not kw:
                continue
            if kw in answer:
                hit_keywords.append(kw)
                continue
            # 英文大小写兼容
            if kw.lower() in answer.lower():
                hit_keywords.append(kw)

        hit_rate = (len(hit_keywords) / len(expected)) if expected else 0.0
        strict_ok = bool(expected) and len(hit_keywords) == len(expected)
        if strict_ok:
            strict_pass += 1
        hit_rate_sum += hit_rate

        details.append(
            {
                "question": question,
                "expected_keywords": expected,
                "answer": answer,
                "hit_keywords": hit_keywords,
                "hit_rate": round(hit_rate, 4),
                "strict_pass": strict_ok,
            }
        )

    total = len(details)
    strict_pass_rate = round((strict_pass / total) if total else 0.0, 4)
    avg_hit_rate = round((hit_rate_sum / total) if total else 0.0, 4)

    # 写入评估记录（用于前端“训练评估”页复用展示）
    evaluation = task.evaluation or TrainingEvaluation(task_id=task.id, created_by=user.id, evaluator=user.username or user.email)
    evaluation.evaluation_type = "keyword"
    evaluation.metrics = json.dumps(
        {"strict_pass_rate": strict_pass_rate, "avg_hit_rate": avg_hit_rate, "total": total},
        ensure_ascii=False,
    )
    evaluation.recommended_plan = "keyword_eval"
    evaluation.notes = f"关键词评估完成：strict_pass_rate={strict_pass_rate} avg_hit_rate={avg_hit_rate} total={total}"
    db.add(evaluation)
    db.commit()

    return KeywordEvalResponse(
        task_id=task.id,
        total=total,
        strict_pass_rate=strict_pass_rate,
        avg_hit_rate=avg_hit_rate,
        details=details,
    )

@router.get("/tasks", response_model=TrainingTaskListResponse)
def list_tasks(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    tasks = (
        db.query(TrainingTask)
        .filter(TrainingTask.created_by == user.id)
        .order_by(TrainingTask.created_at.desc())
        .all()
    )
    return {"items": [_task_to_response(task) for task in tasks]}


@router.post("/tasks/{task_id}/start", response_model=TaskStartResponse)
async def start_task(
    task_id: int,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TaskStartResponse:
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id, TrainingTask.created_by == user.id).first()
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    if task.status not in {TrainingStatus.PENDING, TrainingStatus.FAILED}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务状态不可启动")

    task.status = TrainingStatus.QUEUED
    db.add(task)
    db.commit()

    background_tasks.add_task(training_service.start_training, task_id)
    return TaskStartResponse(message="训练已加入队列", task_id=task_id)


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


@router.post("/tasks/{task_id}/publish", response_model=ModelPublishResponse)
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
) -> ModelPublishResponse:
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

    return ModelPublishResponse(
        model_version_id=model_version.id,
        version=model_version.version,
        endpoint_url=payload.endpoint_url,
    )


@router.post("/models", response_model=ModelVersionCreateResponse)
def create_model_version(
    payload: ModelVersionCreateRequest = Body(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    record = ModelVersion(
        name=payload.name,
        model_type=payload.model_type,
        version=payload.version,
        model_path=payload.model_path,
        config=payload.config,
        metrics=payload.metrics,
        created_by=user.id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return ModelVersionCreateResponse(model_version_id=record.id)
