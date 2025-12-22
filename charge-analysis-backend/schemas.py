"""Pydantic 数据模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ---------- 用户 / 认证 ----------


class UserBase(BaseModel):
    email: EmailStr
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(min_length=6, max_length=128)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "engineer.li@example.com",
                "password": "Charge#2024",
                "username": "li_engineer",
                "first_name": "Li",
                "last_name": "Lei",
            }
        }
    )


class UserLogin(BaseModel):
    email: EmailStr
    password: str

    model_config = ConfigDict(
        json_schema_extra={"example": {"email": "engineer.li@example.com", "password": "Charge#2024"}}
    )


class UserRead(UserBase):
    id: int
    role: str

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 7,
                "email": "engineer.li@example.com",
                "username": "li_engineer",
                "first_name": "Li",
                "last_name": "Lei",
                "role": "user",
            }
        },
    )


class AuthResponse(BaseModel):
    user: UserRead
    token: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user": {
                    "id": 7,
                    "email": "engineer.li@example.com",
                    "username": "li_engineer",
                    "first_name": "Li",
                    "last_name": "Lei",
                    "role": "user",
                },
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            }
        }
    )


# ---------- 分析 ----------


class AnalysisRead(BaseModel):
    id: int
    name: str
    description: Optional[str]
    file_path: str
    file_size: Optional[int]
    file_type: Optional[str]
    status: str
    progress: float
    result_data: Optional[str]
    error_message: Optional[str]
    user_id: int
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 42,
                "name": "国标充电日志-2024Q4",
                "description": "12月现场采集的DC桩充电过程",
                "file_path": "/data/uploads/7/1733909123_Log_001.blf",
                "file_size": 1844674407,
                "file_type": "blf",
                "status": "processing",
                "progress": 32.5,
                "result_data": None,
                "error_message": None,
                "user_id": 7,
                "created_at": "2024-12-11T08:30:12.000Z",
                "updated_at": "2024-12-11T08:35:20.000Z",
                "started_at": "2024-12-11T08:31:00.000Z",
                "completed_at": None,
            }
        },
    )


class AnalysisRunRequest(BaseModel):
    """分析运行请求"""

    signal_names: Optional[list[str]] = Field(
        default=None,
        description="要解析的信号名称列表，如果为空则解析所有信号",
        example=["BatteryVoltage", "ChargeCurrent", "StateOfCharge"],
    )


class AnalysisRunResponse(BaseModel):
    analysis_id: int
    status: str

    model_config = ConfigDict(
        json_schema_extra={"example": {"analysis_id": 42, "status": "processing"}}
    )


class SignalInfo(BaseModel):
    """信号信息"""

    name: str
    message_name: str
    message_id: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"name": "BatteryVoltage", "message_name": "BMS_BatteryState", "message_id": "0x102"}
        }
    )


class AvailableSignalsResponse(BaseModel):
    """可用信号列表响应"""

    signals: list[SignalInfo]
    total_count: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "signals": [
                    {"name": "BatteryVoltage", "message_name": "BMS_BatteryState", "message_id": "0x102"},
                    {"name": "ChargeCurrent", "message_name": "PCS_ChargeCtrl", "message_id": "0x221"},
                ],
                "total_count": 2,
            }
        }
    )


class AnalysisResultRead(BaseModel):
    id: int
    analysis_id: int
    result_type: str
    title: str
    content: Optional[str]
    meta_info: Optional[str]
    confidence_score: Optional[float]
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 3,
                "analysis_id": 42,
                "result_type": "anomaly_summary",
                "title": "电压波动超标段落",
                "content": "关键桩在 12:31-12:33 期间出现 15% 波动。",
                "meta_info": '{"frame_id":258,"sample_count":120}',
                "confidence_score": 0.82,
                "created_at": "2024-12-11T08:40:12.000Z",
            }
        },
    )


class AnalysisResultsResponse(BaseModel):
    analysis: AnalysisRead
    results: list[AnalysisResultRead]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "analysis": AnalysisRead.model_config["json_schema_extra"]["example"],
                "results": [AnalysisResultRead.model_config["json_schema_extra"]["example"]],
            }
        }
    )


class AnalysisListResponse(BaseModel):
    items: list[AnalysisRead]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    AnalysisRead.model_config["json_schema_extra"]["example"],
                    {
                        **AnalysisRead.model_config["json_schema_extra"]["example"],
                        "id": 41,
                        "name": "交流桩-调试样本",
                        "status": "completed",
                        "progress": 100,
                    },
                ]
            }
        }
    )


# ---------- RAG ----------


class RagQueryRequest(BaseModel):
    collection_id: int
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50, description="返回的命中条目数量")
    show_retrieval: bool = Field(default=True, description="是否返回检索命中明细（证据链）")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "collection_id": 12,
                "query": "停充码1001代表什么含义？",
                "top_k": 5,
                "show_retrieval": True,
            }
        }
    )


class RagQueryResponse(BaseModel):
    response: str
    documents: list[dict[str, Any]]
    query_time: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "基于集合 12 找到 2 个相关文档，请重点检查PCS控制板与桩体接地。",
                "documents": [
                    {
                        "id": 18,
                        "filename": "DC充电常见问题.pdf",
                        "snippet": "当输出电压波动超过±10%时，应立即停机并检查...",
                        "score": 4.0,
                    }
                ],
                "query_time": 48,
            }
        }
    )


class RagCollectionCreate(BaseModel):
    name: str
    description: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"name": "充电协议FAQ", "description": "收录国标充电相关问题与答案"}
        }
    )


class RagCollectionRead(BaseModel):
    id: int
    name: str
    description: Optional[str]
    collection_type: str
    document_count: int
    embedding_model: str
    is_active: bool
    created_by: Optional[int]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 12,
                "name": "充电协议FAQ",
                "description": "收录国标直流充电常见问题",
                "collection_type": "document",
                "document_count": 5,
                "embedding_model": "bge-base-zh-v1.5",
                "is_active": True,
                "created_by": 7,
                "created_at": "2024-12-10T03:21:00.000Z",
                "updated_at": "2024-12-11T09:15:00.000Z",
            }
        },
    )


class KnowledgeDocumentRead(BaseModel):
    id: int
    collection_id: int
    filename: str
    file_path: str
    file_size: Optional[int]
    file_type: Optional[str]
    chunk_count: int
    meta_info: Optional[str] = None
    upload_status: str
    processing_error: Optional[str]
    uploaded_by: Optional[int]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 18,
                "collection_id": 12,
                "filename": "国标直流充电FAQ.pdf",
                "file_path": "/knowledge-docs/12/faq.pdf",
                "file_size": 524288,
                "file_type": "application/pdf",
                "chunk_count": 16,
                "meta_info": '{"pages":12}',
                "upload_status": "completed",
                "processing_error": None,
                "uploaded_by": 7,
                "created_at": "2024-12-11T08:10:00.000Z",
                "updated_at": "2024-12-11T08:11:00.000Z",
            }
        },
    )


class RagQueryRecord(BaseModel):
    id: int
    collection_id: int
    query_text: str
    result_count: int
    response_text: Optional[str]
    user_id: Optional[int]
    query_time_ms: Optional[int]
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 102,
                "collection_id": 12,
                "query_text": "输出电压波动异常",
                "result_count": 2,
                "response_text": "基于集合 12 找到 2 个相关文档。",
                "user_id": 7,
                "query_time_ms": 48,
                "created_at": "2024-12-11T09:20:12.000Z",
            }
        },
    )


# ---------- 通用响应 ----------


class MessageResponse(BaseModel):
    message: str

    model_config = ConfigDict(json_schema_extra={"example": {"message": "操作成功"}})


class AnalysisCancelResponse(BaseModel):
    message: str
    analysis_id: int
    was_running: bool

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"message": "分析已取消", "analysis_id": 42, "was_running": True}
        }
    )


class DatasetUploadResponse(BaseModel):
    dataset_id: int
    sample_count: int

    model_config = ConfigDict(
        json_schema_extra={"example": {"dataset_id": 3, "sample_count": 128}}
    )


class TrainingTaskCreateResponse(BaseModel):
    task_id: int
    status: str

    model_config = ConfigDict(
        json_schema_extra={"example": {"task_id": 15, "status": "pending"}}
    )


class TaskStartResponse(BaseModel):
    message: str
    task_id: int

    model_config = ConfigDict(
        json_schema_extra={"example": {"message": "训练已加入队列", "task_id": 15}}
    )


class ModelPublishResponse(BaseModel):
    model_version_id: int
    version: str
    endpoint_url: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_version_id": 9,
                "version": "v1.0.0",
                "endpoint_url": "https://rag.example.com/models/dc-diagnosis",
            }
        }
    )


class ModelVersionCreateResponse(BaseModel):
    model_version_id: int

    model_config = ConfigDict(
        json_schema_extra={"example": {"model_version_id": 9}}
    )


# ---------- 训练 ----------


class DatasetCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_type: str = Field(default="standard")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "直流桩异常样本-12月",
                "description": "来自上海实验室的现场调试日志",
                "dataset_type": "standard",
            }
        }
    )


class TrainingConfigRequest(BaseModel):
    name: str
    base_model: str
    model_path: str
    adapter_type: Literal["lora"] = "lora"
    model_size: Literal["1.5b", "7b"]
    dataset_strategy: str = "full"
    hyperparameters: dict[str, Any] | None = None
    notes: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "DC充电LoRA配置",
                "base_model": "Qwen2-1.5B",
                "model_path": "/models/qwen2",
                "adapter_type": "lora",
                "model_size": "1.5b",
                "dataset_strategy": "full",
                "hyperparameters": {"learning_rate": 0.0003, "batch_size": 4, "epochs": 5},
                "notes": "适用于国标充电日志微调",
            }
        }
    )


class TrainingConfigResponse(TrainingConfigRequest):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 2,
                "name": "DC充电LoRA配置",
                "base_model": "Qwen2-1.5B",
                "model_path": "/models/qwen2",
                "adapter_type": "lora",
                "model_size": "1.5b",
                "dataset_strategy": "full",
                "hyperparameters": {"learning_rate": 0.0003, "batch_size": 4, "epochs": 5},
                "notes": "适用于国标充电日志微调",
                "created_at": "2024-12-05T07:30:00.000Z",
                "updated_at": "2024-12-10T12:15:00.000Z",
            }
        },
    )


class TrainingTaskCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_id: int
    config_id: Optional[int] = None
    model_type: str = Field(default="flow_control")
    model_size: Literal["1.5b", "7b"] = "1.5b"
    adapter_type: Literal["lora"] = "lora"
    hyperparameters: dict[str, Any] | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "国标异常诊断-批次1",
                "description": "使用Dataset 3和Config 2训练直流桩诊断模型",
                "dataset_id": 3,
                "config_id": 2,
                "model_type": "flow_control",
                "model_size": "1.5b",
                "adapter_type": "lora",
                "hyperparameters": {"learning_rate": 0.0002, "epochs": 6, "batch_size": 8},
            }
        }
    )


class YamlParseRequest(BaseModel):
    yaml_text: str = Field(min_length=1, description="YAML 配置内容（LLaMAFactory 风格的最小子集）")


class YamlParseResponse(BaseModel):
    config: dict[str, Any]


class SftLoraTaskCreateRequest(BaseModel):
    """通过 YAML 一键创建 sft+lora 训练任务（最小化 llamafactory-cli train）。"""

    name: str
    description: Optional[str] = None
    dataset_id: int
    model_type: str = Field(default="flow_control")
    yaml_text: str = Field(min_length=1)


class KeywordEvalItem(BaseModel):
    question: str = Field(min_length=1)
    expected_keywords: list[str] = Field(default_factory=list)


class KeywordEvalDetail(BaseModel):
    question: str
    expected_keywords: list[str]
    answer: str
    hit_keywords: list[str]
    hit_rate: float
    strict_pass: bool


class KeywordEvalResponse(BaseModel):
    task_id: int
    total: int
    strict_pass_rate: float
    avg_hit_rate: float
    details: list[KeywordEvalDetail]


class TrainingTaskResponse(BaseModel):
    id: int
    name: str
    status: str
    progress: float
    model_size: str
    adapter_type: str
    dataset_id: Optional[int]
    config_id: Optional[int]

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 15,
                "name": "国标异常诊断-批次1",
                "status": "processing",
                "progress": 45.0,
                "model_size": "1.5b",
                "adapter_type": "lora",
                "dataset_id": 3,
                "config_id": 2,
            }
        },
    )


class TrainingTaskDetailResponse(TrainingTaskResponse):
    current_epoch: Optional[int]
    total_epochs: Optional[int]
    metrics: Optional[dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                **TrainingTaskResponse.model_config["json_schema_extra"]["example"],
                "current_epoch": 2,
                "total_epochs": 6,
                "metrics": {"accuracy": 0.87, "loss": 0.42},
                "created_at": "2024-12-11T05:00:00.000Z",
                "updated_at": "2024-12-11T08:30:00.000Z",
            }
        },
    )


class TrainingLogResponse(BaseModel):
    id: int
    log_level: str
    message: str
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 120,
                "log_level": "INFO",
                "message": "Epoch 2 finished, val_loss=0.38",
                "created_at": "2024-12-11T08:12:00.000Z",
            }
        },
    )


class TrainingMetricPoint(BaseModel):
    epoch: int
    step: int
    loss: Optional[float]
    accuracy: Optional[float]
    learning_rate: Optional[float]
    gpu_memory: Optional[float]
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "epoch": 2,
                "step": 1200,
                "loss": 0.42,
                "accuracy": 0.88,
                "learning_rate": 0.0002,
                "gpu_memory": 14.5,
                "created_at": "2024-12-11T08:10:00.000Z",
            }
        },
    )


class TrainingEvaluationRequest(BaseModel):
    evaluation_type: str = "automatic"
    metrics: dict[str, Any]
    recommended_plan: str
    notes: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "evaluation_type": "automatic",
                "metrics": {"accuracy": 0.91, "loss": 0.34, "latency_ms": 120},
                "recommended_plan": "增加故障日志数据并降低学习率",
                "notes": "桩A样本表现最佳",
            }
        }
    )


class TrainingEvaluationResponse(TrainingEvaluationRequest):
    id: int
    task_id: int
    evaluator: Optional[str]
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 5,
                "task_id": 15,
                "evaluator": "li_engineer",
                "evaluation_type": "automatic",
                "metrics": {"accuracy": 0.91, "loss": 0.34, "latency_ms": 120},
                "recommended_plan": "增加故障日志数据并降低学习率",
                "notes": "桩A样本表现最佳",
                "created_at": "2024-12-11T09:00:00.000Z",
            }
        },
    )


class ModelPublishRequest(BaseModel):
    version: str
    target_environment: str
    endpoint_url: Optional[str] = None
    notes: Optional[str] = None
    set_default: bool = False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "version": "v1.0.0",
                "target_environment": "prod-shanghai",
                "endpoint_url": "https://rag.example.com/models/dc-diagnosis",
                "notes": "首个外部可用版本",
                "set_default": True,
            }
        }
    )


class TrainingTaskListResponse(BaseModel):
    items: list[TrainingTaskDetailResponse]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    TrainingTaskDetailResponse.model_config["json_schema_extra"]["example"]
                ]
            }
        }
    )
