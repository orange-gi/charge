"""Pydantic 数据模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Literal

from pydantic import BaseModel, EmailStr, Field


# ---------- 用户 / 认证 ----------


class UserBase(BaseModel):
    email: EmailStr
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(min_length=6, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserRead(UserBase):
    id: int
    role: str

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    user: UserRead
    token: str


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

    class Config:
        from_attributes = True


class AnalysisRunRequest(BaseModel):
    """分析运行请求"""
    signal_names: Optional[list[str]] = Field(
        default=None,
        description="要解析的信号名称列表，如果为空则解析所有信号"
    )


class AnalysisRunResponse(BaseModel):
    analysis_id: int
    status: str


class SignalInfo(BaseModel):
    """信号信息"""
    name: str
    message_name: str
    message_id: str


class AvailableSignalsResponse(BaseModel):
    """可用信号列表响应"""
    signals: list[SignalInfo]
    total_count: int


class AnalysisListResponse(BaseModel):
    items: list[AnalysisRead]


# ---------- RAG ----------


class RagQueryRequest(BaseModel):
    collection_id: int
    query: str = Field(min_length=1, max_length=2000)


class RagQueryResponse(BaseModel):
    response: str
    documents: list[dict[str, Any]]
    query_time: int


class RagCollectionCreate(BaseModel):
    name: str
    description: Optional[str] = None


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

    class Config:
        from_attributes = True


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

    class Config:
        from_attributes = True


class RagQueryRecord(BaseModel):
    id: int
    collection_id: int
    query_text: str
    result_count: int
    response_text: Optional[str]
    user_id: Optional[int]
    query_time_ms: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


# ---------- 训练 ----------


class DatasetCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_type: str = Field(default="standard")


class TrainingConfigRequest(BaseModel):
    name: str
    base_model: str
    model_path: str
    adapter_type: Literal["lora"] = "lora"
    model_size: Literal["1.5b", "7b"]
    dataset_strategy: str = "full"
    hyperparameters: dict[str, Any] | None = None
    notes: Optional[str] = None


class TrainingConfigResponse(TrainingConfigRequest):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TrainingTaskCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_id: int
    config_id: Optional[int] = None
    model_type: str = Field(default="flow_control")
    model_size: Literal["1.5b", "7b"] = "1.5b"
    adapter_type: Literal["lora"] = "lora"
    hyperparameters: dict[str, Any] | None = None


class TrainingTaskResponse(BaseModel):
    id: int
    name: str
    status: str
    progress: float
    model_size: str
    adapter_type: str
    dataset_id: Optional[int]
    config_id: Optional[int]

    class Config:
        from_attributes = True


class TrainingTaskDetailResponse(TrainingTaskResponse):
    current_epoch: Optional[int]
    total_epochs: Optional[int]
    metrics: Optional[dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class TrainingLogResponse(BaseModel):
    id: int
    log_level: str
    message: str
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingMetricPoint(BaseModel):
    epoch: int
    step: int
    loss: Optional[float]
    accuracy: Optional[float]
    learning_rate: Optional[float]
    gpu_memory: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingEvaluationRequest(BaseModel):
    evaluation_type: str = "automatic"
    metrics: dict[str, Any]
    recommended_plan: str
    notes: Optional[str] = None


class TrainingEvaluationResponse(TrainingEvaluationRequest):
    id: int
    task_id: int
    evaluator: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ModelPublishRequest(BaseModel):
    version: str
    target_environment: str
    endpoint_url: Optional[str] = None
    notes: Optional[str] = None
    set_default: bool = False


class TrainingTaskListResponse(BaseModel):
    items: list[TrainingTaskDetailResponse]
