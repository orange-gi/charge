"""Pydantic 数据模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

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
    status: str
    progress: float
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisRunResponse(BaseModel):
    analysis_id: int
    status: str


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


# ---------- 训练 ----------


class DatasetCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_type: str = Field(default="standard")


class TrainingTaskCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_id: int
    model_type: str = Field(default="flow_control")
    hyperparameters: dict[str, Any] | None = None


class TrainingTaskResponse(BaseModel):
    id: int
    name: str
    status: str
    progress: float

    class Config:
        from_attributes = True
