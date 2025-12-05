from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, 
    JSON, Enum, BigInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()


class UserRole(str, enum.Enum):
    """用户角色枚举"""
    USER = "user"
    ADMIN = "admin"


class AnalysisStatus(str, enum.Enum):
    """分析状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATION_FAILED = "validation_failed"
    MAX_ITERATIONS = "max_iterations_reached"


class DocumentType(str, enum.Enum):
    """文档类型枚举"""
    DOCUMENT = "document"
    GUIDE = "guide"
    FAQ = "faq"


class UploadStatus(str, enum.Enum):
    """上传状态枚举"""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingStatus(str, enum.Enum):
    """训练状态枚举"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LogLevel(str, enum.Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class User(Base):
    """用户模型"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    avatar_url = Column(String(255))
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # 关系
    analyses = relationship("ChargingAnalysis", back_populates="user")
    sessions = relationship("UserSession", back_populates="user")
    knowledge_collections = relationship("KnowledgeCollection", back_populates="created_by_user")
    documents = relationship("KnowledgeDocument", back_populates="uploaded_by_user")
    training_datasets = relationship("TrainingDataset", back_populates="created_by_user")
    training_tasks = relationship("TrainingTask", back_populates="created_by_user")
    model_versions = relationship("ModelVersion", back_populates="created_by_user")
    system_logs = relationship("SystemLog", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")


class UserSession(Base):
    """用户会话模型"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(255), nullable=False, unique=True)
    expires_at = Column(DateTime, nullable=False)
    ip_address = Column(String(45))  # IPv6 支持
    user_agent = Column(Text)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # 关系
    user = relationship("User", back_populates="sessions")


class ChargingAnalysis(Base):
    """充电分析模型"""
    __tablename__ = "charging_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger)
    file_type = Column(String(20), default="blf")
    status = Column(Enum(AnalysisStatus), default=AnalysisStatus.PENDING, nullable=False)
    progress = Column(Float, default=0.0)
    result_data = Column(Text)  # JSON 数据
    error_message = Column(Text)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # 关系
    user = relationship("User", back_populates="analyses")
    results = relationship("AnalysisResult", back_populates="analysis", cascade="all, delete-orphan")


class AnalysisResult(Base):
    """分析结果模型"""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("charging_analyses.id", ondelete="CASCADE"), nullable=False)
    result_type = Column(String(50), nullable=False)
    title = Column(String(200), nullable=False)
    content = Column(Text)
    confidence_score = Column(Float)
    meta_info = Column(Text)  # JSON 数据
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # 关系
    analysis = relationship("ChargingAnalysis", back_populates="results")


class KnowledgeCollection(Base):
    """知识库集合模型"""
    __tablename__ = "knowledge_collections"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    collection_type = Column(Enum(DocumentType), default=DocumentType.DOCUMENT, nullable=False)
    chroma_collection_id = Column(String(255), unique=True)
    document_count = Column(Integer, default=0)
    embedding_model = Column(String(100), default="bge-base-zh-v1.5")
    is_active = Column(Boolean, default=True, nullable=False)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # 关系
    created_by_user = relationship("User", back_populates="knowledge_collections")
    documents = relationship("KnowledgeDocument", back_populates="collection", cascade="all, delete-orphan")
    queries = relationship("RAGQuery", back_populates="collection")


class KnowledgeDocument(Base):
    """知识文档模型"""
    __tablename__ = "knowledge_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(Integer, ForeignKey("knowledge_collections.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger)
    file_type = Column(String(50))
    content = Column(Text)
    chunk_count = Column(Integer, default=0)
    meta_info = Column(Text)  # JSON 数据
    upload_status = Column(Enum(UploadStatus), default=UploadStatus.UPLOADING, nullable=False)
    processing_error = Column(Text)
    uploaded_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # 关系
    collection = relationship("KnowledgeCollection", back_populates="documents")
    uploaded_by_user = relationship("User", back_populates="documents")


class RAGQuery(Base):
    """RAG 查询历史模型"""
    __tablename__ = "rag_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(Integer, ForeignKey("knowledge_collections.id", ondelete="CASCADE"), nullable=False)
    query_text = Column(Text, nullable=False)
    result_count = Column(Integer, default=0)
    response_text = Column(Text)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    query_time_ms = Column(Integer)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # 关系
    collection = relationship("KnowledgeCollection", back_populates="queries")


class TrainingDataset(Base):
    """训练数据集模型"""
    __tablename__ = "training_datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    dataset_type = Column(String(50), default="standard")
    file_path = Column(String(500))
    sample_count = Column(Integer, default=0)
    meta_info = Column(Text)  # JSON 数据
    is_public = Column(Boolean, default=False, nullable=False)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # 关系
    created_by_user = relationship("User", back_populates="training_datasets")
    training_tasks = relationship("TrainingTask", back_populates="dataset")


class ModelVersion(Base):
    """模型版本模型"""
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), default="flow_control")
    version = Column(String(50), nullable=False)
    model_path = Column(String(500), nullable=False)
    config = Column(Text)  # JSON 数据
    metrics = Column(Text)  # JSON 数据
    is_active = Column(Boolean, default=False, nullable=False)
    is_default = Column(Boolean, default=False, nullable=False)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # 关系
    created_by_user = relationship("User", back_populates="model_versions")
    training_tasks = relationship("TrainingTask", back_populates="model_version")


class TrainingTask(Base):
    """训练任务模型"""
    __tablename__ = "training_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    dataset_id = Column(Integer, ForeignKey("training_datasets.id", ondelete="SET NULL"))
    model_version_id = Column(Integer, ForeignKey("model_versions.id", ondelete="SET NULL"))
    model_type = Column(String(50), nullable=False)
    hyperparameters = Column(Text)  # JSON 数据
    status = Column(Enum(TrainingStatus), default=TrainingStatus.PENDING, nullable=False)
    progress = Column(Float, default=0.0)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer)
    metrics = Column(Text)  # JSON 数据
    logs = Column(Text)
    error_message = Column(Text)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_seconds = Column(Integer)
    gpu_memory_usage = Column(Text)  # JSON 数据
    model_path = Column(String(500))
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # 关系
    created_by_user = relationship("User", back_populates="training_tasks")
    dataset = relationship("TrainingDataset", back_populates="training_tasks")
    model_version = relationship("ModelVersion", back_populates="training_tasks")
    metrics_history = relationship("TrainingMetrics", back_populates="task", cascade="all, delete-orphan")


class TrainingMetrics(Base):
    """训练指标历史模型"""
    __tablename__ = "training_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("training_tasks.id", ondelete="CASCADE"), nullable=False)
    epoch = Column(Integer, nullable=False)
    step = Column(Integer, nullable=False)
    loss = Column(Float)
    accuracy = Column(Float)
    learning_rate = Column(Float)
    gpu_memory = Column(Float)
    custom_metrics = Column(Text)  # JSON 数据
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # 关系
    task = relationship("TrainingTask", back_populates="metrics_history")


class SystemLog(Base):
    """系统日志模型"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    level = Column(Enum(LogLevel), nullable=False)
    module = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    logger_name = Column(String(100))
    function_name = Column(String(100))
    line_number = Column(Integer)
    file_path = Column(String(255))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    request_id = Column(String(100))
    session_id = Column(String(100))
    meta_info = Column(Text)  # JSON 数据
    ip_address = Column(String(45))  # IPv6 支持
    user_agent = Column(Text)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # 关系
    user = relationship("User", back_populates="system_logs")


class AuditLog(Base):
    """操作审计模型"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(50), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(100))
    old_values = Column(Text)  # JSON 数据
    new_values = Column(Text)  # JSON 数据
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    ip_address = Column(String(45))  # IPv6 支持
    user_agent = Column(Text)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # 关系
    user = relationship("User", back_populates="audit_logs")