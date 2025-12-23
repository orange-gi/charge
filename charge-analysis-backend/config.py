from pathlib import Path
from typing import Optional, Any
import json

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用基本配置
    app_name: str = "Charge Analysis Backend"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # 服务器配置
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=True, env="RELOAD")
    
    # 数据库配置
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/charge_analysis",
        env="DATABASE_URL"
    )
    
    # JWT 配置
    secret_key: str = Field(
        default="your-secret-key-here-change-in-production",
        env="SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=30, 
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    refresh_token_expire_days: int = Field(
        default=30, 
        env="REFRESH_TOKEN_EXPIRE_DAYS"
    )
    
    # Redis 配置
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # 文件上传配置
    upload_path: Path = Field(
        default=Path("uploads"),
        env="UPLOAD_PATH"
    )
    max_file_size: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        env="MAX_FILE_SIZE"
    )
    allowed_extensions: list[str] = Field(
        default=[".blf", ".csv", ".xlsx", ".pdf", ".doc", ".docx", ".txt"],
        env="ALLOWED_EXTENSIONS"
    )
    
    # ChromaDB 配置
    chroma_persist_directory: Path = Field(
        default=Path("chromadb_data"),
        env="CHROMA_PERSIST_DIRECTORY"
    )
    chroma_collection_name: str = Field(
        default="charging_knowledge",
        env="CHROMA_COLLECTION_NAME"
    )
    
    # 模型配置
    bge_model_name: str = Field(
        default="BAAI/bge-base-zh-v1.5",
        env="BGE_MODEL_NAME"
    )
    bge_model_path: Optional[Path] = Field(
        default=Path("../CHARGE/models"),
        env="BGE_MODEL_PATH"
    )
    small_model_path: Path = Field(
        default=Path("models/1.5b_flow_control_model"),
        env="SMALL_MODEL_PATH"
    )
    llm_model_path: Path = Field(
        default=Path("models/llm_model"),
        env="LLM_MODEL_PATH"
    )
    
    # LLM 配置
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        env="OPENAI_BASE_URL"
    )
    llm_model_name: str = Field(
        default="deepseek-reasoner",
        env="LLM_MODEL_NAME"
    )
    
    # 训练配置
    training_workers: int = Field(default=4, env="TRAINING_WORKERS")
    max_training_time: int = Field(
        default=3600,  # 1小时
        env="MAX_TRAINING_TIME"
    )
    
    # CORS 配置
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="ALLOWED_ORIGINS"
    )
    
    @model_validator(mode="before")
    @classmethod
    def parse_list_fields(cls, data: Any) -> Any:
        """解析列表字段，支持 JSON 或逗号分隔的字符串"""
        if isinstance(data, dict):
            # 处理 allowed_extensions
            if "ALLOWED_EXTENSIONS" in data or "allowed_extensions" in data:
                key = "ALLOWED_EXTENSIONS" if "ALLOWED_EXTENSIONS" in data else "allowed_extensions"
                value = data[key]
                if isinstance(value, str) and value:
                    try:
                        data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        # 如果不是 JSON，则按逗号分隔
                        data[key] = [ext.strip() for ext in value.split(",") if ext.strip()]
            
            # 处理 allowed_origins
            if "ALLOWED_ORIGINS" in data or "allowed_origins" in data:
                key = "ALLOWED_ORIGINS" if "ALLOWED_ORIGINS" in data else "allowed_origins"
                value = data[key]
                if isinstance(value, str) and value:
                    try:
                        data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        # 如果不是 JSON，则按逗号分隔
                        data[key] = [origin.strip() for origin in value.split(",") if origin.strip()]
        
        return data
    
    # 日志配置
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[Path] = Field(default=None, env="LOG_FILE")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # 忽略额外的字段
    }


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取应用配置"""
    return settings