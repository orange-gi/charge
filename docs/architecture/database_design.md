# 数据库设计

## 1. 数据库架构概述

本系统使用多数据库架构：
- **PostgreSQL**: 主关系数据库，存储用户数据、训练任务、系统日志等
- **ChromaDB**: 向量数据库，存储 RAG 知识库和文档嵌入向量
- **Redis**: 缓存数据库，存储会话、训练状态等临时数据

## 2. PostgreSQL 数据库设计

### 2.1 用户认证模块

#### 用户表 (users)
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    is_active BOOLEAN DEFAULT true,
    avatar_url VARCHAR(255),
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_created_at ON users(created_at);
```

#### 用户会话表 (user_sessions)
```sql
CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_expires ON user_sessions(expires_at);
```

### 2.2 充电分析模块

#### 充电分析任务表 (charging_analyses)
```sql
CREATE TABLE charging_analyses (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    file_path VARCHAR(255) NOT NULL,
    file_size BIGINT,
    file_type VARCHAR(20) DEFAULT 'blf',
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    progress DECIMAL(5,2) DEFAULT 0,
    result_data JSONB,
    error_message TEXT,
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- 索引
CREATE INDEX idx_charging_analyses_user_id ON charging_analyses(user_id);
CREATE INDEX idx_charging_analyses_status ON charging_analyses(status);
CREATE INDEX idx_charging_analyses_created_at ON charging_analyses(created_at);
```

#### 分析结果详情表 (analysis_results)
```sql
CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    analysis_id INTEGER REFERENCES charging_analyses(id) ON DELETE CASCADE,
    result_type VARCHAR(50) NOT NULL,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    confidence_score DECIMAL(5,4),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_analysis_results_analysis_id ON analysis_results(analysis_id);
CREATE INDEX idx_analysis_results_type ON analysis_results(result_type);
```

### 2.3 RAG 管理模块

#### 知识库集合表 (knowledge_collections)
```sql
CREATE TABLE knowledge_collections (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    collection_type VARCHAR(50) DEFAULT 'document',
    chroma_collection_id VARCHAR(255) UNIQUE,
    document_count INTEGER DEFAULT 0,
    embedding_model VARCHAR(100) DEFAULT 'bge-base-zh-v1.5',
    is_active BOOLEAN DEFAULT true,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_knowledge_collections_created_by ON knowledge_collections(created_by);
CREATE INDEX idx_knowledge_collections_type ON knowledge_collections(collection_type);
CREATE INDEX idx_knowledge_collections_active ON knowledge_collections(is_active);
```

#### 知识文档表 (knowledge_documents)
```sql
CREATE TABLE knowledge_documents (
    id SERIAL PRIMARY KEY,
    collection_id INTEGER REFERENCES knowledge_collections(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT,
    file_type VARCHAR(50),
    content TEXT,
    chunk_count INTEGER DEFAULT 0,
    metadata JSONB,
    upload_status VARCHAR(20) DEFAULT 'uploading' CHECK (upload_status IN ('uploading', 'processing', 'completed', 'failed')),
    processing_error TEXT,
    uploaded_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_knowledge_documents_collection_id ON knowledge_documents(collection_id);
CREATE INDEX idx_knowledge_documents_uploaded_by ON knowledge_documents(uploaded_by);
CREATE INDEX idx_knowledge_documents_status ON knowledge_documents(upload_status);
```

#### 检索历史表 (rag_queries)
```sql
CREATE TABLE rag_queries (
    id SERIAL PRIMARY KEY,
    collection_id INTEGER REFERENCES knowledge_collections(id),
    query_text TEXT NOT NULL,
    result_count INTEGER DEFAULT 0,
    response_text TEXT,
    user_id INTEGER REFERENCES users(id),
    query_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_rag_queries_collection_id ON rag_queries(collection_id);
CREATE INDEX idx_rag_queries_user_id ON rag_queries(user_id);
CREATE INDEX idx_rag_queries_created_at ON rag_queries(created_at);
```

### 2.4 训练管理模块

#### 训练数据集表 (training_datasets)
```sql
CREATE TABLE training_datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    dataset_type VARCHAR(50) DEFAULT 'standard' CHECK (dataset_type IN ('standard', 'chain_of_thought')),
    file_path VARCHAR(500),
    sample_count INTEGER DEFAULT 0,
    metadata JSONB,
    is_public BOOLEAN DEFAULT false,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_training_datasets_created_by ON training_datasets(created_by);
CREATE INDEX idx_training_datasets_type ON training_datasets(dataset_type);
CREATE INDEX idx_training_datasets_public ON training_datasets(is_public);
```

#### 模型版本表 (model_versions)
```sql
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) DEFAULT 'flow_control' CHECK (model_type IN ('flow_control', 'llm')),
    version VARCHAR(50) NOT NULL,
    model_path VARCHAR(500) NOT NULL,
    config JSONB,
    metrics JSONB,
    is_active BOOLEAN DEFAULT false,
    is_default BOOLEAN DEFAULT false,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_model_versions_type ON model_versions(model_type);
CREATE INDEX idx_model_versions_active ON model_versions(is_active);
CREATE INDEX idx_model_versions_default ON model_versions(is_default);
```

#### 训练任务表 (training_tasks)
```sql
CREATE TABLE training_tasks (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    dataset_id INTEGER REFERENCES training_datasets(id),
    model_version_id INTEGER REFERENCES model_versions(id),
    model_type VARCHAR(50) NOT NULL,
    hyperparameters JSONB,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'queued', 'running', 'completed', 'failed', 'cancelled')),
    progress DECIMAL(5,2) DEFAULT 0,
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER,
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER,
    metrics JSONB,
    logs TEXT,
    error_message TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds INTEGER,
    gpu_memory_usage JSONB,
    model_path VARCHAR(500),
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_training_tasks_dataset_id ON training_tasks(dataset_id);
CREATE INDEX idx_training_tasks_created_by ON training_tasks(created_by);
CREATE INDEX idx_training_tasks_status ON training_tasks(status);
CREATE INDEX idx_training_tasks_created_at ON training_tasks(created_at);
```

#### 训练指标历史表 (training_metrics)
```sql
CREATE TABLE training_metrics (
    id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES training_tasks(id) ON DELETE CASCADE,
    epoch INTEGER NOT NULL,
    step INTEGER NOT NULL,
    loss DECIMAL(10,6),
    accuracy DECIMAL(5,4),
    learning_rate DECIMAL(10,8),
    gpu_memory DECIMAL(10,2),
    custom_metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_training_metrics_task_id ON training_metrics(task_id);
CREATE INDEX idx_training_metrics_epoch ON training_metrics(epoch);
CREATE INDEX idx_training_metrics_step ON training_metrics(step);
```

### 2.5 日志管理模块

#### 系统日志表 (system_logs)
```sql
CREATE TABLE system_logs (
    id SERIAL PRIMARY KEY,
    level VARCHAR(20) NOT NULL CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    module VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    logger_name VARCHAR(100),
    function_name VARCHAR(100),
    line_number INTEGER,
    file_path VARCHAR(255),
    user_id INTEGER REFERENCES users(id),
    request_id VARCHAR(100),
    session_id VARCHAR(100),
    metadata JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_system_logs_level ON system_logs(level);
CREATE INDEX idx_system_logs_module ON system_logs(module);
CREATE INDEX idx_system_logs_user_id ON system_logs(user_id);
CREATE INDEX idx_system_logs_timestamp ON system_logs(timestamp);
CREATE INDEX idx_system_logs_level_timestamp ON system_logs(level, timestamp);
```

#### 操作审计表 (audit_logs)
```sql
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    old_values JSONB,
    new_values JSONB,
    user_id INTEGER REFERENCES users(id),
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_resource_type ON audit_logs(resource_type);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);
```

## 3. 数据关系图

```
users (1) -----> (M) user_sessions
users (1) -----> (M) charging_analyses
users (1) -----> (M) knowledge_collections
users (1) -----> (M) knowledge_documents
users (1) -----> (M) rag_queries
users (1) -----> (M) training_datasets
users (1) -----> (M) model_versions
users (1) -----> (M) training_tasks
users (1) -----> (M) system_logs
users (1) -----> (M) audit_logs

knowledge_collections (1) -----> (M) knowledge_documents
knowledge_collections (1) -----> (M) rag_queries
training_datasets (1) -----> (M) training_tasks
model_versions (1) -----> (M) training_tasks
charging_analyses (1) -----> (M) analysis_results
training_tasks (1) -----> (M) training_metrics
```

## 4. 数据安全策略

### 4.1 数据加密
```sql
-- 密码哈希（使用 bcrypt）
ALTER TABLE users ADD COLUMN password_salt VARCHAR(255);
ALTER TABLE users DROP COLUMN password_hash;
ALTER TABLE users ADD COLUMN password_hash VARCHAR(255);

-- 敏感字段加密
ALTER TABLE user_sessions ADD COLUMN token_salt VARCHAR(255);
```

### 4.2 数据权限
```sql
-- 行级安全策略
ALTER TABLE charging_analyses ENABLE ROW LEVEL SECURITY;
CREATE POLICY charging_analyses_policy ON charging_analyses
    FOR ALL TO authenticated
    USING (user_id = current_setting('app.current_user_id')::int);

ALTER TABLE training_tasks ENABLE ROW LEVEL SECURITY;
CREATE POLICY training_tasks_policy ON training_tasks
    FOR ALL TO authenticated
    USING (created_by = current_setting('app.current_user_id')::int);
```

### 4.3 数据备份策略
```sql
-- 定期备份脚本
#!/bin/bash
# backup.sh
PGPASSWORD=$DB_PASSWORD pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME \
    --format=custom --compress=9 \
    --file=backup_$(date +%Y%m%d_%H%M%S).dump
```

## 5. ChromaDB 设计

### 5.1 集合设计
```python
# ChromaDB 集合配置
COLLECTIONS = {
    "charging_knowledge": {
        "metadata": {
            "description": "充电相关知识库",
            "embedding_model": "bge-base-zh-v1.5",
            "created_by": "system"
        }
    },
    "troubleshooting_guides": {
        "metadata": {
            "description": "故障排除指南",
            "embedding_model": "bge-base-zh-v1.5",
            "created_by": "system"
        }
    },
    "technical_specs": {
        "metadata": {
            "description": "技术规格文档",
            "embedding_model": "bge-base-zh-v1.5",
            "created_by": "system"
        }
    }
}
```

### 5.2 文档分块策略
```python
# 文档分块配置
CHUNK_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separator": "\n\n",
    "metadata_fields": [
        "source", "type", "category", "tags", 
        "created_date", "author", "version"
    ]
}
```

### 5.3 检索配置
```python
# 检索配置
RETRIEVAL_CONFIG = {
    "top_k": 5,
    "score_threshold": 0.7,
    "include_metadata": True,
    "include_documents": True,
    "filter_by_metadata": {
        "category": "charging",
        "language": "zh"
    }
}
```

## 6. Redis 设计

### 6.1 缓存策略
```python
# Redis 键值设计
CACHE_KEYS = {
    "user_sessions": "user:session:{user_id}",
    "analysis_progress": "analysis:progress:{analysis_id}",
    "training_status": "training:status:{task_id}",
    "system_stats": "system:stats",
    "rag_cache": "rag:cache:{query_hash}",
    "model_cache": "model:cache:{model_id}"
}

# TTL 配置
TTL_CONFIG = {
    "user_sessions": 3600,  # 1小时
    "analysis_progress": 1800,  # 30分钟
    "training_status": 300,  # 5分钟
    "rag_cache": 600,  # 10分钟
    "model_cache": 3600  # 1小时
}
```

### 6.2 消息队列
```python
# 任务队列配置
QUEUES = {
    "analysis_queue": {
        "name": "charging_analysis",
        "max_workers": 4,
        "timeout": 3600
    },
    "training_queue": {
        "name": "model_training", 
        "max_workers": 2,
        "timeout": 7200
    },
    "rag_queue": {
        "name": "document_processing",
        "max_workers": 6,
        "timeout": 1800
    }
}
```

## 7. 数据迁移和版本控制

### 7.1 Alembic 迁移脚本
```python
# migrations/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

config = context.config
fileConfig(config.config_file_name)

target_metadata = None  # 设置为你的模型元数据

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    engine = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with engine.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()
```

### 7.2 数据版本迁移示例
```python
# migrations/001_initial_schema.py
def upgrade():
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=20), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )

def downgrade():
    op.drop_table('users')
```

## 8. 性能优化

### 8.1 数据库优化
```sql
-- 分区表（按时间分区）
CREATE TABLE system_logs_2024 PARTITION OF system_logs
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- 复合索引
CREATE INDEX idx_training_tasks_status_created ON training_tasks(status, created_at);
CREATE INDEX idx_logs_level_timestamp ON system_logs(level, timestamp);

-- 查询优化
EXPLAIN ANALYZE SELECT * FROM training_tasks 
WHERE status = 'running' AND created_by = 1;
```

### 8.2 缓存优化
```python
# 缓存预热
def warmup_cache():
    # 预加载常用数据
    cache.set('system_stats', get_system_stats(), ttl=300)
    cache.set('active_users', get_active_users(), ttl=1800)
    
# 缓存失效策略
def invalidate_cache(pattern):
    keys = redis.keys(pattern)
    if keys:
        redis.delete(*keys)
```

## 9. 监控和维护

### 9.1 数据库监控
```sql
-- 慢查询监控
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;

-- 连接监控
SELECT count(*) as active_connections
FROM pg_stat_activity 
WHERE state = 'active';
```

### 9.2 定期维护任务
```python
# maintenance.py
def cleanup_expired_sessions():
    """清理过期会话"""
    pass

def optimize_database():
    """数据库优化"""
    pass

def archive_old_logs():
    """归档旧日志"""
    pass

def update_statistics():
    """更新统计信息"""
    pass
```