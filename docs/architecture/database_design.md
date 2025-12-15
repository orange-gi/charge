# 数据库设计与本地部署（PostgreSQL）

> 本文档同时覆盖：**数据库设计**与**本地 PostgreSQL 部署/初始化**。
> 
> 以当前后端代码（`charge-analysis-backend/models.py`、`database.py`、`config.py`）为准：启动后端时会在 `startup` 阶段调用 `init_db()`，通过 `SQLAlchemy Base.metadata.create_all()` 自动建表。

---

## 1. 范围与总体架构

### 1.1 数据存储划分

- **PostgreSQL（主库）**：存储用户、分析任务、RAG 文档元数据、训练任务与日志/审计等结构化数据。
- **ChromaDB（向量库）**：存储文档向量与分块内容（持久化目录由 `CHROMA_PERSIST_DIRECTORY` 控制）。
- **Redis（可选缓存/状态）**：用于缓存、短期状态与异步任务协作（本仓库主要通过配置项预留）。

本文重点描述 **PostgreSQL** 的库表设计与本地部署。

### 1.2 关键约定（以代码现状为准）

- **主键**：各表使用自增整型 `id`。
- **时间字段**：多数字段使用 `created_at/updated_at`（以及部分任务 `started_at/completed_at`）。
  - 代码中使用 `DateTime`（不强制时区）；推荐生产环境使用 `timestamptz`，并统一以 UTC 写入。
- **“JSON”字段**：代码中多个字段以 `Text` 存储 JSON 字符串（例如 `result_data`、`meta_info`、`hyperparameters`、`metrics`）。
  - 这意味着数据库层**不会**进行 JSON 结构校验与 JSONB 索引优化。
  - 如需高性能检索/过滤，建议后续演进为 `JSONB`。
- **枚举**：代码中使用 SQLAlchemy `Enum(...)` 绑定 Python `Enum`。
  - 在 PostgreSQL 上会生成对应的 enum 类型（类型名通常由 SQLAlchemy 自动生成，受版本与命名策略影响）。

---

## 2. 连接配置

后端通过 `DATABASE_URL` 连接数据库（见 `charge-analysis-backend/config.py`）：

- **默认值**：`postgresql://postgres:password@localhost:5432/charge_analysis`
- 本仓库 `.env` 示例使用：`postgresql://orange@localhost:5432/charge_analysis`

建议本地开发也使用具备密码的专用账号（见第 5 章）。

---

## 3. 表结构设计（PostgreSQL）

> 说明：下列 DDL 以“业务含义 + 推荐实现”为主；你当前代码使用 `create_all()` 建表时，字段类型（特别是 `Enum`、`Text`/`JSON`）可能与下方“推荐类型”略有差异，但字段名/关系/约束与 `models.py` 一致。

### 3.1 用户与会话

#### 3.1.1 `users`（用户）

- **用途**：系统用户、鉴权与权限控制
- **关键字段**：`username`、`email` 唯一；`role`（user/admin）；`password_hash`

建议表结构：

```sql
CREATE TABLE IF NOT EXISTS users (
  id            SERIAL PRIMARY KEY,
  username      VARCHAR(50)  NOT NULL UNIQUE,
  email         VARCHAR(100) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  first_name    VARCHAR(50),
  last_name     VARCHAR(50),
  role          VARCHAR(20)  NOT NULL DEFAULT 'user',
  is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
  avatar_url    VARCHAR(255),
  last_login    TIMESTAMP,
  created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email    ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role     ON users(role);
```

> 代码实现：`role` 实际为 SQLAlchemy `Enum(UserRole)`；`created_at/updated_at` 使用 `func.now()`。

#### 3.1.2 `user_sessions`（会话）

- **用途**：持久化登录会话（token 的 hash、到期时间、客户端信息）
- **关键约束**：`token_hash` 唯一；`user_id` 外键级联删除

```sql
CREATE TABLE IF NOT EXISTS user_sessions (
  id         SERIAL PRIMARY KEY,
  user_id    INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  token_hash VARCHAR(255) NOT NULL UNIQUE,
  expires_at TIMESTAMP   NOT NULL,
  ip_address VARCHAR(45),
  user_agent TEXT,
  created_at TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id   ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires   ON user_sessions(expires_at);
```

---

### 3.2 充电分析

#### 3.2.1 `charging_analyses`（分析任务）

- **用途**：上传文件后的分析任务元数据、进度、结果与错误
- **状态**：`pending/processing/completed/failed/validation_failed/max_iterations_reached`

```sql
CREATE TABLE IF NOT EXISTS charging_analyses (
  id            SERIAL PRIMARY KEY,
  name          VARCHAR(100) NOT NULL,
  description   TEXT,
  file_path     VARCHAR(500) NOT NULL,
  file_size     BIGINT,
  file_type     VARCHAR(20) DEFAULT 'blf',
  status        VARCHAR(40) NOT NULL DEFAULT 'pending',
  progress      DOUBLE PRECISION DEFAULT 0.0,
  result_data   TEXT,
  error_message TEXT,
  user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  started_at    TIMESTAMP,
  completed_at  TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_charging_analyses_user_id ON charging_analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_charging_analyses_status  ON charging_analyses(status);
CREATE INDEX IF NOT EXISTS idx_charging_analyses_created ON charging_analyses(created_at);
```

> 代码实现：`result_data` 为 `Text`（存 JSON 字符串）；`status` 为 SQLAlchemy `Enum(AnalysisStatus)`。

#### 3.2.2 `analysis_results`（分析结果明细）

- **用途**：将单次分析拆分成多个“结果条目”（类型、标题、内容、置信度、元信息）

```sql
CREATE TABLE IF NOT EXISTS analysis_results (
  id               SERIAL PRIMARY KEY,
  analysis_id       INTEGER NOT NULL REFERENCES charging_analyses(id) ON DELETE CASCADE,
  result_type       VARCHAR(50)  NOT NULL,
  title             VARCHAR(200) NOT NULL,
  content           TEXT,
  confidence_score  DOUBLE PRECISION,
  meta_info         TEXT,
  created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_analysis_results_analysis_id ON analysis_results(analysis_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_type        ON analysis_results(result_type);
```

---

### 3.3 RAG 知识库管理（元数据层）

> 向量与分块内容主要由 ChromaDB 持久化；PostgreSQL 侧管理集合/文档元数据与查询历史。

#### 3.3.1 `knowledge_collections`（知识库集合）

- **用途**：逻辑集合（例如：文档/指南/FAQ）与对应的 Chroma collection id

```sql
CREATE TABLE IF NOT EXISTS knowledge_collections (
  id                  SERIAL PRIMARY KEY,
  name                VARCHAR(100) NOT NULL,
  description         TEXT,
  collection_type     VARCHAR(50)  NOT NULL DEFAULT 'document',
  chroma_collection_id VARCHAR(255) UNIQUE,
  document_count      INTEGER DEFAULT 0,
  embedding_model     VARCHAR(100) DEFAULT 'bge-base-zh-v1.5',
  is_active           BOOLEAN NOT NULL DEFAULT TRUE,
  created_by          INTEGER REFERENCES users(id) ON DELETE SET NULL,
  created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_knowledge_collections_created_by ON knowledge_collections(created_by);
CREATE INDEX IF NOT EXISTS idx_knowledge_collections_active     ON knowledge_collections(is_active);
CREATE INDEX IF NOT EXISTS idx_knowledge_collections_type       ON knowledge_collections(collection_type);
```

> 代码实现：`collection_type` 为 SQLAlchemy `Enum(DocumentType)`。

#### 3.3.2 `knowledge_documents`（知识文档）

- **用途**：文档文件元信息、解析内容、分块数量、处理状态

```sql
CREATE TABLE IF NOT EXISTS knowledge_documents (
  id               SERIAL PRIMARY KEY,
  collection_id    INTEGER NOT NULL REFERENCES knowledge_collections(id) ON DELETE CASCADE,
  filename         VARCHAR(255) NOT NULL,
  file_path        VARCHAR(500) NOT NULL,
  file_size        BIGINT,
  file_type        VARCHAR(50),
  content          TEXT,
  chunk_count      INTEGER DEFAULT 0,
  meta_info        TEXT,
  upload_status    VARCHAR(20) NOT NULL DEFAULT 'uploading',
  processing_error TEXT,
  uploaded_by      INTEGER REFERENCES users(id) ON DELETE SET NULL,
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_knowledge_documents_collection_id ON knowledge_documents(collection_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_uploaded_by   ON knowledge_documents(uploaded_by);
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_status        ON knowledge_documents(upload_status);
```

> 代码实现：`upload_status` 为 SQLAlchemy `Enum(UploadStatus)`。

#### 3.3.3 `rag_queries`（检索历史）

- **用途**：记录一次检索的输入、命中数量、响应文本与耗时

```sql
CREATE TABLE IF NOT EXISTS rag_queries (
  id           SERIAL PRIMARY KEY,
  collection_id INTEGER NOT NULL REFERENCES knowledge_collections(id) ON DELETE CASCADE,
  query_text   TEXT NOT NULL,
  result_count INTEGER DEFAULT 0,
  response_text TEXT,
  user_id      INTEGER REFERENCES users(id) ON DELETE SET NULL,
  query_time_ms INTEGER,
  created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rag_queries_collection_id ON rag_queries(collection_id);
CREATE INDEX IF NOT EXISTS idx_rag_queries_created_at    ON rag_queries(created_at);
```

> 注意：代码中 `RAGQuery` 未显式建立 `user` relationship，但 `user_id` 字段存在。

---

### 3.4 训练管理

#### 3.4.1 `training_datasets`（训练数据集）

```sql
CREATE TABLE IF NOT EXISTS training_datasets (
  id           SERIAL PRIMARY KEY,
  name         VARCHAR(100) NOT NULL,
  description  TEXT,
  dataset_type VARCHAR(50) DEFAULT 'standard',
  file_path    VARCHAR(500),
  sample_count INTEGER DEFAULT 0,
  meta_info    TEXT,
  is_public    BOOLEAN NOT NULL DEFAULT FALSE,
  created_by   INTEGER REFERENCES users(id) ON DELETE SET NULL,
  created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_datasets_created_by ON training_datasets(created_by);
CREATE INDEX IF NOT EXISTS idx_training_datasets_public     ON training_datasets(is_public);
CREATE INDEX IF NOT EXISTS idx_training_datasets_type       ON training_datasets(dataset_type);
```

#### 3.4.2 `training_configs`（训练配置预设）

- **用途**：可复用的训练配置（基座模型、适配器类型、超参等）

```sql
CREATE TABLE IF NOT EXISTS training_configs (
  id              SERIAL PRIMARY KEY,
  name            VARCHAR(100) NOT NULL,
  base_model      VARCHAR(100) NOT NULL,
  model_path      VARCHAR(500) NOT NULL,
  adapter_type    VARCHAR(50)  NOT NULL DEFAULT 'lora',
  model_size      VARCHAR(20)  NOT NULL DEFAULT '1.5b',
  dataset_strategy VARCHAR(50) DEFAULT 'full',
  hyperparameters TEXT,
  notes           TEXT,
  created_by      INTEGER REFERENCES users(id) ON DELETE SET NULL,
  created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_configs_created_by ON training_configs(created_by);
```

#### 3.4.3 `model_versions`（模型版本）

```sql
CREATE TABLE IF NOT EXISTS model_versions (
  id          SERIAL PRIMARY KEY,
  name        VARCHAR(100) NOT NULL,
  model_type  VARCHAR(50) DEFAULT 'flow_control',
  version     VARCHAR(50) NOT NULL,
  model_path  VARCHAR(500) NOT NULL,
  config      TEXT,
  metrics     TEXT,
  is_active   BOOLEAN NOT NULL DEFAULT FALSE,
  is_default  BOOLEAN NOT NULL DEFAULT FALSE,
  created_by  INTEGER REFERENCES users(id) ON DELETE SET NULL,
  created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_versions_type    ON model_versions(model_type);
CREATE INDEX IF NOT EXISTS idx_model_versions_active  ON model_versions(is_active);
CREATE INDEX IF NOT EXISTS idx_model_versions_default ON model_versions(is_default);
```

#### 3.4.4 `training_tasks`（训练任务）

- **用途**：一次训练任务的生命周期、进度、日志与结果汇总
- **状态**：`pending/queued/running/completed/failed/cancelled`

```sql
CREATE TABLE IF NOT EXISTS training_tasks (
  id              SERIAL PRIMARY KEY,
  name            VARCHAR(100) NOT NULL,
  description     TEXT,
  dataset_id      INTEGER REFERENCES training_datasets(id) ON DELETE SET NULL,
  config_id       INTEGER REFERENCES training_configs(id) ON DELETE SET NULL,
  model_version_id INTEGER REFERENCES model_versions(id) ON DELETE SET NULL,
  model_type      VARCHAR(50) NOT NULL,
  adapter_type    VARCHAR(50) DEFAULT 'lora',
  model_size      VARCHAR(20) DEFAULT '1.5b',
  hyperparameters TEXT,
  status          VARCHAR(20) NOT NULL DEFAULT 'pending',
  progress        DOUBLE PRECISION DEFAULT 0.0,
  current_epoch   INTEGER DEFAULT 0,
  total_epochs    INTEGER,
  current_step    INTEGER DEFAULT 0,
  total_steps     INTEGER,
  metrics         TEXT,
  logs            TEXT,
  error_message   TEXT,
  start_time      TIMESTAMP,
  end_time        TIMESTAMP,
  duration_seconds INTEGER,
  gpu_memory_usage TEXT,
  model_path      VARCHAR(500),
  created_by      INTEGER REFERENCES users(id) ON DELETE SET NULL,
  created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_tasks_dataset_id ON training_tasks(dataset_id);
CREATE INDEX IF NOT EXISTS idx_training_tasks_config_id  ON training_tasks(config_id);
CREATE INDEX IF NOT EXISTS idx_training_tasks_status     ON training_tasks(status);
CREATE INDEX IF NOT EXISTS idx_training_tasks_created_at ON training_tasks(created_at);
```

#### 3.4.5 `training_metrics`（训练指标历史）

```sql
CREATE TABLE IF NOT EXISTS training_metrics (
  id            SERIAL PRIMARY KEY,
  task_id       INTEGER NOT NULL REFERENCES training_tasks(id) ON DELETE CASCADE,
  epoch         INTEGER NOT NULL,
  step          INTEGER NOT NULL,
  loss          DOUBLE PRECISION,
  accuracy      DOUBLE PRECISION,
  learning_rate DOUBLE PRECISION,
  gpu_memory    DOUBLE PRECISION,
  custom_metrics TEXT,
  created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_metrics_task_id ON training_metrics(task_id);
CREATE INDEX IF NOT EXISTS idx_training_metrics_epoch   ON training_metrics(epoch);
CREATE INDEX IF NOT EXISTS idx_training_metrics_step    ON training_metrics(step);
```

#### 3.4.6 `training_logs`（训练日志记录）

```sql
CREATE TABLE IF NOT EXISTS training_logs (
  id        SERIAL PRIMARY KEY,
  task_id   INTEGER NOT NULL REFERENCES training_tasks(id) ON DELETE CASCADE,
  log_level VARCHAR(20) NOT NULL DEFAULT 'INFO',
  message   TEXT NOT NULL,
  meta_info TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_logs_task_id    ON training_logs(task_id);
CREATE INDEX IF NOT EXISTS idx_training_logs_created_at ON training_logs(created_at);
```

#### 3.4.7 `training_evaluations`（训练评估记录）

```sql
CREATE TABLE IF NOT EXISTS training_evaluations (
  id              SERIAL PRIMARY KEY,
  task_id          INTEGER NOT NULL REFERENCES training_tasks(id) ON DELETE CASCADE,
  evaluator        VARCHAR(100),
  evaluation_type  VARCHAR(50) DEFAULT 'automatic',
  metrics          TEXT,
  recommended_plan VARCHAR(100),
  notes            TEXT,
  created_by       INTEGER REFERENCES users(id) ON DELETE SET NULL,
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_evaluations_task_id ON training_evaluations(task_id);
```

---

### 3.5 系统日志与审计

#### 3.5.1 `system_logs`（系统日志）

```sql
CREATE TABLE IF NOT EXISTS system_logs (
  id            SERIAL PRIMARY KEY,
  level         VARCHAR(20) NOT NULL,
  module        VARCHAR(50) NOT NULL,
  message       TEXT NOT NULL,
  logger_name   VARCHAR(100),
  function_name VARCHAR(100),
  line_number   INTEGER,
  file_path     VARCHAR(255),
  user_id       INTEGER REFERENCES users(id) ON DELETE SET NULL,
  request_id    VARCHAR(100),
  session_id    VARCHAR(100),
  meta_info     TEXT,
  ip_address    VARCHAR(45),
  user_agent    TEXT,
  timestamp     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_system_logs_level     ON system_logs(level);
CREATE INDEX IF NOT EXISTS idx_system_logs_module    ON system_logs(module);
CREATE INDEX IF NOT EXISTS idx_system_logs_user_id   ON system_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp);
```

#### 3.5.2 `audit_logs`（操作审计）

```sql
CREATE TABLE IF NOT EXISTS audit_logs (
  id            SERIAL PRIMARY KEY,
  action        VARCHAR(50) NOT NULL,
  resource_type VARCHAR(50) NOT NULL,
  resource_id   VARCHAR(100),
  old_values    TEXT,
  new_values    TEXT,
  user_id       INTEGER REFERENCES users(id) ON DELETE SET NULL,
  ip_address    VARCHAR(45),
  user_agent    TEXT,
  timestamp     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id      ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource_type ON audit_logs(resource_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action       ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp    ON audit_logs(timestamp);
```

---

## 4. 主要关系（ER 概览）

```
users (1) -> (N) user_sessions
users (1) -> (N) charging_analyses -> (N) analysis_results
users (1) -> (N) knowledge_collections -> (N) knowledge_documents
knowledge_collections (1) -> (N) rag_queries
users (1) -> (N) training_datasets -> (N) training_tasks
users (1) -> (N) training_configs  -> (N) training_tasks
users (1) -> (N) model_versions    -> (N) training_tasks
training_tasks (1) -> (N) training_metrics
training_tasks (1) -> (N) training_logs
training_tasks (1) -> (1) training_evaluations
users (1) -> (N) system_logs
users (1) -> (N) audit_logs
```

---

## 5. 本地 PostgreSQL 部署/初始化（你已安装 Postgres）

以下步骤默认你可以执行：`sudo -u postgres psql`。

### 5.1 创建数据库与账号（推荐：专用账号 + 密码）

1) 进入 psql：

```bash
sudo -u postgres psql
```

2) 在 psql 中执行（把密码替换成你自己的）：

```sql
-- 1) 创建专用账号
CREATE ROLE charge_analysis WITH LOGIN PASSWORD 'please_change_me';

-- 2) 创建数据库并指定 owner
CREATE DATABASE charge_analysis OWNER charge_analysis;

-- 3) （可选）限制公共 schema 权限，更安全
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT  CREATE ON SCHEMA public TO charge_analysis;
```

3) 退出：

```sql
\q
```

4) 配置后端连接串（`charge-analysis-backend/.env`）：

```env
DATABASE_URL=postgresql://charge_analysis:please_change_me@localhost:5432/charge_analysis
```

### 5.2 备选：使用本机系统用户（peer/免密）

如果你希望像示例 `.env` 那样免密（例如 `postgresql://orange@localhost:5432/charge_analysis`），需要 PostgreSQL 的认证方式允许本机用户通过 peer/trust 登录。这个配置依赖你的系统 `pg_hba.conf`，不同发行版路径不同；生产环境不建议使用 trust。

### 5.3 初始化建表（本项目的真实方式）

后端在启动时会自动建表：`charge-analysis-backend/main.py` 的 `startup` 事件里调用 `init_db()`。

你可以用两种方式初始化：

- **方式 A（推荐）**：直接启动后端一次，自动建表。
- **方式 B（手动）**：在后端目录加载 `.env` 后运行一段 Python，显式执行建表：

```bash
python -c "from database import init_db; init_db(); print('ok')"
```

> 注意：方式 B 要在能导入到 `charge-analysis-backend` 包/模块的工作目录执行，并确保 `DATABASE_URL` 已正确设置。

### 5.4 验证建表是否成功

```bash
sudo -u postgres psql -d charge_analysis -c "\dt"
```

应能看到：
- `users`、`user_sessions`
- `charging_analyses`、`analysis_results`
- `knowledge_collections`、`knowledge_documents`、`rag_queries`
- `training_datasets`、`training_configs`、`training_tasks`、`training_metrics`、`training_logs`、`training_evaluations`
- `system_logs`、`audit_logs`

### 5.5 备份与恢复（本地）

- **备份**（自定义格式，便于压缩与选择性恢复）：

```bash
pg_dump -Fc -h localhost -U charge_analysis -d charge_analysis -f charge_analysis.dump
```

- **恢复**：

```bash
createdb -h localhost -U charge_analysis charge_analysis_restored
pg_restore -h localhost -U charge_analysis -d charge_analysis_restored charge_analysis.dump
```

---

## 6. 生产化建议（不改变现有代码的前提下）

- **不要依赖 `create_all()` 做 schema 演进**：它只负责“缺表就建”，不会安全地做字段变更、索引变更与数据迁移。
- **启用迁移工具**：本项目依赖里已包含 `alembic`，建议补齐迁移目录与版本管理流程。
- **JSON 字段演进为 JSONB**：例如 `result_data/meta_info/hyperparameters/metrics` 等，便于校验与索引（GIN）。
- **统一时间类型为 `timestamptz`**：避免跨时区环境的时间歧义。

---

## 7. 常见问题（Troubleshooting）

- **连接被拒绝**：确认 PostgreSQL 在监听 `localhost:5432`（systemd: `systemctl status postgresql`）。
- **认证失败**：检查 `DATABASE_URL` 的用户/密码与 `pg_hba.conf` 规则是否匹配。
- **枚举类型冲突**：若曾手动创建过同名 enum 类型，可能导致 `create_all()` 失败；建议清理旧 schema 或用迁移工具统一管理。
