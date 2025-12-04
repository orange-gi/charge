# 部署与配置指南

本指南面向负责部署、运维与扩展 Charge Analysis System 的团队，覆盖从基础设施准备、环境搭建、数据库与存储规划，到监控、安全、备份与升级的完整流程。重点章节（第 6 章）对数据库与持久化存储提供了详细的落地实践，可直接用于生产环境实施。

---

## 1. 文档适用范围

- **目标读者**：DevOps、SRE、后端工程师、安全团队。
- **适用阶段**：PoC、预生产、正式生产与多区域扩展。
- **部署模式**：本地开发、Docker Compose 单节点、Kubernetes 多节点、混合云（数据库托管 + 自建应用）。

---

## 2. 基础环境准备

### 2.1 硬件资源建议

| 组件 | 最低规格 | 推荐规格 | 说明 |
| --- | --- | --- | --- |
| 应用节点（后端+前端） | 4 vCPU / 8 GB RAM / 100 GB SSD | 8 vCPU / 16 GB RAM / 200 GB SSD | 依据并发与模型体积可水平扩展 |
| 数据库节点（PostgreSQL） | 4 vCPU / 16 GB RAM / 200 GB SSD | 8 vCPU / 32 GB RAM / 500 GB NVMe | 需要高 IOPS，详见 6.1.2 |
| 缓存节点（Redis） | 2 vCPU / 4 GB RAM / 50 GB SSD | 4 vCPU / 8 GB RAM / 100 GB SSD | 若启用持久化需双盘冗余 |
| 向量存储（ChromaDB） | 与应用节点共享 | 独立 4 vCPU / 8 GB RAM / 200 GB SSD | 取决于知识库规模 |

### 2.2 软件依赖

- **操作系统**：Ubuntu 20.04+/22.04, Debian 12, CentOS 8 Stream, 或等价云镜像。
- **容器栈**：Docker 20.10+, Docker Compose 2.0+, 或 Kubernetes 1.26+。
- **编译/运行环境**：Node.js 18+、pnpm 8+/npm 9+（前端）；Python 3.10+、Poetry/ pip（后端）。
- **数据库客户端**：psql、pg_dump、redis-cli、wal-g（可选）。

### 2.3 网络与账号

- 申请业务域名，并在 DNS 中创建 `api.<domain>`、`app.<domain>`、`ws.<domain>` 等记录。
- 在云供应商（如 AWS、GCP、阿里云）中预先创建 VPC/VNet、子网、防火墙策略以及 Secrets/Parameter Store 命名空间。
- 准备 Cloud KMS 或 Vault 账户用于托管敏感密钥（JWT、数据库密码、OpenAI Key 等）。

---

## 3. 系统组件拓扑

```
[用户浏览器]
      │
  (Nginx/Ingress)
      │
 ┌────┴────┐
 │         │
前端 SPA   后端 API (FastAPI)
             │││
             ││└─ Redis 7（会话/任务队列）
             │└── ChromaDB（向量检索）
             └── PostgreSQL 15（事务库）
```

- **上游集成**：可选对接 OpenAI API、本地 LLM、对象存储（MinIO/S3）等。
- **监控链路**：Prometheus + Grafana；日志统一写入 Elastic/ Loki 或对象存储。

---

## 4. 部署流程速览

### 4.1 Docker Compose（推荐快速上线）

1. 克隆仓库并进入根目录。
2. 复制 `.env.example` 为 `.env` 并补齐凭证。
3. 运行 `docker compose up -d --build`。
4. 通过 `docker compose ps`、`docker compose logs -f backend` 验证状态。
5. （可选）挂载 SSL 证书给 `nginx` 服务，开放 80/443 端口。

### 4.2 本地开发流程

#### 后端
```bash
cd charge-analysis-backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # 调整 DATABASE_URL 指向本地 PostgreSQL
alembic upgrade head
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 前端
```bash
cd charge-analysis-frontend
pnpm install  # 或 npm install
echo "VITE_API_URL=http://localhost:8000/api/v1" > .env.local
pnpm dev --host 0.0.0.0 --port 3000
```

### 4.3 Kubernetes（生产多副本）

- `k8s/namespace.yaml` 创建命名空间。
- 针对 PostgreSQL/Redis 使用 StatefulSet + PersistentVolume（或托管服务 RDS/MemoryDB）。
- 后端/前端使用 Deployment + HPA，Ingress 控制器（Nginx/Traefik）暴露外网。
- 配置 ConfigMap/Secret 提供环境变量与密钥。

> **提示**：若数据库采用云托管（Aurora、Cloud SQL、PolarDB），需在 VPC 内建立 PrivateLink 或 VPC Peering，避免公网暴露。

---

## 5. 配置与密钥管理

### 5.1 环境变量分层

| 层级 | 文件/位置 | 内容 | 管控建议 |
| --- | --- | --- | --- |
| 全局 | 根目录 `.env` | Docker Compose 统一变量 | 仅存放非敏感配置，敏感项引用 `secret.env` |
| 后端 | `charge-analysis-backend/.env` | 数据库、Redis、JWT、LLM、对象存储 | 生产环境使用 Secret Manager 注入 |
| 前端 | `charge-analysis-frontend/.env.production` | API URL、WebSocket、标题 | 构建前由 CI 写入 |
| Kubernetes | ConfigMap/Secret | 分离配置与密钥 | 使用 `kubectl seal` 或 Vault Operator |

### 5.2 核心变量示例

```bash
# 后端 (.env)
APP_NAME=Charge Analysis Backend
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
DATABASE_URL=postgresql://charge_user:${DB_PASSWORD}@postgres:5432/charge_analysis
REDIS_URL=redis://redis:6379/0
SECRET_KEY=${JWT_SECRET}
BGE_MODEL_NAME=BAAI/bge-base-zh-v1.5
CHROMA_PERSIST_DIRECTORY=/var/lib/charge-analysis/chromadb_data
OPENAI_API_KEY=${OPENAI_API_KEY}
ALLOWED_ORIGINS=["https://app.example.com", "https://admin.example.com"]
```

```bash
# 前端 (.env.production)
VITE_APP_TITLE=Charge Analysis System
VITE_API_URL=https://api.example.com/api/v1
VITE_WS_URL=wss://api.example.com/ws
```

---

## 6. 数据库与持久化存储

### 6.1 PostgreSQL 15（核心事务库）

#### 6.1.1 架构与拓扑

- **单节点**：适合 PoC/小规模，使用本地持久盘或云块存储。
- **主从同步**：1 主 + 至少 1 只读副本（Streaming Replication / Managed Read Replica），提供高可用与查询卸载。
- **高可用组件**：
  - 自建：Patroni + etcd，或 repmgr。
  - 云托管：启用 Multi-AZ（Aurora、Cloud SQL、RDS、PolarDB）。
- **网络**：只接受来自应用子网或指定堡垒机的连接；启用 TLS 加密 (`ssl = on`)；使用安全组限制 5432 端口。

#### 6.1.2 资源与存储规划

| 环境 | 数据量 | 推荐 vCPU/RAM | 存储 | 备注 |
| --- | --- | --- | --- | --- |
| PoC | < 10 GB | 2C / 8 GB | 100 GB SSD | 单 AZ 即可 |
| 生产基础 | 10-200 GB | 4-8C / 16-32 GB | 500 GB NVMe / io2 | IOPS ≥ 6000 |
| 大规模 | > 200 GB | 8-16C / 32-64 GB | 1 TB NVMe + WAL 独立盘 | 建议分离数据与 WAL，开启压缩 |

- 建议开启自动扩容（LVM/云盘），定期评估 `pg_stat_database` 中的 `deadlocks`、`blks_read`。
- WAL 日志持久化至独立磁盘或对象存储（使用 wal-g/wal-e）。

#### 6.1.3 初始化与权限

```sql
-- 1. 创建数据库与角色
CREATE DATABASE charge_analysis OWNER postgres;
CREATE USER charge_user WITH PASSWORD 'REPLACE_ME';
GRANT CONNECT ON DATABASE charge_analysis TO charge_user;

-- 2. 切换到业务库
d \c charge_analysis

-- 3. 授权 schema
CREATE SCHEMA IF NOT EXISTS public AUTHORIZATION charge_user;
GRANT USAGE, CREATE ON SCHEMA public TO charge_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO charge_user;

-- 4. 批准迁移用户
CREATE ROLE alembic LOGIN PASSWORD 'REPLACE_ME' IN ROLE charge_user;
```

> 在生产环境中，推荐：
> - 使用 `pgpass` 或 Secret Manager 注入密码。
> - 将 `alembic` 迁移账号设置更严格权限（仅 DDL）。

#### 6.1.4 参数调优

在 `postgresql.conf` 或 Parameter Group 中设置：

| 参数 | 建议值 | 说明 |
| --- | --- | --- |
| `max_connections` | 200（视连接池大小调整） | 若使用 PgBouncer，可降低到 100 |
| `shared_buffers` | `RAM * 0.25`，如 8 GB | 提升缓存命中率 |
| `effective_cache_size` | `RAM * 0.5`-`0.75` | 优化查询计划 |
| `work_mem` | 32MB-64MB | 复杂查询需更高 |
| `maintenance_work_mem` | 256MB | DDL/索引维护 |
| `wal_level` | `replica`（HA）/`logical`（CDC） | 取决于复制需求 |
| `archive_mode` | on（生产） | 配合 `archive_command` 推送 WAL |
| `checkpoint_timeout` | 15min | 与 `max_wal_size` 配合，减少写放大 |

#### 6.1.5 连接管理与中间件

- **SQLAlchemy/AsyncPG**：在后端启用 `QueuePool`，示例：
  ```python
  engine = create_engine(
      DATABASE_URL,
      pool_size=20,
      max_overflow=5,
      pool_timeout=30,
      pool_pre_ping=True,
  )
  ```
- **PgBouncer**（推荐）
  - 运行在应用同一子网，模式为 `transaction pooling`。
  - 配置：`default_pool_size=50`、`max_client_conn=1000`、`ignore_startup_parameters=extra_float_digits`。
  - 应用层 `DATABASE_URL` 改为 `postgresql://charge_user@pgbouncer:6432/charge_analysis`。

#### 6.1.6 安全策略

- **加密**：开启 TLS，证书可自签或使用内部 CA；客户端连接字符串添加 `?sslmode=require`。
- **最小权限**：应用只授予 CRUD；分析/报表另行创建只读角色。
- **审计**：开启 `pgaudit.log_catalog=on`、`log_statement='ddl'`，日志输出到集中系统。
- **访问控制**：`pg_hba.conf` 中使用 CIDR 白名单与 SCRAM-SHA-256 认证。

#### 6.1.7 迁移与版本管理

- 使用 Alembic：
  ```bash
  alembic revision --autogenerate -m "add training metrics table"
  alembic upgrade head
  ```
- 在 CI/CD 中：
  1. 运行单元测试与静态检查。
  2. 在待发布数据库执行 `alembic upgrade head`。
  3. 若失败回滚：`alembic downgrade -1` 并恢复备份。

#### 6.1.8 备份与恢复

**逻辑备份（每日）**
```bash
#!/bin/bash
set -euo pipefail
DB_NAME="charge_analysis"
BACKUP_DIR="/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump --format=custom --no-owner $DB_NAME \
  | gzip > $BACKUP_DIR/${DB_NAME}_${DATE}.dump.gz
find $BACKUP_DIR -name "${DB_NAME}_*.dump.gz" -mtime +30 -delete
```

**物理/WAL 备份（推荐）**
- 使用 `wal-g backup-push /var/lib/postgresql/data`。
- `wal-g wal-push` 配合 `archive_command` 将 WAL 上传到对象存储（S3/OSS）。
- 恢复时执行 `wal-g backup-fetch` + `wal-g wal-fetch` 实现 Point-In-Time Recovery。

**快速恢复流程**
1. 停止应用写入（切换只读）。
2. 在新实例上还原基础备份。
3. 应用 WAL 到指定时间点。
4. 更新 DNS/连接池指向新实例。
5. 验证 `alembic current`、运行集成自检脚本。

#### 6.1.9 日常维护

| 任务 | 频率 | 命令/说明 |
| --- | --- | --- |
| VACUUM (FULL) 大表 | 每周或大批量删除后 | `VACUUM (FULL, ANALYZE) table_name;` |
| 重建索引 | 每季度 | `REINDEX TABLE ...;` |
| 慢查询分析 | 每日 | `SELECT * FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;` |
| 容量巡检 | 每周 | `
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC LIMIT 20;
` |

### 6.2 Redis 7（缓存 + 任务状态）

- 模式：
  - 开发/小规模：单实例 + AOF。
  - 生产：1 主 2 从 + Sentinel 或云托管（AWS Elasticache, Azure Cache for Redis）。
- `redis.conf` 关键设置：
  ```conf
  bind 0.0.0.0
  protected-mode yes
  requirepass <REPLACE_ME>
  maxmemory 1gb
  maxmemory-policy allkeys-lru
  appendonly yes
  appendfsync everysec
  save 900 1
  save 300 10
  save 60 10000
  ```
- 网络与安全：限制访问源，启用 TLS（若使用 6.0+ enterprise 或云托管）。

### 6.3 ChromaDB（向量存储）

- 推荐将 `CHROMA_PERSIST_DIRECTORY` 挂载到独立持久卷，定期快照（tar + rsync/S3）。
- 大规模知识库可切换到托管向量服务（Pinecone、Milvus、Weaviate）。
- 建议在部署时预热向量索引：
  ```python
  client = chromadb.PersistentClient(path="/var/lib/charge-analysis/chromadb_data")
  collection = client.get_or_create_collection(
      name="charging_knowledge",
      metadata={"description": "充电业务知识库"}
  )
  ```

---

## 7. 应用部署细节

### 7.1 Docker Compose 样例（生产）

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    env_file: .env
    environment:
      POSTGRES_DB: charge_analysis
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  backend:
    build:
      context: ./charge-analysis-backend
    env_file:
      - .env
      - backend.env
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    volumes:
      - uploads_data:/app/uploads
      - chromadb_data:/app/chromadb_data
      - models_data:/app/models
    ports:
      - "8000:8000"

  frontend:
    build:
      context: ./charge-analysis-frontend
    environment:
      - VITE_API_URL=${VITE_API_URL}
    ports:
      - "3000:80"
    depends_on:
      - backend

  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - frontend
      - backend

volumes:
  postgres_data:
  redis_data:
  uploads_data:
  chromadb_data:
  models_data:
```

### 7.2 Kubernetes 关键点

- 使用 `StatefulSet + PersistentVolumeClaim` 部署 PostgreSQL/Redis，并结合 `initContainers` 执行权限修复。
- 后端 Deployment 添加：
  - `readinessProbe`：`httpGet /health`，初始延迟 10s。
  - `livenessProbe`：`httpGet /health`，失败阈值 3。
- 配置 `HorizontalPodAutoscaler`（CPU 60%、内存 70%）。
- Ingress 中开启 `proxy-body-size 100m` 以支持大文件上传。

---

## 8. 监控与日志

### 8.1 日志

- 后端使用 Loguru，将 INFO/ERROR 分流到 `logs/app.log` 与 `logs/error.log`，并通过 Fluent Bit/Vector 派送到集中系统。
- 前端在开发环境输出 `console`，生产仅通过 `/api/v1/logs` 上报，需启用速率限制与鉴权。

```python
# logging_config.py（节选）
logger.add(
    "logs/app.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    enqueue=True,
)
```

### 8.2 指标

- Prometheus 抓取：
  - 后端：`/metrics`（Starlette/Prometheus 中间件）。
  - PostgreSQL：`postgres-exporter:9187`。
  - Redis：`redis-exporter:9121`。
- 关键指标：请求量、延迟、任务队列深度、活动分析数、向量检索耗时、数据库缓存命中率、复制延迟。

---

## 9. 安全与网络

### 9.1 HTTPS / 反向代理

```nginx
server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    add_header Strict-Transport-Security "max-age=63072000" always;

    location / {
        proxy_pass http://frontend:80;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /ws/ {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 9.2 防火墙

- Ubuntu（UFW）
  ```bash
  sudo ufw allow OpenSSH
  sudo ufw allow 80,443/tcp
  sudo ufw allow from 10.0.0.0/8 to any port 5432 proto tcp
  sudo ufw allow from 10.0.0.0/8 to any port 6379 proto tcp
  sudo ufw enable
  ```
- Kubernetes：使用 NetworkPolicy，限制后端 Pod 仅能访问 PostgreSQL/Redis/ChromaDB。

---

## 10. 备份与恢复

### 10.1 PostgreSQL
- 逻辑备份脚本参见 6.1.8。
- 配置 Cron：
  ```cron
  0 2 * * * /opt/scripts/backup_database.sh
  ```

### 10.2 文件与向量库

```bash
#!/bin/bash
BACKUP_DIR="/backups/files"
DATE=$(date +%Y%m%d_%H%M%S)
TEMP_DIR="/tmp/ca_files_$DATE"
mkdir -p "$TEMP_DIR"
cp -r /var/lib/charge-analysis/uploads "$TEMP_DIR"/uploads
cp -r /var/lib/charge-analysis/chromadb_data "$TEMP_DIR"/chromadb
tar -czf $BACKUP_DIR/files_$DATE.tar.gz -C "$TEMP_DIR" .
rm -rf "$TEMP_DIR"
```

### 10.3 自动化

```cron
0 3 * * * /opt/scripts/backup_files.sh
0 4 * * 0 find /backups -name "*.tar.gz" -mtime +90 -delete
```

---

## 11. 故障排除

| 问题 | 诊断步骤 | 解决方案 |
| --- | --- | --- |
| 数据库连接失败 | `psql -h <host> -U <user> -d charge_analysis -c 'SELECT 1'` | 检查安全组、防火墙、`pg_hba.conf`、DNS 是否指向正确实例 |
| 迁移卡住 | 查看 Alembic 日志、`alembic current` | 回滚到上一个版本，修复脚本后重跑 |
| Redis 无法写入 | `redis-cli info persistence` | 检查磁盘空间、AOF 状态，必要时重启 + RDB 恢复 |
| 内存压力 | `free -h`、`top -o %MEM` | 增加 swap、调低并发、扩容实例 |
| Docker 容器频繁重启 | `docker compose logs <service>` | 校验环境变量、依赖服务 readiness、卷权限 |

日志分析：
```bash
tail -n 200 logs/app.log
journalctl -u docker --since "10 min ago"
docker compose logs -f backend
```

性能调优：
```sql
SELECT query, mean_time, calls
FROM pg_stat_statements
ORDER BY mean_time DESC LIMIT 10;
```

---

## 12. 升级流程

```bash
# 1) 备份
./backup_database.sh && ./backup_files.sh

# 2) 停止服务
docker compose down  # 或 kubectl rollout pause

# 3) 获取新版本
git pull origin main

# 4) 更新依赖
(cd charge-analysis-backend && pip install -r requirements.txt)
(cd charge-analysis-frontend && pnpm install)

# 5) 运行迁移
cd charge-analysis-backend && alembic upgrade head

# 6) 重建并启动
cd .. && docker compose up -d --build

# 7) 验证
curl -f https://api.example.com/health
curl -f https://app.example.com
```

在 Kubernetes 中，使用分阶段滚动发布：`kubectl rollout restart deployment/charge-analysis-backend`，观察 `kubectl get pods` 与 `kubectl logs` 确认无错误。

---

## 13. 支持与资源

- **技术支持**：support@charge-analysis.com
- **官方文档**：https://docs.charge-analysis.com
- **GitHub**：https://github.com/your-org/charge-analysis-system
- **社区**：论坛、Slack、Wiki（参考 README 中链接）

---

若在部署过程中遇到文档未覆盖的场景，可将问题和上下文记录在 GitHub Issues 或内部知识库，以便后续版本补充。祝部署顺利！
