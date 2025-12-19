# 快速启动指南

本指南将帮助你快速启动 Charge Analysis System。

## 前置要求

- Python 3.10+ 
- Node.js 18+
- pnpm（可通过 `npm install -g pnpm` 安装）
- PostgreSQL 和 Redis 已安装并运行（推荐使用本地安装，参考 [LOCAL_SETUP.md](./LOCAL_SETUP.md)）

## 步骤 1: 启动数据库服务（PostgreSQL + Redis）

确保 PostgreSQL 和 Redis 服务已启动。如果使用 Homebrew 安装：

```bash
# 启动 PostgreSQL
brew services start postgresql@16

# 启动 Redis
brew services start redis
```

验证服务运行状态：

```bash
# 检查 PostgreSQL
psql -U postgres -c "SELECT version();"

# 检查 Redis
redis-cli ping
```

## 步骤 2: 配置环境变量

### 后端环境变量

在 `charge-analysis-backend/` 目录下创建 `.env` 文件：

```bash
cd charge-analysis-backend
cat > .env << 'EOF'
# 应用配置
DEBUG=true
ENVIRONMENT=development
HOST=127.0.0.1
PORT=8000
RELOAD=true

# 数据库配置
DATABASE_URL=postgresql://postgres:password@localhost:5432/charge_analysis

# Redis 配置
REDIS_URL=redis://localhost:6379/0

# JWT 配置
SECRET_KEY=your-secret-key-here-change-in-production-please-replace-this
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=30

# 文件上传配置
UPLOAD_PATH=uploads
MAX_FILE_SIZE=104857600
ALLOWED_EXTENSIONS=.blf,.csv,.xlsx,.pdf,.doc,.docx,.txt

# ChromaDB 配置
CHROMA_PERSIST_DIRECTORY=chromadb_data
CHROMA_COLLECTION_NAME=charging_knowledge

# 模型配置
BGE_MODEL_NAME=BAAI/bge-base-zh-v1.5
SMALL_MODEL_PATH=models/1.5b_flow_control_model
LLM_MODEL_PATH=models/llm_model

# LLM 配置
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1

# 训练配置
TRAINING_WORKERS=4
MAX_TRAINING_TIME=3600

# CORS 配置
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=
EOF
```

### 前端环境变量

在 `charge-analysis-frontend/` 目录下创建 `.env.local` 文件：

```bash
cd ../charge-analysis-frontend
cat > .env.local << 'EOF'
# 前端环境变量
VITE_API_BASE_URL=http://localhost:8000
EOF
```

## 步骤 3: 启动后端服务

在新终端窗口中：

```bash
# 方法 1: 使用启动脚本
./start-backend.sh

# 方法 2: 手动启动
cd charge-analysis-backend
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python3 -c "from database import init_db; init_db()"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

后端服务启动后：
- API 地址: http://127.0.0.1:8000（远程访问请用 http://<服务器IP>:8000）
- API 文档: http://127.0.0.1:8000/docs（远程访问请用 http://<服务器IP>:8000/docs）
- 健康检查: http://127.0.0.1:8000/health（远程访问请用 http://<服务器IP>:8000/health）

## 步骤 4: 启动前端服务

在新终端窗口中：

```bash
# 方法 1: 使用启动脚本
./start-frontend.sh

# 方法 2: 手动启动
cd charge-analysis-frontend
pnpm install
pnpm dev --host 0.0.0.0 --port 3000
```

前端服务启动后访问: http://localhost:3000

## 验证安装

1. **检查后端**：
   ```bash
   curl http://127.0.0.1:8000/health
   ```
   应该返回: `{"status":"ok"}`

2. **检查数据库连接**：
   ```bash
   psql -U postgres -d charge_analysis -c "SELECT version();"
   ```

3. **检查 Redis**：
   ```bash
   redis-cli ping
   ```
   应该返回: `PONG`

## 常用命令

### 数据库操作

```bash
# 进入 PostgreSQL
psql -U postgres -d charge_analysis

# 备份数据库
pg_dump -U postgres charge_analysis > backup.sql

# 查看 Redis
redis-cli
```

## 故障排查

### 端口被占用

如果端口 5432、6379、8000 或 3000 被占用：

1. **PostgreSQL (5432)**:
   ```bash
   lsof -i :5432
   # 或修改 PostgreSQL/Redis 配置文件中的端口
   ```

2. **Redis (6379)**:
   ```bash
   lsof -i :6379
   ```

3. **后端 (8000)**:
   修改 `charge-analysis-backend/.env` 中的 `PORT=8000`

4. **前端 (3000)**:
   修改 `vite.config.ts` 中的 `server.port`

### 数据库连接失败

1. 确保 PostgreSQL 和 Redis 服务正在运行：
   ```bash
   # 检查 PostgreSQL
   brew services list | grep postgresql
   
   # 检查 Redis
   brew services list | grep redis
   ```

2. 验证连接字符串是否正确

### 前端无法连接后端

1. 确保后端服务正在运行
2. 检查 `.env.local` 中的 `VITE_API_BASE_URL` 是否正确
3. 检查浏览器控制台是否有 CORS 错误（应该已在后端配置）

## 停止所有服务

```bash
# 停止后端 (Ctrl+C)
# 停止前端 (Ctrl+C)

# 停止数据库服务（如果需要）
brew services stop postgresql@16
brew services stop redis
```

## 下一步

- 查看 [部署指南](docs/deployment/deployment_guide.md) 了解详细配置
- 查看 [本地服务安装指南](./LOCAL_SETUP.md) 了解本地安装详情
- 查看 [API 文档](docs/architecture/api_design.md) 了解 API 设计

