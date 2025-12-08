# 技术选型和依赖

## 1. 前端技术栈详细说明

### 1.1 核心框架
```json
{
  "name": "charge-analysis-frontend",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.8.0",
    "typescript": "^5.0.0"
  }
}
```

### 1.2 状态管理 - Zustand
```json
{
  "dependencies": {
    "zustand": "^4.3.2"
  }
}
```

**选择理由**:
- 轻量级，API 简单
- TypeScript 支持良好
- 不需要 Provider 包装
- 支持中间件和异步操作

### 1.3 UI 组件库 - Ant Design
```json
{
  "dependencies": {
    "antd": "^5.2.0",
    "@ant-design/icons": "^5.0.0"
  }
}
```

**选择理由**:
- 完整的组件生态系统
- 优秀的 TypeScript 支持
- 丰富的企业级组件
- 良好的可定制性

### 1.4 HTTP 客户端 - Axios
```json
{
  "dependencies": {
    "axios": "^1.3.0"
  }
}
```

**选择理由**:
- 请求/响应拦截
- 请求取消
- 请求进度追踪
- 错误处理机制

### 1.5 图表库 - Plotly.js
```json
{
  "dependencies": {
    "plotly.js-dist": "^2.18.0",
    "react-plotly.js": "^2.6.0"
  }
}
```

**选择理由**:
- 丰富的图表类型
- 良好的交互性
- 支持大数据量渲染
- 与现有代码兼容

### 1.6 实时通信 - Socket.io
```json
{
  "dependencies": {
    "socket.io-client": "^4.6.0"
  }
}
```

**选择理由**:
- 双向通信
- 自动重连
- 房间管理
- 事件驱动

## 2. 后端技术栈详细说明

### 2.1 Web 框架 - FastAPI
```python
fastapi==0.100.0
uvicorn[standard]==0.22.0
```

**选择理由**:
- 高性能 ASGI 框架
- 自动 API 文档生成
- 类型注解支持
- 内置数据验证

### 2.2 Agent 框架 - LangChain + LangGraph
```python
langchain==0.1.0
langchain-community==0.0.6
langgraph==0.0.45
```

**选择理由**:
- 现代化的 Agent 框架
- 图形化工作流定义
- 状态管理机制
- 节点间数据传递

### 2.3 向量数据库 - ChromaDB
```python
chromadb==0.4.8
```

**选择理由**:
- 轻量级嵌入式数据库
- 支持多种嵌入模型
- 简单 API 设计
- Python 原生支持

### 2.4 嵌入模型 - BGE-Base-ZH-v1.5
```python
sentence-transformers==2.2.2
torch==2.0.0
```

**选择理由**:
- 中文语义理解能力强
- 模型大小适中
- 社区支持好
- 性能稳定

### 2.5 数据库 ORM - SQLAlchemy
```python
sqlalchemy[asyncio]==2.0.0
aiosqlite==0.19.0
alembic==1.10.0
```

**选择理由**:
- 成熟的 ORM 框架
- 异步支持
- 数据库迁移工具
- 类型安全

### 2.6 身份认证 - FastAPI-Users
```python
fastapi-users==12.0.0
fastapi-users-db-sqlalchemy==11.0.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
```

**选择理由**:
- 完整的用户认证解决方案
- JWT 支持
- 密码加密
- 用户权限管理

### 2.7 数据处理 - Pandas + NumPy
```python
pandas==2.0.0
numpy==1.24.0
scipy==1.10.0
```

**选择理由**:
- 强大的数据处理能力
- 与现有代码兼容
- 丰富的统计函数
- 高性能计算

### 2.8 机器学习 - PyTorch + Transformers
```python
torch==2.0.0
transformers==4.27.0
accelerate==0.20.0
```

**选择理由**:
- 1.5B 模型训练支持
- 丰富的预训练模型
- 良好的 GPU 支持
- 社区活跃

### 2.9 日志 - Loguru
```python
loguru==0.7.0
structlog==23.0.0
```

**选择理由**:
- 简单易用的日志库
- 结构化日志支持
- 自动日志轮转
- 与标准库兼容

### 2.10 异步任务 - Celery
```python
celery==5.3.0
redis==4.5.0
```

**选择理由**:
- 分布式任务队列
- 异步训练任务
- 任务监控和调度
- 良好的扩展性

## 3. 数据库设计

### 3.1 PostgreSQL (主数据库)
```sql
-- 用户表
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 训练任务表
CREATE TABLE training_tasks (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    dataset_id INTEGER,
    model_type VARCHAR(50),
    hyperparameters JSONB,
    status VARCHAR(20) DEFAULT 'pending',
    progress DECIMAL(5,2) DEFAULT 0,
    logs TEXT,
    model_path VARCHAR(255),
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 日志表
CREATE TABLE system_logs (
    id SERIAL PRIMARY KEY,
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    user_id INTEGER REFERENCES users(id),
    module VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3.2 ChromaDB (向量数据库)
```python
# RAG 知识库集合
collections = {
    "charging_knowledge": {
        "documents": "充电相关文档内容",
        "metadatas": {
            "source": "文档来源",
            "type": "文档类型",
            "uploaded_by": "上传用户",
            "tags": ["充电", "故障", "诊断"]
        }
    }
}
```

## 4. API 接口设计

### 4.1 RESTful API 结构
```
/api/v1/
├── auth/
│   ├── POST /login
│   ├── POST /register
│   ├── POST /logout
│   └── GET /me
├── charging/
│   ├── POST /upload
│   ├── POST /analyze
│   ├── GET /history
│   └── GET /results/{id}
├── rag/
│   ├── GET /collections
│   ├── POST /upload
│   ├── POST /query
│   └── DELETE /collections/{id}
├── training/
│   ├── GET /tasks
│   ├── POST /tasks
│   ├── GET /tasks/{id}
│   ├── POST /datasets
│   └── GET /datasets
└── logs/
    ├── GET /
    ├── GET /export
    └── GET /statistics
```

### 4.2 WebSocket 接口
```
/ws/
├── /ws/analysis/{session_id} - 充电分析实时进度
├── /ws/training/{task_id} - 训练任务实时状态
└── /ws/notifications - 系统通知
```

## 5. 配置文件

### 5.1 环境配置 (.env)
```bash
# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/charge_analysis
REDIS_URL=redis://localhost:6379

# JWT 配置
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# 文件上传配置
UPLOAD_PATH=./uploads
MAX_FILE_SIZE=100MB
ALLOWED_EXTENSIONS=.blf,.csv,.xlsx

# ChromaDB 配置
CHROMA_PERSIST_DIRECTORY=./chromadb_data
CHROMA_COLLECTION_NAME=charging_knowledge

# 模型配置
BGE_MODEL_NAME=sentence-transformers/BAAI/bge-base-zh-v1.5
SMALL_MODEL_PATH=./models/1.5b_flow_control_model
LLM_MODEL_PATH=./models/llm_model

# 训练配置
TRAINING_WORKERS=4
MAX_TRAINING_TIME=3600
```


## 6. 开发工具链

### 6.1 开发环境工具
```json
{
  "devDependencies": {
    "@types/react": "^18.0.0",
    "@types/react-dom": "^18.0.0",
    "@vitejs/plugin-react": "^3.1.0",
    "vite": "^4.1.0",
    "eslint": "^8.34.0",
    "@typescript-eslint/eslint-plugin": "^5.52.0",
    "prettier": "^2.8.4",
    "typescript": "^4.9.5"
  }
}
```

### 6.2 代码质量工具
- **ESLint**: 代码静态分析
- **Prettier**: 代码格式化
- **Husky**: Git hooks
- **CommitLint**: 提交信息规范

### 6.3 测试工具
```json
{
  "devDependencies": {
    "@testing-library/react": "^13.4.0",
    "@testing-library/jest-dom": "^5.16.5",
    "jest": "^29.4.1",
    "cypress": "^12.6.0"
  }
}
```

## 7. 部署架构

### 7.1 生产环境部署
```nginx
# nginx.conf
server {
    listen 80;
    server_name your-domain.com;
    
    # 前端静态文件
    location / {
        root /var/www/charge-analysis/frontend;
        try_files $uri $uri/ /index.html;
    }
    
    # 后端 API 代理
    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # WebSocket 代理
    location /ws/ {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 7.2 监控和日志
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'charge-analysis-backend'
    static_configs:
      - targets: ['backend:8000']
```

## 8. 性能优化策略

### 8.1 前端优化
- 代码分割和懒加载
- 组件缓存策略
- 图片懒加载和压缩
- Bundle 大小优化

### 8.2 后端优化
- 数据库查询优化
- Redis 缓存策略
- 异步处理任务
- API 响应压缩

### 8.3 数据库优化
- 索引优化
- 连接池配置
- 查询语句优化
- 数据分区策略