# 部署和配置文档

## 1. 环境配置指南

### 1.1 系统要求

#### 硬件要求
- **CPU**: 4核心以上 (推荐8核心)
- **内存**: 8GB以上 (推荐16GB)
- **存储**: 100GB以上可用空间
- **网络**: 稳定的互联网连接

#### 软件要求
- **操作系统**: Ubuntu 20.04+ / CentOS 8+ / Windows 10+ / macOS 11+
- **Docker**: 20.10.0+
- **Docker Compose**: 2.0.0+
- **Node.js**: 18.0+ (开发环境)
- **Python**: 3.10+ (开发环境)

### 1.2 依赖服务

#### 必需的外部服务
1. **PostgreSQL 15**: 主数据库
2. **Redis 7**: 缓存和会话存储
3. **ChromaDB**: 向量数据库
4. **OpenAI API** (可选): 用于LLM分析

#### 可选的服务
1. **Nginx**: 反向代理
2. **Prometheus**: 监控
3. **Grafana**: 监控面板

## 2. 安装指南

### 2.1 后端安装

#### 2.1.1 使用 Docker 安装

```bash
# 克隆项目
git clone https://github.com/your-repo/charge-analysis-system.git
cd charge-analysis-system

# 进入后端目录
cd charge-analysis-backend

# 创建环境配置文件
cp .env.example .env
# 编辑 .env 文件，配置数据库连接等参数

# 构建和启动服务
docker-compose up -d --build
```

#### 2.1.2 本地开发安装

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 创建数据库
createdb charge_analysis

# 运行数据库迁移
alembic upgrade head

# 启动开发服务器
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2.2 前端安装

#### 2.2.1 使用 Docker 安装

```bash
# 进入前端目录
cd charge-analysis-frontend

# 构建 Docker 镜像
docker build -t charge-analysis-frontend .

# 启动前端服务
docker run -d -p 3000:3000 charge-analysis-frontend
```

#### 2.2.2 本地开发安装

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

## 3. 配置说明

### 3.1 环境变量配置

#### 3.1.1 后端环境变量 (.env)

```bash
# 应用配置
APP_NAME=Charge Analysis Backend
APP_VERSION=1.0.0
DEBUG=false
ENVIRONMENT=production

# 服务器配置
HOST=0.0.0.0
PORT=8000

# 数据库配置
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/charge_analysis

# JWT 配置
SECRET_KEY=your-super-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis 配置
REDIS_URL=redis://localhost:6379/0

# 文件上传配置
UPLOAD_PATH=/var/lib/charge-analysis/uploads
MAX_FILE_SIZE=104857600  # 100MB

# ChromaDB 配置
CHROMA_PERSIST_DIRECTORY=/var/lib/charge-analysis/chromadb_data

# 模型配置
BGE_MODEL_NAME=BAAI/bge-base-zh-v1.5
SMALL_MODEL_PATH=/var/lib/charge-analysis/models/1.5b_flow_control_model
LLM_MODEL_PATH=/var/lib/charge-analysis/models/llm_model

# LLM 配置 (可选)
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# CORS 配置
ALLOWED_ORIGINS=["http://localhost:3000", "https://your-domain.com"]

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=/var/log/charge-analysis/app.log
```

#### 3.1.2 前端环境变量

```bash
# .env.local
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000
VITE_APP_TITLE=Charge Analysis System
```

### 3.2 数据库配置

#### 3.2.1 PostgreSQL 配置

```sql
-- 创建数据库
CREATE DATABASE charge_analysis;

-- 创建用户
CREATE USER charge_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE charge_analysis TO charge_user;

-- 配置连接限制
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
```

#### 3.2.2 Redis 配置

```bash
# redis.conf
maxmemory 1gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### 3.3 ChromaDB 配置

```python
# chromadb_config.py
import chromadb

# 创建持久化客户端
client = chromadb.PersistentClient(
    path="/var/lib/charge-analysis/chromadb_data"
)

# 创建集合
collection = client.create_collection(
    name="charging_knowledge",
    metadata={"description": "充电相关知识库"}
)
```

## 4. 部署流程

### 4.1 Docker Compose 部署

#### 4.1.1 生产环境 docker-compose.yml

```yaml
version: '3.8'

services:
  # PostgreSQL 数据库
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: charge_analysis
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - charge-network

  # Redis 缓存
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    networks:
      - charge-network

  # 后端 API 服务
  backend:
    build:
      context: ./charge-analysis-backend
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/charge_analysis
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - uploads_data:/app/uploads
      - chromadb_data:/app/chromadb_data
      - models_data:/app/models
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    networks:
      - charge-network

  # 前端 Web 服务
  frontend:
    build:
      context: ./charge-analysis-frontend
      dockerfile: Dockerfile
    environment:
      - VITE_API_URL=${VITE_API_URL}
    ports:
      - "3000:80"
    depends_on:
      - backend
    restart: unless-stopped
    networks:
      - charge-network

  # Nginx 反向代理
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - frontend
      - backend
    restart: unless-stopped
    networks:
      - charge-network

volumes:
  postgres_data:
  redis_data:
  uploads_data:
  chromadb_data:
  models_data:

networks:
  charge-network:
    driver: bridge
```

#### 4.1.2 启动服务

```bash
# 创建环境变量文件
cat > .env << EOF
POSTGRES_PASSWORD=your_secure_password
SECRET_KEY=your_super_secret_jwt_key
VITE_API_URL=https://your-domain.com/api/v1
EOF

# 启动所有服务
docker-compose up -d

# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs -f backend
```

### 4.2 Kubernetes 部署

#### 4.2.1 部署文件结构

```
k8s/
├── namespace.yaml
├── postgres/
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── deployment.yaml
│   └── service.yaml
├── redis/
│   ├── deployment.yaml
│   └── service.yaml
├── backend/
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
└── frontend/
    ├── deployment.yaml
    ├── service.yaml
    └── ingress.yaml
```

#### 4.2.2 示例后端部署

```yaml
# k8s/backend/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: charge-analysis-backend
  namespace: charge-analysis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: charge-analysis-backend
  template:
    metadata:
      labels:
        app: charge-analysis-backend
    spec:
      containers:
      - name: backend
        image: charge-analysis-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: redis-config
              key: url
        volumeMounts:
        - name: uploads
          mountPath: /app/uploads
        - name: chromadb
          mountPath: /app/chromadb_data
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: uploads
        persistentVolumeClaim:
          claimName: uploads-pvc
      - name: chromadb
        persistentVolumeClaim:
          claimName: chromadb-pvc
```

## 5. 监控和日志

### 5.1 日志配置

#### 5.1.1 后端日志

```python
# logging_config.py
import logging
from loguru import logger
import sys

# 移除默认处理器
logger.remove()

# 添加控制台输出
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# 添加文件输出
logger.add(
    "logs/app.log",
    rotation="1 day",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO"
)

# 添加错误文件输出
logger.add(
    "logs/error.log",
    rotation="1 day",
    retention="90 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR"
)
```

#### 5.1.2 前端日志

```javascript
// logging.js
class Logger {
  static info(message, context = {}) {
    if (import.meta.env.DEV) {
      console.log(`[INFO] ${message}`, context);
    }
    // 发送到后端日志服务
    this.sendLog('INFO', message, context);
  }

  static error(message, error = null, context = {}) {
    if (import.meta.env.DEV) {
      console.error(`[ERROR] ${message}`, error, context);
    }
    this.sendLog('ERROR', message, { error, ...context });
  }

  static async sendLog(level, message, context = {}) {
    try {
      await fetch('/api/v1/logs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          level,
          message,
          context,
          timestamp: new Date().toISOString()
        })
      });
    } catch (error) {
      // 静默失败
    }
  }
}

export default Logger;
```

### 5.2 性能监控

#### 5.2.1 Prometheus 配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'charge-analysis-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

#### 5.2.2 关键指标

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# 请求计数器
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

# 业务指标
ACTIVE_ANALYSES = Gauge('active_analyses_total', 'Number of active analyses')
ANALYSIS_COMPLETION_TIME = Histogram('analysis_completion_seconds', 'Time to complete analysis')
VECTOR_SEARCH_TIME = Histogram('vector_search_seconds', 'Time for vector searches')

# 启动指标服务器
start_http_server(8001)
```

## 6. 安全配置

### 6.1 HTTPS 配置

#### 6.1.1 Nginx SSL 配置

```nginx
# nginx/ssl.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    
    # 安全头
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    location / {
        proxy_pass http://frontend:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /ws/ {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}

# HTTP 重定向到 HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

### 6.2 防火墙配置

```bash
# Ubuntu UFW 配置
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow from 10.0.0.0/8 to any port 5432
sudo ufw allow from 10.0.0.0/8 to any port 6379

# CentOS firewalld 配置
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --permanent --add-rich-rule='rule family="ipv4" source address="10.0.0.0/8" port protocol="tcp" port="5432" accept'
sudo firewall-cmd --permanent --add-rich-rule='rule family="ipv4" source address="10.0.0.0/8" port protocol="tcp" port="6379" accept'
sudo firewall-cmd --reload
```

## 7. 备份和恢复

### 7.1 数据库备份

```bash
#!/bin/bash
# backup_database.sh

DB_NAME="charge_analysis"
DB_USER="postgres"
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份
pg_dump -U $DB_USER -h localhost $DB_NAME > $BACKUP_DIR/backup_$DATE.sql

# 压缩备份
gzip $BACKUP_DIR/backup_$DATE.sql

# 删除30天前的备份
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete

echo "数据库备份完成: backup_$DATE.sql.gz"
```

### 7.2 文件备份

```bash
#!/bin/bash
# backup_files.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
TEMP_DIR="/tmp/backup_$DATE"

# 创建临时目录
mkdir -p $TEMP_DIR

# 备份上传文件
cp -r /var/lib/charge-analysis/uploads $TEMP_DIR/

# 备份 ChromaDB 数据
cp -r /var/lib/charge-analysis/chromadb_data $TEMP_DIR/

# 压缩备份
tar -czf $BACKUP_DIR/files_backup_$DATE.tar.gz -C /tmp backup_$DATE

# 清理临时目录
rm -rf $TEMP_DIR

echo "文件备份完成: files_backup_$DATE.tar.gz"
```

### 7.3 自动化备份

```bash
# 添加到 crontab
# 每天凌晨2点备份数据库
0 2 * * * /path/to/backup_database.sh

# 每天凌晨3点备份文件
0 3 * * * /path/to/backup_files.sh

# 每周清理旧备份
0 4 * * 0 find /backups -name "backup_*.sql.gz" -mtime +90 -delete
```

## 8. 故障排除

### 8.1 常见问题

#### 8.1.1 数据库连接问题

```bash
# 检查数据库状态
sudo systemctl status postgresql

# 检查连接
psql -h localhost -U postgres -d charge_analysis -c "SELECT version();"

# 检查配置
sudo -u postgres psql -c "SHOW max_connections;"
```

#### 8.1.2 Redis 连接问题

```bash
# 检查 Redis 状态
sudo systemctl status redis

# 测试连接
redis-cli ping

# 检查配置
redis-cli config get "*"
```

#### 8.1.3 内存不足

```bash
# 检查内存使用
free -h
df -h

# 检查进程
top -o %MEM

# 调整交换空间
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 8.2 日志分析

```bash
# 查看应用日志
tail -f logs/app.log

# 查看错误日志
tail -f logs/error.log

# 分析日志
grep "ERROR" logs/app.log | head -20

# 查看 Docker 容器日志
docker-compose logs -f backend
```

### 8.3 性能调优

#### 8.3.1 数据库优化

```sql
-- 查看慢查询
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- 分析表大小
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname = 'public'
ORDER BY n_distinct DESC;
```

#### 8.3.2 应用优化

```python
# 启用连接池
from sqlalchemy.pool import QueuePool

engine = create_engine(
    database_url,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True
)

# 启用缓存
from functools import lru_cache
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

@lru_cache(maxsize=100)
def expensive_function(param):
    # 昂贵的计算
    return result
```

## 9. 升级指南

### 9.1 版本升级流程

```bash
# 1. 备份数据
./backup_database.sh
./backup_files.sh

# 2. 停止服务
docker-compose down

# 3. 更新代码
git pull origin main

# 4. 更新依赖
cd charge-analysis-backend
pip install -r requirements.txt --upgrade

cd ../charge-analysis-frontend
npm install

# 5. 运行数据库迁移
alembic upgrade head

# 6. 重新构建和启动
docker-compose up -d --build

# 7. 验证升级
curl -f http://localhost:8000/health
curl -f http://localhost:3000
```

### 9.2 数据库迁移

```bash
# 创建迁移脚本
alembic revision --autogenerate -m "Add new feature"

# 编辑迁移文件
vim versions/abc123_add_feature.py

# 应用迁移
alembic upgrade head

# 回滚迁移（如需要）
alembic downgrade -1
```

## 10. 联系支持

### 10.1 技术支持
- **邮箱**: support@charge-analysis.com
- **文档**: https://docs.charge-analysis.com
- **GitHub**: https://github.com/your-org/charge-analysis-system

### 10.2 社区资源
- **论坛**: https://forum.charge-analysis.com
- **Slack**: https://charge-analysis.slack.com
- **Wiki**: https://wiki.charge-analysis.com

这份部署和配置文档提供了完整的环境配置、部署流程、监控配置和安全设置指南，确保系统能够稳定、安全地运行在生产环境中。