#!/bin/bash

# 启动所有服务的综合脚本

set -e

echo "🚀 启动 Charge Analysis System 所有服务..."
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查数据库服务
echo "📦 检查数据库服务..."
cd "$(dirname "$0")"

# 检查 PostgreSQL
if ! psql -U postgres -c "SELECT 1" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  PostgreSQL 可能未运行，请确保服务已启动${NC}"
    echo "   可以使用: brew services start postgresql@16"
else
    echo -e "${GREEN}✅ PostgreSQL 正在运行${NC}"
fi

# 检查 Redis
if ! redis-cli ping > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Redis 可能未运行，请确保服务已启动${NC}"
    echo "   可以使用: brew services start redis"
else
    echo -e "${GREEN}✅ Redis 正在运行${NC}"
fi

echo ""

# 检查后端环境
echo "🔧 检查后端环境..."
cd charge-analysis-backend

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  后端 .env 文件不存在，正在创建...${NC}"
    cat > .env << 'ENVEOF'
DEBUG=true
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000
RELOAD=true
DATABASE_URL=postgresql://postgres:password@localhost:5432/charge_analysis
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key-here-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=30
UPLOAD_PATH=uploads
MAX_FILE_SIZE=104857600
ALLOWED_EXTENSIONS=.blf,.csv,.xlsx,.pdf,.doc,.docx,.txt
CHROMA_PERSIST_DIRECTORY=chromadb_data
CHROMA_COLLECTION_NAME=charging_knowledge
BGE_MODEL_NAME=BAAI/bge-base-zh-v1.5
SMALL_MODEL_PATH=models/1.5b_flow_control_model
LLM_MODEL_PATH=models/llm_model
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
TRAINING_WORKERS=4
MAX_TRAINING_TIME=3600
# 远程部署建议把这里改成你的前端地址，例如：
# ALLOWED_ORIGINS=http://<服务器IP>:3000
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
LOG_LEVEL=INFO
LOG_FILE=
ENVEOF
    echo -e "${GREEN}✅ .env 文件已创建${NC}"
fi

# 使用 Python 3.12
PYTHON_CMD="python3.12"
if ! command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3"
    echo -e "${YELLOW}⚠️  警告: python3.12 未找到，使用 python3${NC}"
fi

# 设置虚拟环境
if [ ! -d ".venv" ]; then
    echo "📦 使用 Python 3.12 创建虚拟环境..."
    $PYTHON_CMD -m venv .venv
fi

echo "🔌 激活虚拟环境..."
source .venv/bin/activate

# 安装依赖（如果需要）
if [ ! -f ".venv/bin/uvicorn" ]; then
    echo "📥 安装后端依赖（使用阿里云镜像源）..."
    pip install --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com > /dev/null 2>&1
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com > /dev/null 2>&1 || {
        echo -e "${YELLOW}⚠️  依赖安装可能需要一些时间，继续启动...${NC}"
    }
fi

# 初始化数据库
echo "🗄️  初始化数据库..."
python3 -c "from database import init_db; init_db()" 2>/dev/null || {
    echo -e "${YELLOW}⚠️  数据库初始化失败，可能服务未就绪，将在后台继续尝试...${NC}"
}

# 启动后端（后台）
echo "🚀 启动后端服务..."
nohup uvicorn main:app --reload --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../backend.pid
echo -e "${GREEN}✅ 后端服务已启动 (PID: $BACKEND_PID)${NC}"
echo "   日志文件: backend.log"
echo ""

# 等待后端启动
sleep 3

# 检查前端环境
echo "🎨 检查前端环境..."
cd ../charge-analysis-frontend

if [ ! -f ".env.local" ]; then
    echo -e "${YELLOW}⚠️  前端 .env.local 文件不存在，正在创建...${NC}"
    # 留空让前端自动推断：当前 hostname + :8000（远程部署更安全）
    echo "VITE_API_BASE_URL=" > .env.local
    echo -e "${GREEN}✅ .env.local 文件已创建${NC}"
fi

# 检查 pnpm
if ! command -v pnpm &> /dev/null; then
    echo "📦 安装 pnpm..."
    npm install -g pnpm
fi

# 安装依赖（如果需要）
if [ ! -d "node_modules" ]; then
    echo "📥 安装前端依赖..."
    pnpm install > /dev/null 2>&1 || {
        echo -e "${YELLOW}⚠️  前端依赖安装可能需要一些时间...${NC}"
    }
fi

# 启动前端（后台）
echo "🚀 启动前端服务..."
nohup pnpm dev --host 0.0.0.0 --port 3000 > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../frontend.pid
echo -e "${GREEN}✅ 前端服务已启动 (PID: $FRONTEND_PID)${NC}"
echo "   日志文件: frontend.log"
echo ""

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 5

# 检查服务状态
echo "📊 服务状态检查："
echo ""

# 检查后端
if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ 后端服务运行正常${NC}"
else
    echo -e "${YELLOW}⚠️  后端服务可能还在启动中...${NC}"
fi

# 检查前端
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ 前端服务运行正常${NC}"
else
    echo -e "${YELLOW}⚠️  前端服务可能还在启动中...${NC}"
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 所有服务启动完成！${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""
echo "📱 访问地址："
echo "   - 前端: http://localhost:3000"
echo "   - 后端 API: http://127.0.0.1:8000（远程访问请用 http://<服务器IP>:8000）"
echo "   - API 文档: http://127.0.0.1:8000/docs（远程访问请用 http://<服务器IP>:8000/docs）"
echo ""
echo "📋 管理命令："
echo "   - 查看后端日志: tail -f backend.log"
echo "   - 查看前端日志: tail -f frontend.log"
echo "   - 停止所有服务: ./stop-all.sh"
echo "   - 查看服务状态: ./status.sh"
echo ""
echo "🛑 停止服务："
echo "   按 Ctrl+C 或运行 ./stop-all.sh"
echo ""

