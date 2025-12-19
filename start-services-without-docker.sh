#!/bin/bash
# 不使用 Docker 启动服务的脚本（适用于已有本地数据库的情况）

echo "🚀 启动后端和前端服务（不使用 Docker）..."
echo ""
echo "⚠️  注意：此脚本假设你已有 PostgreSQL 和 Redis 在本地运行"
echo ""

cd "$(dirname "$0")"

# 启动后端
cd charge-analysis-backend

# 使用 Python 3.12
PYTHON_CMD="python3.12"
if ! command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "⚠️  警告: python3.12 未找到，使用 python3"
fi

if [ ! -d ".venv" ]; then
    echo "📦 使用 Python 3.12 创建虚拟环境..."
    $PYTHON_CMD -m venv .venv
fi

echo "🔌 激活虚拟环境..."
source .venv/bin/activate

if [ ! -f ".venv/bin/uvicorn" ]; then
    echo "📥 安装后端依赖（使用阿里云镜像源）..."
    pip install --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
fi

echo "🗄️  初始化数据库..."
python3 -c "from database import init_db; init_db()" || echo "⚠️  数据库初始化失败，请检查数据库连接"

echo "🚀 启动后端服务..."
nohup uvicorn main:app --reload --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../backend.pid
echo "✅ 后端已启动 (PID: $BACKEND_PID)"

sleep 3

# 启动前端
cd ../charge-analysis-frontend

if [ ! -d "node_modules" ]; then
    echo "📥 安装前端依赖..."
    pnpm install
fi

echo "🚀 启动前端服务..."
nohup pnpm dev --host 0.0.0.0 --port 3000 > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../frontend.pid
echo "✅ 前端已启动 (PID: $FRONTEND_PID)"

echo ""
echo "🎉 服务启动完成！"
echo "  - 前端: http://localhost:3000"
echo "  - 后端: http://127.0.0.1:8000（远程访问请用 http://<服务器IP>:8000）"
