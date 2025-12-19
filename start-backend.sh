#!/bin/bash

# 后端启动脚本

set -e

cd "$(dirname "$0")/charge-analysis-backend"

echo "🚀 启动后端服务..."

# 使用 Python 3.12
PYTHON_CMD="python3.12"
if ! command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "⚠️  警告: python3.12 未找到，使用 python3"
fi

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "📦 使用 Python 3.12 创建虚拟环境..."
    $PYTHON_CMD -m venv .venv
fi

# 兜底：如果 .venv 目录存在但激活脚本缺失（例如 venv 创建中断），重建虚拟环境
if [ ! -f ".venv/bin/activate" ]; then
    echo "⚠️  检测到虚拟环境不完整，正在重建 .venv..."
    rm -rf .venv
    $PYTHON_CMD -m venv .venv
fi

# 激活虚拟环境
echo "🔌 激活虚拟环境..."
source .venv/bin/activate

# 升级 pip（使用阿里云镜像源）
echo "⬆️  升级 pip..."
pip install --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 安装依赖（使用阿里云镜像源）
echo "📥 安装依赖（使用阿里云镜像源）..."
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 检查数据库连接
echo "🔍 检查数据库连接..."
if ! python3 -c "from database import engine; engine.connect()" 2>/dev/null; then
    echo "⚠️  警告: 无法连接到数据库，请确保 PostgreSQL 服务已启动"
    echo "   运行: brew services start postgresql@16"
fi

# 初始化数据库
echo "🗄️  初始化数据库..."
python3 -c "from database import init_db; init_db()"

# 启动服务
echo "🎯 启动 FastAPI 服务..."
echo "   访问地址: http://0.0.0.0:8000（本机访问可用 http://127.0.0.1:8000）"
echo "   API 文档: http://0.0.0.0:8000/docs（本机访问可用 http://127.0.0.1:8000/docs）"
echo ""
uvicorn main:app --reload --host 0.0.0.0 --port 8000

