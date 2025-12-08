#!/bin/bash

# 停止所有服务的脚本

echo "🛑 停止所有服务..."

cd "$(dirname "$0")"

# 停止后端
if [ -f "backend.pid" ]; then
    BACKEND_PID=$(cat backend.pid)
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        echo "停止后端服务 (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        rm backend.pid
        echo "✅ 后端服务已停止"
    else
        rm backend.pid
    fi
fi

# 停止前端
if [ -f "frontend.pid" ]; then
    FRONTEND_PID=$(cat frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo "停止前端服务 (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        rm frontend.pid
        echo "✅ 前端服务已停止"
    else
        rm frontend.pid
    fi
fi

echo ""
echo "✅ 所有服务已停止"
echo ""
echo "注意：PostgreSQL 和 Redis 服务仍在运行"
echo "如需停止数据库服务，请运行："
echo "  brew services stop postgresql@16"
echo "  brew services stop redis"

