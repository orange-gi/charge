#!/bin/bash

# 查看所有服务状态的脚本

cd "$(dirname "$0")"

echo "📊 Charge Analysis System 服务状态"
echo "=================================="
echo ""

# 数据库服务
echo "🗄️  数据库服务："
if psql -U postgres -c "SELECT 1" > /dev/null 2>&1; then
    echo "  ✅ PostgreSQL 运行中"
else
    echo "  ❌ PostgreSQL 未运行"
fi

if redis-cli ping > /dev/null 2>&1; then
    echo "  ✅ Redis 运行中"
else
    echo "  ❌ Redis 未运行"
fi
echo ""

# 后端服务
echo "🔧 后端服务："
if [ -f "backend.pid" ]; then
    BACKEND_PID=$(cat backend.pid)
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        echo "  ✅ 运行中 (PID: $BACKEND_PID)"
        if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
            echo "  ✅ 健康检查通过"
        else
            echo "  ⚠️  健康检查失败"
        fi
    else
        echo "  ❌ 未运行"
    fi
else
    echo "  ❌ 未启动"
fi
echo ""

# 前端服务
echo "🎨 前端服务："
if [ -f "frontend.pid" ]; then
    FRONTEND_PID=$(cat frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo "  ✅ 运行中 (PID: $FRONTEND_PID)"
        if curl -s http://localhost:3000 > /dev/null 2>&1; then
            echo "  ✅ 可访问"
        else
            echo "  ⚠️  无法访问"
        fi
    else
        echo "  ❌ 未运行"
    fi
else
    echo "  ❌ 未启动"
fi
echo ""

# 访问地址
echo "🌐 访问地址："
echo "  - 前端: http://localhost:3000"
echo "  - 后端: http://127.0.0.1:8000"
echo "  - API 文档: http://127.0.0.1:8000/docs"

