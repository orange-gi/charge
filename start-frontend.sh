#!/bin/bash

# 前端启动脚本

set -e

cd "$(dirname "$0")/charge-analysis-frontend"

echo "🚀 启动前端服务..."

# 检查 pnpm 是否安装
if ! command -v pnpm &> /dev/null; then
    echo "📦 安装 pnpm..."
    npm install -g pnpm
fi

# 安装依赖
if [ ! -d "node_modules" ]; then
    echo "📥 安装依赖..."
    pnpm install
fi

# 启动开发服务器
echo "🎯 启动前端开发服务器..."
echo "   访问地址: http://localhost:3000"
echo ""
pnpm dev --host 127.0.0.1 --port 3000

