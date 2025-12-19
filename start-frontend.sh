#!/bin/bash

# 前端启动脚本

set -e

cd "$(dirname "$0")/charge-analysis-frontend"

echo "🚀 启动前端服务..."

# 检查 Node 版本（Vite 5 需要 Node >= 18）
if command -v node &> /dev/null; then
    NODE_VERSION="$(node -v | sed 's/^v//')"
    NODE_MAJOR="$(echo "$NODE_VERSION" | cut -d. -f1)"
    if [ "$NODE_MAJOR" -lt 18 ]; then
        echo "❌ Node 版本过低：v$NODE_VERSION（Vite 5 需要 Node >= 18）"
        echo "   解决方案：升级 Node（推荐 18 LTS 或 20 LTS），例如："
        echo "   - 使用 nvm：nvm install 18 && nvm use 18"
        echo "   - 或安装 nodejs 18/20 发行版"
        exit 1
    fi
else
    echo "❌ 未找到 node，请先安装 Node >= 18"
    exit 1
fi

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
echo "   访问地址: http://0.0.0.0:3000（本机访问可用 http://127.0.0.1:3000）"
echo ""
pnpm dev --host 0.0.0.0 --port 3000

