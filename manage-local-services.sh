#!/bin/bash

# 本地服务管理脚本（PostgreSQL 和 Redis）

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 添加 PostgreSQL 到 PATH
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"

show_help() {
    echo "本地服务管理脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  start     启动 PostgreSQL 和 Redis 服务"
    echo "  stop      停止 PostgreSQL 和 Redis 服务"
    echo "  restart   重启服务"
    echo "  status    查看服务状态"
    echo "  logs      查看服务日志"
    echo "  test      测试数据库连接"
    echo "  help      显示此帮助信息"
    echo ""
}

start_services() {
    echo -e "${BLUE}🚀 启动本地服务...${NC}"
    echo ""
    
    # 启动 PostgreSQL
    echo -e "${YELLOW}启动 PostgreSQL...${NC}"
    if brew services list | grep -q "postgresql@16.*started"; then
        echo -e "${GREEN}✅ PostgreSQL 已在运行${NC}"
    else
        brew services start postgresql@16
        echo -e "${GREEN}✅ PostgreSQL 已启动${NC}"
        sleep 2
    fi
    
    # 启动 Redis
    echo -e "${YELLOW}启动 Redis...${NC}"
    if brew services list | grep -q "redis.*started"; then
        echo -e "${GREEN}✅ Redis 已在运行${NC}"
    else
        brew services start redis
        echo -e "${GREEN}✅ Redis 已启动${NC}"
        sleep 1
    fi
    
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ 所有服务启动完成！${NC}"
    echo -e "${GREEN}═══════════════════════════════════════${NC}"
    echo ""
    show_status
}

stop_services() {
    echo -e "${YELLOW}🛑 停止本地服务...${NC}"
    echo ""
    
    # 停止 PostgreSQL
    if brew services list | grep -q "postgresql@16.*started"; then
        brew services stop postgresql@16
        echo -e "${GREEN}✅ PostgreSQL 已停止${NC}"
    else
        echo -e "${YELLOW}⚠️  PostgreSQL 未运行${NC}"
    fi
    
    # 停止 Redis
    if brew services list | grep -q "redis.*started"; then
        brew services stop redis
        echo -e "${GREEN}✅ Redis 已停止${NC}"
    else
        echo -e "${YELLOW}⚠️  Redis 未运行${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}✅ 所有服务已停止${NC}"
}

restart_services() {
    echo -e "${BLUE}🔄 重启本地服务...${NC}"
    echo ""
    
    stop_services
    sleep 2
    start_services
}

show_status() {
    echo -e "${BLUE}📊 服务状态：${NC}"
    echo ""
    
    # PostgreSQL 状态
    if brew services list | grep -q "postgresql@16.*started"; then
        echo -e "${GREEN}✅ PostgreSQL: 运行中${NC}"
        export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"
        psql -d charge_analysis -c "SELECT version();" 2>/dev/null | head -1 || echo "  ⚠️  无法连接"
    else
        echo -e "${RED}❌ PostgreSQL: 未运行${NC}"
    fi
    
    # Redis 状态
    if brew services list | grep -q "redis.*started"; then
        echo -e "${GREEN}✅ Redis: 运行中${NC}"
        if redis-cli ping > /dev/null 2>&1; then
            echo "  连接测试: PONG"
        else
            echo "  ⚠️  无法连接"
        fi
    else
        echo -e "${RED}❌ Redis: 未运行${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}详细信息：${NC}"
    brew services list | grep -E "postgresql|redis"
}

test_connections() {
    echo -e "${BLUE}🔍 测试数据库连接...${NC}"
    echo ""
    
    # 测试 PostgreSQL
    echo -e "${YELLOW}测试 PostgreSQL 连接...${NC}"
    export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"
    if psql -d charge_analysis -c "SELECT current_database(), current_user;" 2>/dev/null; then
        echo -e "${GREEN}✅ PostgreSQL 连接成功${NC}"
    else
        echo -e "${RED}❌ PostgreSQL 连接失败${NC}"
    fi
    
    echo ""
    
    # 测试 Redis
    echo -e "${YELLOW}测试 Redis 连接...${NC}"
    if redis-cli ping 2>/dev/null | grep -q "PONG"; then
        echo -e "${GREEN}✅ Redis 连接成功${NC}"
    else
        echo -e "${RED}❌ Redis 连接失败${NC}"
    fi
}

show_logs() {
    echo -e "${BLUE}📋 查看服务日志...${NC}"
    echo ""
    echo "PostgreSQL 日志位置:"
    echo "  /opt/homebrew/var/log/postgresql@16.log"
    echo ""
    echo "Redis 日志位置:"
    echo "  /opt/homebrew/var/log/redis.log"
    echo ""
    echo "实时查看日志:"
    echo "  tail -f /opt/homebrew/var/log/postgresql@16.log"
    echo "  tail -f /opt/homebrew/var/log/redis.log"
}

# 主逻辑
case "${1:-help}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    test)
        test_connections
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}❌ 未知命令: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

