# 智能充电分析系统 (Intelligent Charging Analysis System)

一个基于 LangGraph 和 React 的智能充电数据分析系统，提供充电数据分析、RAG 查询、模型训练等功能。

## 项目结构

```
.
├── charge-analysis-backend/     # 后端服务（Python + LangGraph）
├── charge-analysis-frontend/    # 前端应用（React + TypeScript）
├── supabase/                    # Supabase 配置和函数
│   ├── functions/               # Edge Functions
│   ├── migrations/              # 数据库迁移
│   └── tables/                  # 数据库表定义
├── docs/                        # 项目文档
└── test_data/                   # 测试数据

```

## 技术栈

### 后端
- Python 3.x
- LangGraph
- Supabase Edge Functions

### 前端
- React 18
- TypeScript
- Vite
- Ant Design
- Zustand

### 数据库
- Supabase (PostgreSQL)

## 功能特性

- 🔋 充电数据分析
- 📊 数据可视化
- 🤖 RAG 智能查询
- 🎓 模型训练管理
- 📝 知识库管理
- 👤 用户认证与授权

## 快速开始

### 后端设置

```bash
cd charge-analysis-backend
pip install -r requirements.txt
```

### 前端设置

```bash
cd charge-analysis-frontend
pnpm install
pnpm dev
```

## 文档

详细文档请查看 [docs/](./docs/) 目录。

## 许可证

MIT License

