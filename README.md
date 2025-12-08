# 智能充电分析系统 (Intelligent Charging Analysis System)

一个基于 LangGraph 和 React 的智能充电数据分析系统，提供充电数据分析、RAG 查询、模型训练等功能。

## 项目结构

```
.
├── charge-analysis-backend/     # 后端服务（Python + LangGraph）
├── charge-analysis-frontend/    # 前端应用（React + TypeScript）
├── docs/                        # 项目文档
└── test_data/                   # 测试数据

```

## 技术栈

### 后端
- Python 3.x
- FastAPI + SQLAlchemy
- LangGraph

### 前端
- React 18
- TypeScript
- Vite
- Ant Design
- Zustand

### 数据库
- PostgreSQL

## 功能特性

- 🔋 充电数据分析
- 📊 数据可视化
- 🤖 RAG 智能查询
- 🎓 模型训练管理
- 📝 知识库管理
- 👤 用户认证与授权

## 快速开始

> 📖 **详细启动指南**: 查看 [QUICK_START.md](./QUICK_START.md) 了解完整的启动步骤。

### 数据库服务安装

项目支持两种方式运行 PostgreSQL 和 Redis：

#### 方式 1: 本地安装（推荐）

使用 Homebrew 在本地安装并运行服务：

```bash
# 安装 PostgreSQL 和 Redis
brew install postgresql@16 redis

# 启动服务
brew services start postgresql@16
brew services start redis

# 或使用管理脚本
./manage-local-services.sh start
```

**服务配置：**
- PostgreSQL: `localhost:5432`
  - 数据库: `charge_analysis`
  - 用户: `orange` (系统用户，无需密码)
- Redis: `localhost:6379`

> 📖 **详细配置**: 查看 [LOCAL_SETUP.md](./LOCAL_SETUP.md) 了解本地安装和配置详情。

### 后端设置

```bash
cd charge-analysis-backend
pip install -r requirements.txt

# 初始化数据库（在数据库服务启动后）
python -c "from database import init_db; init_db()"
```

### 前端设置

```bash
cd charge-analysis-frontend
pnpm install
pnpm dev
```

## 文档

- 📚 [快速启动指南](./QUICK_START.md) - 完整的本地开发环境搭建指南
- 🖥️ [本地服务安装指南](./LOCAL_SETUP.md) - 使用 Homebrew 安装 PostgreSQL 和 Redis
- 📦 [Pip 镜像源配置](./PIP_MIRROR_SETUP.md) - 配置 pip 使用国内镜像源加速下载
- 📖 [部署指南](./docs/deployment/deployment_guide.md) - Windows 本地部署指南
- 📋 更多文档请查看 [docs/](./docs/) 目录

## 加速配置

### Pip 镜像源

项目已配置使用阿里云镜像源加速 Python 包下载：

```bash
# 运行配置脚本（如果还没配置）
./setup-pip-mirror.sh

# 验证配置
pip config list
```

详细配置说明请查看 [PIP_MIRROR_SETUP.md](./PIP_MIRROR_SETUP.md)。

### Python 版本

项目使用 **Python 3.12** 作为默认版本：

```bash
# 检查当前 Python 版本
python3 --version

# 如果版本不对，重新加载配置
source ~/.zshrc
```

项目已配置为全局使用 Python 3.12，所有启动脚本会自动使用 Python 3.12。

详细配置说明请查看 [PYTHON312_SETUP.md](./PYTHON312_SETUP.md)。

## 许可证

MIT License

