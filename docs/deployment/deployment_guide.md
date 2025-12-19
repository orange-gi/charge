# Windows 本地部署指南（Git Bash）

本指南专为在 Windows 10/11 上使用 Git Bash 的开发者编写，目标是在本地机器上运行 Charge Analysis System（后端 FastAPI + 前端 React）。不涉及 Kubernetes 或云端部署，只聚焦"我想在 Windows 上把项目跑起来"这一场景。

---

## 1. 适用范围

- **读者**：需要在个人或团队 Windows 电脑上调试/演示系统的开发者、数据分析师、售前工程师。
- **环境**：Windows 10/11（专业版或家庭版皆可），Shell 使用 Git Bash。
- **输出**：后端 API 运行在 `http://127.0.0.1:8000`，前端运行在 `http://127.0.0.1:3000`。

---

## 2. 必备依赖

| 组件 | 版本建议 | 安装提示 |
| --- | --- | --- |
| Git for Windows | 2.40+ | 安装时勾选 *"Git Bash Here"* 与 *"Use bundled OpenSSH"* |
| Git Bash | 随 Git 安装 | 作为默认 Shell 运行本文全部命令 |
| Python | 3.10-3.11 | 勾选 "Add python.exe to PATH"，建议使用官方 installer 或 Microsoft Store |
| Node.js | 18 LTS | 会自动附带 npm；若已安装 nvm-windows，可通过 `nvm install 18` 管理 |
| pnpm | 8.x | `npm install -g pnpm`，Git Bash 中可直接使用 |
| PostgreSQL | 15.x | 推荐使用 EnterpriseDB 安装包 |
| Redis | 7.x | 可使用 Memurai、grokzen 的 Windows 端口 |

> **版本检查**：在 Git Bash 中执行以下命令，确认环境正常：
>
> ```bash
> git --version
> python --version
> node --version
> pnpm --version
> psql --version     # 若使用本地 PostgreSQL 客户端
> redis-cli --version  # 若使用本地 Redis 客户端
> ```

---

## 3. 克隆仓库并初始设置

```bash
cd /c/workspace    # 任选目录
git clone https://github.com/your-org/charge-analysis-system.git
cd charge-analysis-system
```

推荐在 Windows 资源管理器中右键根目录选择 *"Git Bash Here"*，下文命令均以此目录为根。

---

## 4. 基础服务（PostgreSQL / Redis / 本地存储）

### 4.1 PostgreSQL 准备

#### 方案 A：使用 Windows 安装包（推荐）
1. 从 <https://www.postgresql.org/download/windows/> 下载 15.x 安装包。
2. 安装时记录超级用户密码，启用 `psql` 命令行工具。
3. 安装完成后，在 Git Bash 中执行：
   ```bash
   psql -U postgres -h 127.0.0.1 -c "CREATE DATABASE charge_analysis;"
   psql -U postgres -h 127.0.0.1 -d charge_analysis -c "CREATE USER charge_user WITH PASSWORD 'REPLACE_ME';"
   psql -U postgres -h 127.0.0.1 -d charge_analysis -c "GRANT ALL PRIVILEGES ON DATABASE charge_analysis TO charge_user;"
   ```

### 4.2 Redis 准备

- **原生方案**：安装 Memurai Community 版或 <https://github.com/tporadowski/redis/releases> 中的 Windows 构建，启动后监听 `localhost:6379`。

### 4.3 Chroma/上传目录

创建本地持久目录（用于后端配置 `CHROMA_PERSIST_DIRECTORY` 与文件上传）：
```bash
mkdir -p /c/charge-data/chromadb
mkdir -p /c/charge-data/uploads
```

记录路径，稍后写入后端 `.env`。

---

## 5. 环境变量与密钥

### 5.1 根目录公共配置

复制示例文件：
```bash
cp charge-analysis-backend/.env.example charge-analysis-backend/.env
cp charge-analysis-frontend/.env.example charge-analysis-frontend/.env.local 2>/dev/null || true
```

### 5.2 后端 `.env` 关键项

```bash
APP_NAME="Charge Analysis Backend"
ENVIRONMENT=local
HOST=127.0.0.1
PORT=8000
DATABASE_URL=postgresql+psycopg://charge_user:REPLACE_ME@127.0.0.1:5432/charge_analysis
REDIS_URL=redis://127.0.0.1:6379/0
SECRET_KEY=请替换为随机字符串
BGE_MODEL_NAME=BAAI/bge-base-zh-v1.5
CHROMA_PERSIST_DIRECTORY=/c/charge-data/chromadb
UPLOAD_DIR=/c/charge-data/uploads
OPENAI_API_KEY=可选
```

> Git Bash 支持以 `/c/...` 形式引用 Windows 路径；FastAPI 会自动识别。

### 5.3 前端 `.env.local`

```bash
VITE_APP_TITLE=Charge Analysis System (Local)
VITE_API_URL=http://127.0.0.1:8000/api/v1
VITE_WS_URL=ws://127.0.0.1:8000/ws
```

---

## 6. 后端启动（FastAPI）

```bash
cd charge-analysis-backend
python -m venv .venv
source .venv/Scripts/activate  # Git Bash 可直接 source Windows venv
python -m pip install --upgrade pip
pip install -r requirements.txt

alembic upgrade head  # 同步数据库结构
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- 首次运行若遇到 `psycopg` 编译失败，请执行 `pip install --upgrade setuptools wheel` 再重试。
- 需要停止服务时使用 `Ctrl+C`。
- 若想在后台运行，可在 Git Bash 中使用 `nohup uvicorn ... &`，结束时 `kill <pid>`。

---

## 7. 前端启动（React + Vite）

在新的 Git Bash 窗口中：
```bash
cd /c/workspace/charge-analysis-system/charge-analysis-frontend
pnpm install  # 或 npm install
pnpm dev --host 0.0.0.0 --port 3000
```

- 访问 `http://127.0.0.1:3000` 查看前端。
- 若需要生成生产构建：
  ```bash
  pnpm build
  pnpm preview --host 0.0.0.0 --port 4173
  ```

---

## 8. 联调与验证

1. 确保 PostgreSQL、Redis 正常运行（可通过 `psql -h 127.0.0.1 -U charge_user -d charge_analysis -c 'SELECT 1'` 与 `redis-cli ping` 测试）。
2. 后端启动后，访问 `http://127.0.0.1:8000/docs` 验证 Swagger UI。
3. 前端启动后，打开 `http://127.0.0.1:3000`，在浏览器开发者工具中确认 API 请求指向 `127.0.0.1:8000`。
4. 也可以使用 `curl` 快速检查：
   ```bash
   curl -f http://127.0.0.1:8000/health
   curl -f http://127.0.0.1:8000/api/v1/analyses
   ```

---

## 9. 常见问题（Windows + Git Bash）

| 症状 | 排查 | 解决方案 |
| --- | --- | --- |
| `source .venv/bin/activate` 找不到 | Windows venv 位于 `Scripts` | 使用 `source .venv/Scripts/activate` |
| `uvicorn` 端口被占用 | `netstat -ano | findstr 8000` | 在 PowerShell 中 `Stop-Process -Id <PID>` 或修改 `PORT` |
| `OSError: [WinError 126]` | 缺少 VC++ 运行库 | 安装 "Microsoft Visual C++ Redistributable"，重启 Git Bash |
| `psycopg` 编译失败 | 缺少 `pg_config` | 确认 PostgreSQL `bin` 目录已加入 PATH，或先安装 `psycopg[binary]` |
| `node`/`pnpm` 命令未找到 | PATH 未刷新 | 重新打开 Git Bash，或在 `~/.bashrc` 中 `export PATH="$PATH:/c/Program Files/nodejs"` |
| Redis Windows 端口不稳定 | 端口 6379 被占 | 修改 `redis.conf` 端口或使用其他端口 |

---

## 10. 建议的开发工作流

1. 使用两个 Git Bash 窗口分别运行后端与前端，避免端口冲突。
2. 开发前执行 `git pull --rebase`，确保与主分支同步。
3. 修改依赖后运行：
   ```bash
   # 后端
   pip freeze > requirements.lock  # 可选
   pytest

   # 前端
   pnpm lint
   pnpm test
   ```
4. 若需重置数据库，可在 `psql` 中执行 `DROP SCHEMA public CASCADE; CREATE SCHEMA public;` 后重新运行 `alembic upgrade head`。

---

## 11. 日常维护（本地）

- **备份**：手动复制 `charge-analysis-backend/.env`、`charge-analysis-frontend/.env.local` 以及 `/c/charge-data` 目录即可。
- **更新项目**：
  ```bash
  git pull origin main
  (cd charge-analysis-backend && source .venv/Scripts/activate && pip install -r requirements.txt)
  (cd charge-analysis-frontend && pnpm install)
  alembic upgrade head
  pnpm build  # 如需产物
  ```
- **关闭服务**：`Ctrl+C` 终止前端/后端。

---

## 12. 反馈

若在 Windows + Git Bash 场景下遇到本文未覆盖的问题，可在仓库 Issues 中附上：
- 操作系统版本（`winver`）
- Git Bash 版本
- 关键命令输出与错误日志

我们会在后续版本补充相应的排障步骤。
