"""FastAPI 入口。"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html

from config import get_settings
from database import init_db
from routers import analyses, auth, rag, training

settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url=None,  # 禁用默认文档，使用自定义
    redoc_url=None,  # 可选：禁用 redoc
)

# 重要：CORS 中间件必须在路由注册之前添加
# 说明：
# - 远程部署时，前端通常以 http://<服务器IP>:3000 访问；如果后端只允许 localhost，会被浏览器 CORS 拦截。
# - 为了开箱即用：development 环境默认允许任意来源（通过 regex 回显具体 Origin，兼容 credentials）。
# - 生产环境请设置 ENVIRONMENT=production 并显式配置 ALLOWED_ORIGINS。
cors_common = dict(
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

if str(getattr(settings, "environment", "development")).lower() in {"production", "prod"}:
    app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, **cors_common)
else:
    app.add_middleware(CORSMiddleware, allow_origin_regex=".*", **cors_common)

# 使用国内可访问的 CDN 配置自定义 Swagger UI
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    # 使用 unpkg.com CDN（国内访问通常较稳定）
    # 如果 unpkg 访问有问题，可以切换到以下备选 CDN：
    # - Cloudflare: https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.9.0/
    # - BootCDN: https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.9.0/
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
    )

# 可选：添加 redoc 支持
@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    from fastapi.openapi.docs import get_redoc_html
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
    )


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    Path(settings.upload_path).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)


app.include_router(auth.router)
app.include_router(analyses.router)
app.include_router(rag.router)
app.include_router(training.router)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}
