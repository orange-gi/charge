"""用户级 DBC 配置管理。

目的：
- 允许用户上传自己的 DBC 文件用于 CAN 解码
- 避免引入数据库迁移（create_all 不会自动 ALTER 表）
- 通过 uploads/<user_id>/dbc/dbc_config.json 记录当前使用的 DBC
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DbcConfig:
    """当前生效的 DBC 配置（按用户）。"""

    user_id: int
    file_path: str
    filename: str
    uploaded_at: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_user_dbc_dir(user_id: int) -> Path:
    settings = get_settings()
    return Path(settings.upload_path) / str(user_id) / "dbc"


def get_user_dbc_config_file(user_id: int) -> Path:
    return get_user_dbc_dir(user_id) / "dbc_config.json"


def load_user_dbc_config(user_id: int) -> Optional[DbcConfig]:
    cfg_path = get_user_dbc_config_file(user_id)
    if not cfg_path.exists():
        return None

    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        file_path = str(raw.get("file_path") or "").strip()
        filename = str(raw.get("filename") or "").strip()
        uploaded_at = str(raw.get("uploaded_at") or "").strip()
        if not file_path:
            return None
        return DbcConfig(user_id=user_id, file_path=file_path, filename=filename, uploaded_at=uploaded_at)
    except Exception as e:  # pragma: no cover
        logger.warning("读取 DBC 配置失败 user_id=%s path=%s err=%s", user_id, str(cfg_path), e)
        return None


def resolve_user_dbc_path(user_id: int) -> Optional[str]:
    """解析用户当前生效的 DBC 路径。

    返回：
    - str：存在且可读的 DBC 文件路径
    - None：未配置/配置无效（调用方应回退到默认 DBC）
    """

    cfg = load_user_dbc_config(user_id)
    if cfg and cfg.file_path:
        p = Path(cfg.file_path)
        if p.exists() and p.is_file():
            return str(p)
        logger.warning("DBC 配置指向的文件不存在 user_id=%s file_path=%s", user_id, cfg.file_path)
        return None

    # 兼容：如果没有 json 配置，但存在 current.dbc 则使用它
    fallback = get_user_dbc_dir(user_id) / "current.dbc"
    if fallback.exists() and fallback.is_file():
        return str(fallback)

    return None


def save_user_dbc_config(user_id: int, file_path: Path, original_filename: str) -> DbcConfig:
    """保存用户 DBC 配置（覆盖当前配置）。"""

    dbc_dir = get_user_dbc_dir(user_id)
    dbc_dir.mkdir(parents=True, exist_ok=True)

    uploaded_at = _utc_now_iso()
    cfg = DbcConfig(
        user_id=user_id,
        file_path=str(file_path),
        filename=original_filename,
        uploaded_at=uploaded_at,
    )

    cfg_path = get_user_dbc_config_file(user_id)
    payload: Dict[str, Any] = {
        "file_path": cfg.file_path,
        "filename": cfg.filename,
        "uploaded_at": cfg.uploaded_at,
    }
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 同时维护一个稳定文件名，便于人工排查
    current_path = dbc_dir / "current.dbc"
    try:
        # 用 copy2 保留时间戳信息（便于排查）
        import shutil

        shutil.copy2(str(file_path), str(current_path))
    except Exception as e:  # pragma: no cover
        logger.warning("写入 current.dbc 失败 user_id=%s src=%s err=%s", user_id, str(file_path), e)

    return cfg


def dbc_config_to_response(cfg: Optional[DbcConfig]) -> Dict[str, Any]:
    if not cfg:
        return {"configured": False}

    # 不向前端暴露绝对路径，避免信息泄露
    file_path = Path(cfg.file_path)
    size = None
    mtime = None
    try:
        stat = file_path.stat()
        size = int(stat.st_size)
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        pass

    return {
        "configured": True,
        "filename": cfg.filename or file_path.name,
        "uploaded_at": cfg.uploaded_at,
        "file_size": size,
        "file_mtime": mtime,
    }

