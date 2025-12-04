"""安全与加密相关帮助函数。"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """生成密码哈希。"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码。"""
    return pwd_context.verify(plain_password, hashed_password)


def generate_session_token() -> str:
    """生成随机会话 token。"""
    return secrets.token_hex(32)


def compute_expiration(days: int = 7) -> datetime:
    """计算过期时间。"""
    return datetime.utcnow() + timedelta(days=days)
