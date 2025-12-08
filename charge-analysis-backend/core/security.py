"""安全与加密相关帮助函数。"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta

import bcrypt

# bcrypt 配置：使用 12 轮加密（平衡安全性和性能）
BCRYPT_ROUNDS = 12


def _prepare_password(password: str) -> bytes:
    """预处理密码以处理 bcrypt 的 72 字节限制。
    
    bcrypt 限制：密码不能超过 72 字节。
    如果密码超过 72 字节，先进行 SHA256 哈希（32 字节）再传给 bcrypt。
    """
    password_bytes = password.encode('utf-8')
    
    if len(password_bytes) > 72:
        # 如果密码超过 72 字节，先进行 SHA256 哈希
        return hashlib.sha256(password_bytes).digest()
    
    return password_bytes


def hash_password(password: str) -> str:
    """生成密码哈希。
    
    使用 bcrypt 进行密码哈希，自动处理超过 72 字节的密码。
    """
    password_bytes = _prepare_password(password)
    
    # 生成 salt 并哈希密码
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(password_bytes, salt)
    
    # 返回字符串格式（包含算法标识和哈希值）
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码。
    
    验证明文密码是否匹配已哈希的密码。
    """
    try:
        password_bytes = _prepare_password(plain_password)
        hashed_bytes = hashed_password.encode('utf-8')
        
        # 使用 bcrypt 验证密码
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except (ValueError, TypeError):
        # 如果哈希格式不正确，返回 False
        return False


def generate_session_token() -> str:
    """生成随机会话 token。"""
    return secrets.token_hex(32)


def compute_expiration(days: int = 7) -> datetime:
    """计算过期时间。"""
    return datetime.utcnow() + timedelta(days=days)
