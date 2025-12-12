"""认证相关 API。"""
from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.dependencies import get_current_user
from core.security import compute_expiration, generate_session_token, hash_password, verify_password
from database import get_db
from models import User, UserRole, UserSession
from schemas import AuthResponse, MessageResponse, UserCreate, UserLogin, UserRead

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])
http_bearer = HTTPBearer(auto_error=False)


def _build_auth_response(user: User, token: str) -> AuthResponse:
    payload = UserRead.model_validate(user)
    return AuthResponse(user=payload, token=token)


@router.post("/register", response_model=AuthResponse)
def register(
    payload: UserCreate = Body(
        ...,
        example={
            "email": "engineer@example.com",
            "password": "Charge#2024",
            "username": "charger_admin",
            "first_name": "Li",
            "last_name": "Lei",
        },
    ),
    db: Session = Depends(get_db),
) -> AuthResponse:
    try:
        # 检查邮箱是否已存在
        if db.query(User).filter(User.email == payload.email).first():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="邮箱已存在")

        # 生成用户名（如果未提供）
        username = payload.username or payload.email.split("@")[0]
        
        # 检查用户名是否已存在
        if db.query(User).filter(User.username == username).first():
            # 如果用户名已存在，添加数字后缀
            counter = 1
            base_username = username
            while db.query(User).filter(User.username == username).first():
                username = f"{base_username}{counter}"
                counter += 1

        # 创建用户
        user = User(
            email=payload.email,
            username=username,
            first_name=payload.first_name,
            last_name=payload.last_name,
            password_hash=hash_password(payload.password),
            role=UserRole.USER,
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        # 生成会话 token（重试机制防止 token 冲突）
        max_retries = 5
        session = None
        for _ in range(max_retries):
            try:
                token = generate_session_token()
                session = UserSession(
                    user_id=user.id,
                    token_hash=token,
                    expires_at=compute_expiration(),
                )
                db.add(session)
                db.commit()
                break
            except IntegrityError:
                db.rollback()
                logger.warning(f"Token collision detected, retrying... ({_ + 1}/{max_retries})")
                if _ == max_retries - 1:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="无法创建会话，请重试"
                    )

        return _build_auth_response(user, token)
    except HTTPException:
        raise
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Database integrity error during registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="注册失败，邮箱或用户名可能已存在"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error during registration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="注册失败，请稍后重试"
        )


@router.post("/login", response_model=AuthResponse)
def login(
    payload: UserLogin = Body(
        ...,
        example={"email": "engineer@example.com", "password": "Charge#2024"},
    ),
    db: Session = Depends(get_db),
) -> AuthResponse:
    user: User | None = db.query(User).filter(User.email == payload.email).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="邮箱或密码错误")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="账号已禁用")

    user.last_login = datetime.utcnow()
    token = generate_session_token()
    session = UserSession(user_id=user.id, token_hash=token, expires_at=compute_expiration())
    db.add_all([user, session])
    db.commit()

    return _build_auth_response(user, token)


@router.get("/me", response_model=UserRead)
def current_user(user: User = Depends(get_current_user)) -> UserRead:
    return UserRead.model_validate(user)


@router.post("/logout", response_model=MessageResponse)
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(http_bearer),
    db: Session = Depends(get_db),
) -> MessageResponse:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录")

    token = credentials.credentials
    session = db.query(UserSession).filter(UserSession.token_hash == token).first()
    if session:
        db.delete(session)
        db.commit()
    return MessageResponse(message="已退出")
