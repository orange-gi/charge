"""认证相关 API。"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from core.dependencies import get_current_user
from core.security import compute_expiration, generate_session_token, hash_password, verify_password
from database import get_db
from models import User, UserRole, UserSession
from schemas import AuthResponse, UserCreate, UserLogin, UserRead

router = APIRouter(prefix="/api/auth", tags=["auth"])
http_bearer = HTTPBearer(auto_error=False)


def _build_auth_response(user: User, token: str) -> AuthResponse:
    payload = UserRead.model_validate(user)
    return AuthResponse(user=payload, token=token)


@router.post("/register", response_model=AuthResponse)
def register(payload: UserCreate, db: Session = Depends(get_db)) -> AuthResponse:
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="邮箱已存在")

    username = payload.username or payload.email.split("@")[0]
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

    token = generate_session_token()
    session = UserSession(
        user_id=user.id,
        token_hash=token,
        expires_at=compute_expiration(),
    )
    db.add(session)
    db.commit()

    return _build_auth_response(user, token)


@router.post("/login", response_model=AuthResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)) -> AuthResponse:
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


@router.post("/logout")
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(http_bearer),
    db: Session = Depends(get_db),
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录")

    token = credentials.credentials
    session = db.query(UserSession).filter(UserSession.token_hash == token).first()
    if session:
        db.delete(session)
        db.commit()
    return {"message": "已退出"}
