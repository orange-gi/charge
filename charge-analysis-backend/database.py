"""数据库会话与初始化工具。"""
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import get_settings
from models import Base

settings = get_settings()

engine = create_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """初始化数据库表结构。"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator:
    """FastAPI 依赖注入使用的会话生成器。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope():
    """提供一个自动提交/回滚的事务上下文。"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
