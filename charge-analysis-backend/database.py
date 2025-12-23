"""数据库会话与初始化工具。

注意：
- 该项目没有 Alembic，为了避免“只改模型不改数据库”导致运行期 500，
  这里在启动时做少量、可回滚的 schema 自修复（仅扩容，不做破坏性变更）。
"""
from collections.abc import Generator
from contextlib import contextmanager

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config import get_settings
from models import Base

settings = get_settings()
logger = logging.getLogger(__name__)

engine = create_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def init_db() -> None:
    """初始化数据库表结构。"""
    Base.metadata.create_all(bind=engine)
    _auto_migrate_schema()


def _auto_migrate_schema() -> None:
    """在缺少迁移框架时，做最小化 schema 自修复。

    目前仅处理：
    - knowledge_documents.file_type: 50 -> 255（Excel 的 MIME type 超过 50）
    """
    if engine.dialect.name != "postgresql":
        return

    with engine.begin() as conn:
        _maybe_widen_pg_varchar(conn, table="knowledge_documents", column="file_type", target_len=255)
        # 预防历史版本把这些字段建得过短（不影响新库）
        _maybe_widen_pg_varchar(conn, table="knowledge_documents", column="file_path", target_len=500)
        _maybe_widen_pg_varchar(conn, table="knowledge_documents", column="filename", target_len=255)


def _maybe_widen_pg_varchar(conn, table: str, column: str, target_len: int, schema: str = "public") -> None:
    """如果列是 varchar 且长度小于目标长度，则扩容。"""
    try:
        row = (
            conn.execute(
                text(
                    """
                    SELECT data_type, character_maximum_length
                    FROM information_schema.columns
                    WHERE table_schema = :schema
                      AND table_name = :table
                      AND column_name = :column
                    """
                ),
                {"schema": schema, "table": table, "column": column},
            )
            .mappings()
            .first()
        )
        if not row:
            return

        data_type = (row.get("data_type") or "").lower()
        cur_len = row.get("character_maximum_length")

        # text / json 等类型无需处理；仅针对 varchar/char
        if data_type not in {"character varying", "character"}:
            return
        if cur_len is None or int(cur_len) >= int(target_len):
            return

        conn.execute(
            text(f'ALTER TABLE "{schema}"."{table}" ALTER COLUMN "{column}" TYPE VARCHAR({int(target_len)})')
        )
        logger.info("Auto-migrated %s.%s.%s: varchar(%s) -> varchar(%s)", schema, table, column, cur_len, target_len)
    except Exception as exc:
        # 不阻塞服务启动；但会在日志里提示（便于用户手工迁移）
        logger.warning("Auto-migrate skipped for %s.%s.%s: %s", schema, table, column, exc)


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
