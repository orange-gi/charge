#!/usr/bin/env python3
"""数据库迁移脚本 - 更新表结构以匹配模型定义"""
from sqlalchemy import text, inspect
from database import engine
from models import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_and_add_column(table_name: str, column_name: str, column_def: str):
    """检查列是否存在，如果不存在则添加"""
    inspector = inspect(engine)
    existing_columns = [col['name'] for col in inspector.get_columns(table_name)]
    
    if column_name not in existing_columns:
        logger.info(f"添加列 {table_name}.{column_name}")
        with engine.connect() as conn:
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_def}"))
            conn.commit()
        logger.info(f"✓ 已添加列 {table_name}.{column_name}")
    else:
        logger.info(f"✓ 列 {table_name}.{column_name} 已存在")


def check_and_add_foreign_key(table_name: str, column_name: str, ref_table: str, ref_column: str = "id"):
    """检查外键是否存在，如果不存在则添加"""
    with engine.connect() as conn:
        # 检查外键约束
        result = conn.execute(text(f"""
            SELECT constraint_name 
            FROM information_schema.table_constraints 
            WHERE table_name = '{table_name}' 
            AND constraint_type = 'FOREIGN KEY'
            AND constraint_name LIKE '%{column_name}%'
        """))
        existing_fk = result.fetchone()
        
        if not existing_fk:
            fk_name = f"fk_{table_name}_{column_name}"
            logger.info(f"添加外键 {table_name}.{column_name} -> {ref_table}.{ref_column}")
            conn.execute(text(f"""
                ALTER TABLE {table_name} 
                ADD CONSTRAINT {fk_name} 
                FOREIGN KEY ({column_name}) 
                REFERENCES {ref_table}({ref_column}) 
                ON DELETE SET NULL
            """))
            conn.commit()
            logger.info(f"✓ 已添加外键 {fk_name}")
        else:
            logger.info(f"✓ 外键 {table_name}.{column_name} 已存在")


def migrate_training_tasks():
    """迁移 training_tasks 表"""
    logger.info("检查 training_tasks 表...")
    
    # 添加缺失的列
    check_and_add_column("training_tasks", "config_id", "config_id INTEGER")
    check_and_add_column("training_tasks", "adapter_type", "adapter_type VARCHAR(50) DEFAULT 'lora'")
    check_and_add_column("training_tasks", "model_size", "model_size VARCHAR(20) DEFAULT '1.5b'")
    
    # 添加外键约束
    check_and_add_foreign_key("training_tasks", "config_id", "training_configs")


def check_all_tables():
    """检查所有表的结构"""
    logger.info("检查所有表结构...")
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    logger.info(f"数据库中的表: {len(tables)}")
    for table in sorted(tables):
        columns = inspector.get_columns(table)
        logger.info(f"  {table}: {len(columns)} 列")
    
    # 检查 training_tasks 表
    if "training_tasks" in tables:
        columns = [col['name'] for col in inspector.get_columns("training_tasks")]
        logger.info(f"\ntraining_tasks 表的列: {', '.join(columns)}")
        
        required_columns = ['config_id', 'adapter_type', 'model_size']
        missing = [col for col in required_columns if col not in columns]
        if missing:
            logger.warning(f"缺失的列: {', '.join(missing)}")
        else:
            logger.info("✓ 所有必需的列都存在")


def main():
    """执行迁移"""
    logger.info("开始数据库迁移...")
    logger.info(f"数据库连接: {engine.url}")
    
    try:
        # 检查当前状态
        check_all_tables()
        
        # 执行迁移
        migrate_training_tasks()
        
        # 再次检查
        logger.info("\n迁移完成，验证结果...")
        check_all_tables()
        
        logger.info("\n✓ 数据库迁移完成！")
        
    except Exception as e:
        logger.error(f"迁移失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

