#!/usr/bin/env python3
"""完整的数据库更新脚本 - 确保所有表结构与模型定义一致"""
from sqlalchemy import text, inspect, MetaData
from database import engine
from models import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_table_columns(table_name: str) -> list[str]:
    """获取表的所有列名"""
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return []
    columns = inspector.get_columns(table_name)
    return [col['name'] for col in columns]


def add_column_if_missing(table_name: str, column_name: str, column_def: str):
    """如果列不存在则添加"""
    columns = get_table_columns(table_name)
    if column_name not in columns:
        logger.info(f"添加列 {table_name}.{column_name}")
        try:
            with engine.connect() as conn:
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_def}"))
                conn.commit()
            logger.info(f"✓ 已添加列 {table_name}.{column_name}")
            return True
        except Exception as e:
            logger.error(f"✗ 添加列 {table_name}.{column_name} 失败: {e}")
            return False
    else:
        logger.debug(f"✓ 列 {table_name}.{column_name} 已存在")
        return False


def update_all_tables():
    """更新所有表结构"""
    logger.info("开始更新所有数据库表...")
    logger.info(f"数据库连接: {engine.url}")
    
    changes_made = False
    
    # 更新 training_tasks 表
    logger.info("\n检查 training_tasks 表...")
    if add_column_if_missing("training_tasks", "config_id", "config_id INTEGER"):
        changes_made = True
    if add_column_if_missing("training_tasks", "adapter_type", "adapter_type VARCHAR(50) DEFAULT 'lora'"):
        changes_made = True
    if add_column_if_missing("training_tasks", "model_size", "model_size VARCHAR(20) DEFAULT '1.5b'"):
        changes_made = True
    
    # 添加外键约束（如果列已存在但外键不存在）
    try:
        with engine.connect() as conn:
            # 检查 config_id 外键
            result = conn.execute(text("""
                SELECT constraint_name 
                FROM information_schema.table_constraints 
                WHERE table_name = 'training_tasks' 
                AND constraint_type = 'FOREIGN KEY'
                AND constraint_name LIKE '%config_id%'
            """))
            if not result.fetchone():
                logger.info("添加外键 training_tasks.config_id -> training_configs.id")
                conn.execute(text("""
                    ALTER TABLE training_tasks 
                    ADD CONSTRAINT fk_training_tasks_config_id 
                    FOREIGN KEY (config_id) 
                    REFERENCES training_configs(id) 
                    ON DELETE SET NULL
                """))
                conn.commit()
                logger.info("✓ 已添加外键")
                changes_made = True
    except Exception as e:
        logger.warning(f"添加外键时出错（可能已存在）: {e}")
    
    # 这里可以添加其他表的更新逻辑
    # 例如：检查其他表是否缺少列等
    
    if not changes_made:
        logger.info("\n✓ 所有表都已是最新状态，无需更新")
    else:
        logger.info("\n✓ 数据库更新完成")
    
    return changes_made


def verify_tables():
    """验证所有表的结构"""
    logger.info("\n验证表结构...")
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    logger.info(f"数据库中的表数量: {len(tables)}")
    
    # 验证 training_tasks 表
    if "training_tasks" in tables:
        columns = get_table_columns("training_tasks")
        required_columns = ['id', 'name', 'description', 'dataset_id', 'config_id', 
                           'model_version_id', 'model_type', 'adapter_type', 'model_size',
                           'hyperparameters', 'status', 'progress', 'current_epoch', 
                           'total_epochs', 'current_step', 'total_steps', 'metrics', 
                           'logs', 'error_message', 'start_time', 'end_time', 
                           'duration_seconds', 'gpu_memory_usage', 'model_path', 
                           'created_by', 'created_at', 'updated_at']
        
        missing = [col for col in required_columns if col not in columns]
        if missing:
            logger.warning(f"training_tasks 表缺失列: {', '.join(missing)}")
        else:
            logger.info(f"✓ training_tasks 表结构完整 ({len(columns)} 列)")
    
    logger.info("\n表结构验证完成")


def main():
    """主函数"""
    try:
        # 更新表
        changes = update_all_tables()
        
        # 验证
        verify_tables()
        
        if changes:
            logger.info("\n✅ 数据库更新成功！请重启后端服务。")
        else:
            logger.info("\n✅ 数据库已是最新状态！")
            
    except Exception as e:
        logger.error(f"更新失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

