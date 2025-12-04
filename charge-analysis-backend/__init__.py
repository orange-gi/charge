#!/usr/bin/env python3
"""
Charge Analysis Backend - 基于 LangGraph 的充电分析系统后端

该系统提供了完整的充电数据分析功能，包括：
- 用户认证和权限管理
- 充电数据文件上传和解析
- 基于 LangGraph 的智能分析工作流
- RAG 知识库管理和检索
- 机器学习模型训练和管理
- 系统日志记录和监控

主要技术栈：
- FastAPI: Web 框架
- LangGraph: Agent 工作流编排
- ChromaDB: 向量数据库
- PostgreSQL: 关系数据库
- SQLAlchemy: ORM
- JWT: 身份认证
- WebSocket: 实时通信
"""

__version__ = "1.0.0"
__author__ = "Charge Analysis Team"
__description__ = "Intelligent Charging Analysis System Backend"