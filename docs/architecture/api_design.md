# API 接口设计

## 1. API 设计原则

### 1.1 RESTful 设计
- **资源导向**: URL 代表资源，HTTP 方法代表操作
- **无状态**: 每个请求包含所有必要信息
- **统一接口**: 使用标准 HTTP 方法和状态码
- **可缓存**: 适当使用 HTTP 缓存机制

### 1.2 响应格式
```json
{
  "success": true,
  "data": {},
  "message": "操作成功",
  "timestamp": "2025-11-19T14:47:27Z",
  "request_id": "req_123456789"
}
```

### 1.3 错误处理
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求参数验证失败",
    "details": {
      "field": "email",
      "reason": "邮箱格式不正确"
    }
  },
  "timestamp": "2025-11-19T14:47:27Z",
  "request_id": "req_123456789"
}
```

## 2. 认证相关接口

### 2.1 用户注册
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "testuser",
  "email": "test@example.com",
  "password": "securePassword123",
  "first_name": "Test",
  "last_name": "User"
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "user": {
      "id": 1,
      "username": "testuser",
      "email": "test@example.com",
      "role": "user",
      "created_at": "2025-11-19T14:47:27Z"
    },
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "expires_in": 1800
  }
}
```

### 2.2 用户登录
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "testuser",
  "password": "securePassword123"
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "user": {
      "id": 1,
      "username": "testuser",
      "email": "test@example.com",
      "role": "admin",
      "last_login": "2025-11-19T14:47:27Z"
    },
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "expires_in": 1800
  }
}
```

### 2.3 刷新令牌
```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### 2.4 获取当前用户信息
```http
GET /api/v1/auth/me
Authorization: Bearer {access_token}
```

### 2.5 用户登出
```http
POST /api/v1/auth/logout
Authorization: Bearer {access_token}
```

## 3. 充电分析接口

### 3.1 上传充电数据文件
```http
POST /api/v1/charging/upload
Content-Type: multipart/form-data
Authorization: Bearer {access_token}

file: [BLF文件]
name: "充电测试数据"
description: "2025年11月充电测试数据"
```

**响应**:
```json
{
  "success": true,
  "data": {
    "analysis_id": 1001,
    "file_info": {
      "filename": "charging_test.blf",
      "size": 15728640,
      "upload_time": "2025-11-19T14:47:27Z"
    },
    "status": "uploaded"
  }
}
```

### 3.2 开始分析
```http
POST /api/v1/charging/analyze
Content-Type: application/json
Authorization: Bearer {access_token}

{
  "analysis_id": 1001,
  "analysis_options": {
    "signals_filter": ["BMS_DCChrgSt", "BMS_BattCurrt"],
    "time_range": {
      "start": "2025-11-19T10:00:00Z",
      "end": "2025-11-19T12:00:00Z"
    },
    "analysis_depth": "detailed"
  }
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "task_id": "task_1001",
    "status": "started",
    "estimated_time": 120,
    "websocket_url": "/ws/analysis/task_1001"
  }
}
```

### 3.3 获取分析进度
```http
GET /api/v1/charging/progress/{task_id}
Authorization: Bearer {access_token}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "task_id": "task_1001",
    "status": "processing",
    "progress": 45.5,
    "current_step": "信号分析中...",
    "estimated_remaining": 65
  }
}
```

### 3.4 获取分析结果
```http
GET /api/v1/charging/results/{analysis_id}
Authorization: Bearer {access_token}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "analysis_id": 1001,
    "status": "completed",
    "completed_at": "2025-11-19T14:50:15Z",
    "results": [
      {
        "type": "charging_efficiency",
        "title": "充电效率分析",
        "content": "本次充电过程中，系统表现出良好的充电效率...",
        "confidence": 0.92,
        "metrics": {
          "efficiency": 0.89,
          "duration": 120,
          "energy_consumed": 45.6
        }
      },
      {
        "type": "anomaly_detection",
        "title": "异常检测结果",
        "content": "在时间戳 11:30:15 检测到电流波动异常...",
        "confidence": 0.87,
        "metrics": {
          "anomaly_count": 3,
          "severity": "medium",
          "affected_signals": ["BMS_BattCurrt", "BCL_CurrentRequire"]
        }
      }
    ],
    "visualizations": [
      {
        "type": "line_chart",
        "title": "充电电流变化趋势",
        "data_url": "/api/v1/charging/chart/1001/1",
        "signals": ["BMS_BattCurrt", "BCL_CurrentRequire"]
      }
    ]
  }
}
```

### 3.5 获取分析历史
```http
GET /api/v1/charging/history?page=1&limit=20&status=all
Authorization: Bearer {access_token}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "analyses": [
      {
        "id": 1001,
        "name": "充电测试数据",
        "status": "completed",
        "progress": 100,
        "file_name": "charging_test.blf",
        "created_at": "2025-11-19T14:47:27Z",
        "completed_at": "2025-11-19T14:50:15Z"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total": 15,
      "total_pages": 1
    }
  }
}
```

### 3.6 获取信号图表数据
```http
GET /api/v1/charging/chart/{analysis_id}/{chart_id}
Authorization: Bearer {access_token}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "chart_id": 1,
    "signals": ["BMS_BattCurrt", "BCL_CurrentRequire"],
    "data": [
      {
        "timestamp": "2025-11-19T10:00:00Z",
        "BMS_BattCurrt": 25.6,
        "BCL_CurrentRequire": 30.0
      },
      {
        "timestamp": "2025-11-19T10:00:01Z",
        "BMS_BattCurrt": 26.1,
        "BCL_CurrentRequire": 30.0
      }
    ]
  }
}
```

## 4. RAG 管理接口

### 4.1 获取知识库集合列表
```http
GET /api/v1/rag/collections
Authorization: Bearer {access_token}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "collections": [
      {
        "id": 1,
        "name": "充电技术文档",
        "description": "充电系统相关技术文档",
        "type": "document",
        "document_count": 156,
        "is_active": true,
        "created_at": "2025-11-19T10:00:00Z"
      }
    ]
  }
}
```

### 4.2 创建知识库集合
```http
POST /api/v1/rag/collections
Content-Type: application/json
Authorization: Bearer {access_token}

{
  "name": "故障排除指南",
  "description": "常见故障排除方法和步骤",
  "type": "guide"
}
```

### 4.3 上传文档到知识库
```http
POST /api/v1/rag/upload
Content-Type: multipart/form-data
Authorization: Bearer {access_token}

collection_id: 1
file: [PDF/DOC/TXT文件]
metadata: {"category": "troubleshooting", "tags": ["常见故障", "解决方案"]}
```

### 4.4 检索知识
```http
POST /api/v1/rag/query
Content-Type: application/json
Authorization: Bearer {access_token}

{
  "collection_id": 1,
  "query": "充电电流异常的处理方法",
  "top_k": 5,
  "score_threshold": 0.7
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "query": "充电电流异常的处理方法",
    "results": [
      {
        "document": "当检测到充电电流异常时，首先应检查连接器...",
        "metadata": {
          "source": "故障排除指南.pdf",
          "category": "troubleshooting",
          "score": 0.89
        }
      }
    ],
    "total_results": 3,
    "query_time_ms": 45
  }
}
```

### 4.5 测试检索效果
```http
POST /api/v1/rag/test
Content-Type: application/json
Authorization: Bearer {access_token}

{
  "collection_id": 1,
  "test_queries": [
    "充电系统启动失败怎么办？",
    "电流过大的原因有哪些？"
  ]
}
```

### 4.6 删除集合
```http
DELETE /api/v1/rag/collections/{collection_id}
Authorization: Bearer {access_token}
```

## 5. 训练管理接口

### 5.1 获取数据集列表
```http
GET /api/v1/training/datasets?type=all&page=1&limit=20
Authorization: Bearer {access_token}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "datasets": [
      {
        "id": 1,
        "name": "充电数据标准集",
        "description": "包含正常和异常充电数据",
        "type": "standard",
        "sample_count": 1500,
        "is_public": true,
        "created_at": "2025-11-19T10:00:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total": 5,
      "total_pages": 1
    }
  }
}
```

### 5.2 上传训练数据集
```http
POST /api/v1/training/datasets
Content-Type: multipart/form-data
Authorization: Bearer {access_token}

name: "充电异常数据集"
description: "充电异常情况的数据集"
type: chain_of_thought
file: [CSV/JSON文件]
```

### 5.3 获取训练任务列表
```http
GET /api/v1/training/tasks?status=all&page=1&limit=20
Authorization: Bearer {access_token}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "tasks": [
      {
        "id": 1,
        "name": "1.5B流程控制模型训练",
        "dataset_name": "充电数据标准集",
        "model_type": "flow_control",
        "status": "running",
        "progress": 65.5,
        "current_epoch": 13,
        "total_epochs": 20,
        "current_step": 650,
        "total_steps": 1000,
        "created_at": "2025-11-19T12:00:00Z",
        "estimated_completion": "2025-11-19T16:00:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total": 8,
      "total_pages": 1
    }
  }
}
```

### 5.4 创建训练任务
```http
POST /api/v1/training/tasks
Content-Type: application/json
Authorization: Bearer {access_token}

{
  "name": "新模型训练任务",
  "dataset_id": 1,
  "model_type": "flow_control",
  "hyperparameters": {
    "learning_rate": 0.0001,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "adamw"
  }
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "task_id": 2,
    "status": "queued",
    "queue_position": 1,
    "estimated_start_time": "2025-11-19T15:00:00Z",
    "websocket_url": "/ws/training/task_2"
  }
}
```

### 5.5 获取训练任务详情
```http
GET /api/v1/training/tasks/{task_id}
Authorization: Bearer {access_token}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "1.5B流程控制模型训练",
    "status": "running",
    "progress": 65.5,
    "metrics": {
      "loss": 0.0234,
      "accuracy": 0.9245,
      "learning_rate": 0.0001,
      "gpu_memory_usage": 4.2
    },
    "hyperparameters": {
      "learning_rate": 0.0001,
      "batch_size": 32,
      "epochs": 50
    },
    "start_time": "2025-11-19T12:00:00Z",
    "estimated_completion": "2025-11-19T16:00:00Z",
    "logs": [
      {
        "timestamp": "2025-11-19T14:47:27Z",
        "level": "INFO",
        "message": "Epoch 13/50 completed"
      }
    ]
  }
}
```

### 5.6 获取训练指标历史
```http
GET /api/v1/training/tasks/{task_id}/metrics?epoch_start=1&epoch_end=20
Authorization: Bearer {access_token}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "task_id": 1,
    "metrics": [
      {
        "epoch": 1,
        "step": 50,
        "loss": 0.1567,
        "accuracy": 0.7234,
        "learning_rate": 0.0001,
        "gpu_memory": 3.8
      }
    ]
  }
}
```

### 5.7 取消训练任务
```http
POST /api/v1/training/tasks/{task_id}/cancel
Authorization: Bearer {access_token}
```

### 5.8 获取模型版本列表
```http
GET /api/v1/training/models?type=flow_control
Authorization: Bearer {access_token}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "id": 1,
        "name": "1.5B流程控制模型",
        "type": "flow_control",
        "version": "v1.2.3",
        "metrics": {
          "accuracy": 0.9245,
          "f1_score": 0.8934,
          "inference_time": 0.045
        },
        "is_active": true,
        "is_default": true,
        "created_at": "2025-11-19T10:00:00Z"
      }
    ]
  }
}
```

### 5.9 选择默认模型
```http
POST /api/v1/training/models/{model_id}/set_default
Authorization: Bearer {access_token}
```

## 6. 日志管理接口

### 6.1 获取系统日志
```http
GET /api/v1/logs?level=all&module=all&page=1&limit=50&start_time=2025-11-19T00:00:00Z&end_time=2025-11-19T23:59:59Z
Authorization: Bearer {access_token}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "logs": [
      {
        "id": 1001,
        "level": "INFO",
        "module": "charging_analysis",
        "message": "充电分析任务开始",
        "user_id": 1,
        "timestamp": "2025-11-19T14:47:27Z",
        "metadata": {
          "analysis_id": 1001,
          "user_agent": "Mozilla/5.0..."
        }
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 50,
      "total": 1250,
      "total_pages": 25
    }
  }
}
```

### 6.2 获取日志统计
```http
GET /api/v1/logs/statistics?period=24h
Authorization: Bearer {access_token}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "period": "24h",
    "total_logs": 15420,
    "level_distribution": {
      "DEBUG": 8200,
      "INFO": 5420,
      "WARNING": 1200,
      "ERROR": 550,
      "CRITICAL": 50
    },
    "module_distribution": {
      "charging_analysis": 3200,
      "rag_system": 2100,
      "training_system": 1800,
      "auth_system": 8400
    },
    "error_rate": 0.039,
    "trend_data": [
      {
        "timestamp": "2025-11-19T14:00:00Z",
        "logs_count": 245,
        "error_count": 8
      }
    ]
  }
}
```

### 6.3 导出日志
```http
GET /api/v1/logs/export?level=ERROR&start_time=2025-11-19T00:00:00Z&end_time=2025-11-19T23:59:59Z&format=csv
Authorization: Bearer {access_token}
```

### 6.4 获取审计日志
```http
GET /api/v1/logs/audit?page=1&limit=20
Authorization: Bearer {access_token}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "audit_logs": [
      {
        "id": 1,
        "action": "CREATE",
        "resource_type": "charging_analysis",
        "resource_id": "1001",
        "user_id": 1,
        "timestamp": "2025-11-19T14:47:27Z",
        "changes": {
          "new_values": {
            "name": "测试分析",
            "status": "pending"
          }
        }
      }
    ]
  }
}
```

## 7. WebSocket 接口

### 7.1 连接认证
```javascript
// WebSocket 客户端连接
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function() {
  // 发送认证信息
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'Bearer eyJ0eXAiOiJKV1Qi...'
  }));
};
```

### 7.2 充电分析进度更新
```javascript
// 订阅分析进度
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'analysis_progress',
  task_id: 'task_1001'
}));

// 接收进度更新
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (data.type === 'analysis_progress') {
    console.log(`分析进度: ${data.progress}%`);
    console.log(`当前步骤: ${data.current_step}`);
  }
};
```

### 7.3 训练任务状态更新
```javascript
// 订阅训练进度
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'training_progress',
  task_id: 'task_2'
}));

// 接收训练更新
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (data.type === 'training_progress') {
    console.log(`训练进度: ${data.progress}%`);
    console.log(`当前轮次: ${data.current_epoch}/${data.total_epochs}`);
    console.log(`损失值: ${data.metrics.loss}`);
  }
};
```

### 7.4 系统通知
```javascript
// 订阅系统通知
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'notifications'
}));

// 接收通知
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (data.type === 'notification') {
    console.log(`通知: ${data.message}`);
  }
};
```

## 8. 错误码定义

### 8.1 HTTP 状态码
- `200 OK`: 请求成功
- `201 Created`: 创建成功
- `400 Bad Request`: 请求参数错误
- `401 Unauthorized`: 未授权访问
- `403 Forbidden`: 禁止访问
- `404 Not Found`: 资源不存在
- `422 Unprocessable Entity`: 请求格式正确但语义错误
- `429 Too Many Requests`: 请求频率过高
- `500 Internal Server Error`: 服务器内部错误

### 8.2 业务错误码
```json
{
  "VALIDATION_ERROR": "1001",
  "AUTHENTICATION_FAILED": "1002", 
  "AUTHORIZATION_FAILED": "1003",
  "RESOURCE_NOT_FOUND": "1004",
  "FILE_TOO_LARGE": "1005",
  "UNSUPPORTED_FILE_TYPE": "1006",
  "INSUFFICIENT_PERMISSIONS": "1007",
  "TRAINING_TASK_NOT_FOUND": "2001",
  "TRAINING_FAILED": "2002",
  "MODEL_NOT_AVAILABLE": "2003",
  "DATASET_CORRUPTED": "2004",
  "RAG_QUERY_ERROR": "3001",
  "KNOWLEDGE_BASE_ERROR": "3002",
  "ANALYSIS_FAILED": "4001",
  "FILE_PROCESSING_ERROR": "4002"
}
```

## 9. API 限流和缓存

### 9.1 限流策略
```python
# 限流配置
RATE_LIMITS = {
    "auth": {"requests": 10, "window": 60},  # 每分钟10次
    "charging_upload": {"requests": 5, "window": 300},  # 每5分钟5次
    "rag_query": {"requests": 100, "window": 3600},  # 每小时100次
    "training": {"requests": 3, "window": 3600}  # 每小时3次
}
```

### 9.2 缓存策略
```python
# 缓存配置
CACHE_STRATEGY = {
    "analysis_results": {"ttl": 3600},  # 分析结果缓存1小时
    "rag_query": {"ttl": 600},  # RAG查询结果缓存10分钟
    "model_list": {"ttl": 1800},  # 模型列表缓存30分钟
    "system_stats": {"ttl": 300}  # 系统统计缓存5分钟
}
```

## 10. API 测试文档

### 10.1 Postman 集合
```json
{
  "info": {
    "name": "Charge Analysis API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Authentication",
      "item": [
        {
          "name": "User Registration",
          "request": {
            "method": "POST",
            "header": [
              {
                "key": "Content-Type",
                "value": "application/json"
              }
            ],
            "body": {
              "mode": "raw",
              "raw": "{\n  \"username\": \"testuser\",\n  \"email\": \"test@example.com\",\n  \"password\": \"securePassword123\"\n}"
            },
            "url": {
              "raw": "{{base_url}}/api/v1/auth/register",
              "host": ["{{base_url}}"],
              "path": ["api", "v1", "auth", "register"]
            }
          }
        }
      ]
    }
  ]
}
```

### 10.2 自动化测试脚本
```python
# tests/api/test_auth.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_user_registration():
    async with AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/auth/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "securePassword123"
            }
        )
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "access_token" in data["data"]
```

这个API接口设计文档为系统提供了完整的接口规范，包括认证、充电分析、RAG管理、训练管理和日志管理等所有功能模块的详细接口定义。