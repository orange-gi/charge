# 详细模块实现文档

## 1. 充电分析模块详细设计

### 1.1 模块概述

充电分析模块是系统的核心功能，负责处理充电数据文件的解析、分析和可视化展示。模块基于 LangGraph 工作流，实现了从文件上传到报告生成的完整分析流程。

### 1.2 核心组件设计

#### 1.2.1 文件上传组件
```typescript
// src/components/charging/FileUploader.tsx
import React, { useCallback, useState } from 'react';
import { Upload, Card, Progress, Button, Space, Typography } from 'antd';
import { InboxOutlined, DeleteOutlined } from '@ant-design/icons';
import { useFileUpload } from '../../hooks/useFileUpload';
import { useChargingStore } from '../../stores/chargingStore';
import type { UploadFile, UploadProps } from 'antd';

const { Dragger } = Upload;
const { Text } = Typography;

interface FileUploaderProps {
  onUpload: (file: File) => void;
  uploadedFile: File | null;
  maxSize?: number;
}

export const FileUploader: React.FC<FileUploaderProps> = ({
  onUpload,
  uploadedFile,
  maxSize = 100 * 1024 * 1024
}) => {
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const { uploadFile, isUploading, uploadProgress } = useFileUpload({
    maxSize,
    acceptedTypes: ['.blf', '.csv', '.xlsx'],
    onUpload: (file, response) => {
      onUpload(file);
      setFileList([{
        uid: file.name,
        name: file.name,
        status: 'done',
        response: response
      }]);
    }
  });

  const handleChange: UploadProps['onChange'] = useCallback((info) => {
    const { status } = info.file;
    
    if (status === 'done') {
      message.success(`${info.file.name} 文件上传成功`);
    } else if (status === 'error') {
      message.error(`${info.file.name} 文件上传失败`);
    }
    
    setFileList(info.fileList);
  }, []);

  const handleRemove = useCallback((file: UploadFile) => {
    setFileList([]);
    onUpload(new File([], ''));
  }, [onUpload]);

  const uploadProps: UploadProps = {
    name: 'file',
    multiple: false,
    accept: '.blf,.csv,.xlsx',
    beforeUpload: async (file) => {
      await uploadFile(file);
      return false; // 阻止自动上传
    },
    onChange: handleChange,
    onRemove: handleRemove,
    fileList,
    disabled: isUploading
  };

  return (
    <Card className="file-uploader" size="small">
      <Dragger {...uploadProps} className={isUploading ? 'uploading' : ''}>
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">
          {isUploading ? '上传中...' : '点击或拖拽文件到此区域上传'}
        </p>
        <p className="ant-upload-hint">
          支持 BLF、CSV、Excel 格式文件，单个文件不超过 100MB
        </p>
        
        {isUploading && (
          <div style={{ marginTop: 16 }}>
            <Progress 
              percent={uploadProgress} 
              status="active" 
              strokeColor="#1890ff"
            />
            <Text type="secondary">上传进度: {uploadProgress}%</Text>
          </div>
        )}
      </Dragger>

      {uploadedFile && !isUploading && (
        <div style={{ marginTop: 16, padding: 12, background: '#f6ffed', borderRadius: 6 }}>
          <Space>
            <Text strong>已选择文件:</Text>
            <Text>{uploadedFile.name}</Text>
            <Text type="secondary">
              ({(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB)
            </Text>
            <Button 
              size="small" 
              icon={<DeleteOutlined />} 
              onClick={() => handleRemove({ uid: uploadedFile.name, name: uploadedFile.name } as UploadFile)}
              danger
            >
              移除
            </Button>
          </Space>
        </div>
      )}
    </Card>
  );
};
```

#### 1.2.2 信号图表组件
```typescript
// src/components/charging/SignalChart.tsx
import React, { useMemo, useState } from 'react';
import { Card, Select, Space, Button, Tooltip, Drawer } from 'antd';
import { LineChartOutlined, BarChartOutlined, InfoCircleOutlined } from '@ant-design/icons';
import Plot from 'react-plotly.js';
import { useChargingStore } from '../../stores/chargingStore';
import type { Visualization } from '../../types/charging';

interface SignalChartProps {
  data: Visualization[];
  signals: string[];
}

export const SignalChart: React.FC<SignalChartProps> = ({
  data,
  signals
}) => {
  const [selectedChart, setSelectedChart] = useState<string>('line_chart');
  const [selectedSignals, setSelectedSignals] = useState<string[]>(signals.slice(0, 3));
  const [infoVisible, setInfoVisible] = useState(false);

  const chartData = useMemo(() => {
    const chart = data.find(c => c.type === selectedChart);
    if (!chart) return null;

    switch (selectedChart) {
      case 'line_chart':
        return generateLineChartData(chart, selectedSignals);
      case 'histogram':
        return generateHistogramData(chart);
      case 'correlation_matrix':
        return generateCorrelationData(chart);
      default:
        return null;
    }
  }, [data, selectedChart, selectedSignals]);

  const generateLineChartData = (chart: Visualization, signals: string[]) => {
    if (!chart.data || !Array.isArray(chart.data)) return null;

    const traces = signals.map((signal, index) => {
      const color = getSignalColor(signal, index);
      return {
        x: chart.data.map((d: any) => d.timestamp),
        y: chart.data.map((d: any) => d[signal] || null),
        type: 'scatter',
        mode: 'lines',
        name: signal,
        line: { color, width: 2 },
        hovertemplate: '%{x}<br>%{y:.2f}<extra></extra>'
      };
    });

    return {
      data: traces,
      layout: {
        title: '信号时间序列图',
        xaxis: { title: '时间' },
        yaxis: { title: '信号值' },
        hovermode: 'x unified',
        showlegend: true,
        height: 400
      },
      config: { responsive: true }
    };
  };

  const generateHistogramData = (chart: Visualization) => {
    if (!chart.data || !Array.isArray(chart.data)) return null;

    const values = chart.data.map((d: any) => d.value);
    
    return {
      data: [{
        type: 'histogram',
        x: values,
        nbinsx: 30,
        marker: { color: '#1890ff' },
        hovertemplate: '值: %{x}<br>频次: %{y}<extra></extra>'
      }],
      layout: {
        title: '信号值分布',
        xaxis: { title: '信号值' },
        yaxis: { title: '频次' },
        height: 400
      },
      config: { responsive: true }
    };
  };

  const generateCorrelationData = (chart: Visualization) => {
    if (!chart.data) return null;

    const { matrix, labels } = chart.data;
    
    return {
      data: [{
        type: 'heatmap',
        z: matrix,
        x: labels,
        y: labels,
        colorscale: [
          [0, '#ff4d4f'],
          [0.5, '#ffffff'],
          [1, '#52c41a']
        ],
        hovertemplate: 'x: %{x}<br>y: %{y}<br>相关性: %{z:.3f}<extra></extra>'
      }],
      layout: {
        title: '信号相关性矩阵',
        xaxis: { title: '信号' },
        yaxis: { title: '信号' },
        height: 400
      },
      config: { responsive: true }
    };
  };

  return (
    <Card 
      title={
        <Space>
          <LineChartOutlined />
          信号可视化
          <Tooltip title="查看图表详细信息">
            <Button 
              size="small" 
              icon={<InfoCircleOutlined />} 
              onClick={() => setInfoVisible(true)}
            />
          </Tooltip>
        </Space>
      }
      extra={
        <Space>
          <Select
            value={selectedChart}
            onChange={setSelectedChart}
            style={{ width: 120 }}
          >
            <Select.Option value="line_chart">时间序列</Select.Option>
            <Select.Option value="histogram">分布直方图</Select.Option>
            <Select.Option value="correlation_matrix">相关性矩阵</Select.Option>
          </Select>
          
          {selectedChart === 'line_chart' && (
            <Select
              mode="multiple"
              value={selectedSignals}
              onChange={setSelectedSignals}
              style={{ width: 200 }}
              placeholder="选择要显示的信号"
            >
              {signals.map(signal => (
                <Select.Option key={signal} value={signal}>
                  {signal}
                </Select.Option>
              ))}
            </Select>
          )}
        </Space>
      }
    >
      {chartData ? (
        <Plot
          data={chartData.data}
          layout={chartData.layout}
          config={chartData.config}
          style={{ width: '100%', height: '400px' }}
        />
      ) : (
        <div style={{ 
          height: 400, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          color: '#999'
        }}>
          暂无数据可显示
        </div>
      )}

      <Drawer
        title="图表说明"
        placement="right"
        width={400}
        open={infoVisible}
        onClose={() => setInfoVisible(false)}
      >
        <div>
          <h4>时间序列图</h4>
          <p>显示选定信号随时间的变化趋势，支持多条信号同时显示和比较。</p>
          
          <h4>分布直方图</h4>
          <p>显示信号值的分布情况，帮助识别数据的统计特征。</p>
          
          <h4>相关性矩阵</h4>
          <p>显示不同信号之间的相关性，红色表示负相关，绿色表示正相关。</p>
        </div>
      </Drawer>
    </Card>
  );
};

// 工具函数
const getSignalColor = (signal: string, index: number): string => {
  const colors = [
    '#1890ff', '#52c41a', '#faad14', '#ff4d4f', '#722ed1',
    '#13c2c2', '#eb2f96', '#fa8c16', '#a0d911', '#fadb14'
  ];
  return colors[index % colors.length];
};
```

#### 1.2.3 分析结果组件
```typescript
// src/components/charging/AnalysisResults.tsx
import React, { useState } from 'react';
import { 
  Card, 
  Collapse, 
  Tag, 
  Space, 
  Progress, 
  Button, 
  Table, 
  Drawer,
  Typography,
  Row,
  Col,
  Alert
} from 'antd';
import { 
  CheckCircleOutlined, 
  ExclamationCircleOutlined, 
  InfoCircleOutlined,
  DownloadOutlined,
  EyeOutlined
} from '@ant-design/icons';
import type { AnalysisResult } from '../../types/charging';

const { Panel } = Collapse;
const { Title, Paragraph, Text } = Typography;

interface AnalysisResultsProps {
  results: AnalysisResult[];
}

export const AnalysisResults: React.FC<AnalysisResultsProps> = ({
  results
}) => {
  const [selectedResult, setSelectedResult] = useState<AnalysisResult | null>(null);
  const [detailVisible, setDetailVisible] = useState(false);

  const getResultIcon = (type: string) => {
    switch (type) {
      case 'charging_efficiency':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'anomaly_detection':
        return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
      case 'system_health':
        return <InfoCircleOutlined style={{ color: '#1890ff' }} />;
      default:
        return <InfoCircleOutlined />;
    }
  };

  const getResultColor = (type: string): string => {
    switch (type) {
      case 'charging_efficiency':
        return 'success';
      case 'anomaly_detection':
        return 'warning';
      case 'system_health':
        return 'processing';
      default:
        return 'default';
    }
  };

  const handleViewDetail = (result: AnalysisResult) => {
    setSelectedResult(result);
    setDetailVisible(true);
  };

  const formatMetrics = (metrics: any) => {
    if (!metrics) return null;

    return (
      <Table
        dataSource={Object.entries(metrics).map(([key, value]) => ({
          key,
          metric: key,
          value: typeof value === 'number' ? value.toFixed(3) : value
        }))}
        columns={[
          { title: '指标', dataIndex: 'metric', key: 'metric' },
          { title: '值', dataIndex: 'value', key: 'value' }
        ]}
        pagination={false}
        size="small"
      />
    );
  };

  return (
    <div className="analysis-results">
      <Card 
        title={
          <Space>
            <CheckCircleOutlined style={{ color: '#52c41a' }} />
            分析结果
            <Tag color="blue">{results.length} 项发现</Tag>
          </Space>
        }
        extra={
          <Space>
            <Button icon={<DownloadOutlined />}>
              导出报告
            </Button>
          </Space>
        }
      >
        <Collapse defaultActiveKey={results.map((_, index) => index.toString())}>
          {results.map((result, index) => (
            <Panel
              key={index.toString()}
              header={
                <Space>
                  {getResultIcon(result.type)}
                  <Text strong>{result.title}</Text>
                  <Tag color={getResultColor(result.type)}>
                    置信度: {(result.confidence * 100).toFixed(1)}%
                  </Tag>
                </Space>
              }
              extra={
                <Space>
                  <Button 
                    size="small" 
                    icon={<EyeOutlined />}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleViewDetail(result);
                    }}
                  >
                    详情
                  </Button>
                </Space>
              }
            >
              <Row gutter={[16, 16]}>
                <Col span={24}>
                  <Paragraph>{result.content}</Paragraph>
                </Col>
                
                {result.metrics && Object.keys(result.metrics).length > 0 && (
                  <Col span={24}>
                    <Title level={5}>关键指标</Title>
                    {formatMetrics(result.metrics)}
                  </Col>
                )}
                
                {result.confidence < 0.7 && (
                  <Col span={24}>
                    <Alert
                      message="置信度较低"
                      description="该分析结果的置信度较低，建议结合其他信息进行判断。"
                      type="warning"
                      showIcon
                    />
                  </Col>
                )}
              </Row>
            </Panel>
          ))}
        </Collapse>
      </Card>

      {/* 详情抽屉 */}
      <Drawer
        title={selectedResult?.title}
        placement="right"
        width={600}
        open={detailVisible}
        onClose={() => setDetailVisible(false)}
      >
        {selectedResult && (
          <div>
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <div>
                <Title level={5}>分析内容</Title>
                <Paragraph>{selectedResult.content}</Paragraph>
              </div>

              {selectedResult.metrics && (
                <div>
                  <Title level={5}>详细指标</Title>
                  {formatMetrics(selectedResult.metrics)}
                </div>
              )}

              <div>
                <Title level={5}>置信度评估</Title>
                <Progress 
                  percent={selectedResult.confidence * 100}
                  status={selectedResult.confidence > 0.8 ? 'success' : 'active'}
                  strokeColor={
                    selectedResult.confidence > 0.8 ? '#52c41a' :
                    selectedResult.confidence > 0.6 ? '#1890ff' : '#faad14'
                  }
                />
              </div>

              <div>
                <Title level={5}>技术细节</Title>
                <Alert
                  message="技术实现"
                  description="该分析基于LangGraph工作流，通过流程控制模型、RAG检索和LLM分析的综合处理得到结果。"
                  type="info"
                  showIcon
                />
              </div>
            </Space>
          </div>
        )}
      </Drawer>
    </div>
  );
};
```

### 1.3 后端实现

#### 1.3.1 主分析服务
```python
# src/services/charging_analysis_service.py
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from .langgraph_service import ChargingAnalysisWorkflow
from .file_service import FileService
from .websocket_service import WebSocketService
from ..models.charging_analysis import ChargingAnalysis
from ..repositories.charging_analysis import ChargingAnalysisRepository

class ChargingAnalysisService:
    def __init__(
        self,
        db: AsyncSession,
        workflow: ChargingAnalysisWorkflow,
        file_service: FileService,
        websocket_service: WebSocketService,
        repo: ChargingAnalysisRepository
    ):
        self.db = db
        self.workflow = workflow
        self.file_service = file_service
        self.websocket_service = websocket_service
        self.repo = repo

    async def create_analysis(
        self, 
        user_id: int, 
        name: str, 
        file_path: str, 
        file_size: int
    ) -> ChargingAnalysis:
        """创建新的分析任务"""
        analysis = ChargingAnalysis(
            name=name,
            file_path=file_path,
            file_size=file_size,
            user_id=user_id,
            status='pending',
            created_at=datetime.utcnow()
        )
        
        return await self.repo.create(analysis)

    async def start_analysis(self, analysis_id: str) -> Dict[str, Any]:
        """启动分析任务"""
        analysis = await self.repo.get_by_id(analysis_id)
        if not analysis:
            raise ValueError("分析任务不存在")
        
        if analysis.status != 'pending':
            raise ValueError("分析任务状态不正确")
        
        # 更新状态为处理中
        analysis.status = 'processing'
        analysis.started_at = datetime.utcnow()
        await self.repo.update(analysis)
        
        # 异步执行分析
        asyncio.create_task(self._execute_analysis(analysis_id))
        
        return {
            "task_id": f"analysis_{analysis_id}",
            "status": "started",
            "websocket_url": f"/ws/analysis/analysis_{analysis_id}"
        }

    async def _execute_analysis(self, analysis_id: str):
        """执行分析任务"""
        try:
            analysis = await self.repo.get_by_id(analysis_id)
            if not analysis:
                return
            
            # 初始化工作流状态
            initial_state = {
                "analysis_id": analysis_id,
                "file_path": analysis.file_path,
                "file_name": analysis.name,
                "file_size": analysis.file_size,
                "user_id": analysis.user_id,
                "start_time": datetime.utcnow(),
                "progress_callback": self._create_progress_callback(analysis_id)
            }
            
            # 执行LangGraph工作流
            final_state = await self.workflow.execute(initial_state)
            
            # 更新分析结果
            analysis.status = 'completed'
            analysis.completed_at = datetime.utcnow()
            analysis.result_data = final_state.get('final_report')
            analysis.progress = 100.0
            
            await self.repo.update(analysis)
            
            # 发送完成通知
            await self.websocket_service.send_to_task(
                f"analysis_{analysis_id}",
                {
                    "type": "analysis_completed",
                    "analysis_id": analysis_id,
                    "result": final_state.get('final_report')
                }
            )
            
        except Exception as e:
            # 处理错误
            analysis = await self.repo.get_by_id(analysis_id)
            if analysis:
                analysis.status = 'failed'
                analysis.error_message = str(e)
                analysis.completed_at = datetime.utcnow()
                await self.repo.update(analysis)
                
                await self.websocket_service.send_to_task(
                    f"analysis_{analysis_id}",
                    {
                        "type": "analysis_failed",
                        "analysis_id": analysis_id,
                        "error": str(e)
                    }
                )

    async def _create_progress_callback(self, analysis_id: str):
        """创建进度回调函数"""
        async def callback(step: str, progress: float, message: str = None):
            await self.websocket_service.send_to_task(
                f"analysis_{analysis_id}",
                {
                    "type": "progress_update",
                    "progress": progress,
                    "current_step": message or step,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # 更新数据库中的进度
            analysis = await self.repo.get_by_id(analysis_id)
            if analysis:
                analysis.progress = progress
                await self.repo.update(analysis)
        
        return callback

    async def get_analysis_results(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """获取分析结果"""
        analysis = await self.repo.get_by_id(analysis_id)
        if not analysis:
            return None
        
        if analysis.status != 'completed':
            return {
                "status": analysis.status,
                "progress": analysis.progress,
                "error": analysis.error_message
            }
        
        # 格式化返回结果
        result_data = analysis.result_data or {}
        return {
            "analysis_id": analysis_id,
            "status": analysis.status,
            "completed_at": analysis.completed_at.isoformat(),
            "results": result_data.get('results', []),
            "visualizations": result_data.get('visualizations', []),
            "signals": result_data.get('signals', []),
            "summary": result_data.get('summary', '')
        }

    async def get_analysis_history(
        self, 
        user_id: int, 
        page: int = 1, 
        limit: int = 20
    ) -> Dict[str, Any]:
        """获取分析历史"""
        analyses, total = await self.repo.get_by_user(
            user_id, 
            page=page, 
            limit=limit
        )
        
        return {
            "analyses": [
                {
                    "id": analysis.id,
                    "name": analysis.name,
                    "status": analysis.status,
                    "progress": analysis.progress,
                    "file_name": analysis.name,
                    "created_at": analysis.created_at.isoformat(),
                    "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None
                }
                for analysis in analyses
            ],
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "total_pages": (total + limit - 1) // limit
            }
        }
```

#### 1.3.2 文件处理服务
```python
# src/services/file_service.py
import os
import uuid
from typing import Tuple
from fastapi import UploadFile
from pathlib import Path

class FileService:
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
        # 允许的文件扩展名
        self.allowed_extensions = {'.blf', '.csv', '.xlsx'}
        
        # 最大文件大小 (100MB)
        self.max_file_size = 100 * 1024 * 1024

    async def save_uploaded_file(self, file: UploadFile) -> Tuple[str, int]:
        """保存上传的文件"""
        # 生成唯一文件名
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in self.allowed_extensions:
            raise ValueError(f"不支持的文件类型: {file_extension}")
        
        # 检查文件大小
        content = await file.read()
        if len(content) > self.max_file_size:
            raise ValueError(f"文件大小超过限制: {len(content)} bytes")
        
        # 保存文件
        filename = f"{uuid.uuid4()}{file_extension}"
        file_path = self.upload_dir / filename
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        return str(file_path), len(content)

    def delete_file(self, file_path: str):
        """删除文件"""
        path = Path(file_path)
        if path.exists():
            path.unlink()

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """获取文件信息"""
        path = Path(file_path)
        if not path.exists():
            return None
        
        stat = path.stat()
        return {
            "filename": path.name,
            "size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime),
            "modified_at": datetime.fromtimestamp(stat.st_mtime),
            "extension": path.suffix.lower()
        }
```

### 1.4 数据模型

```python
# src/models/charging_analysis.py
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class ChargingAnalysis(Base):
    __tablename__ = "charging_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    file_path = Column(String(255), nullable=False)
    file_size = Column(Integer)
    file_type = Column(String(20), default='blf')
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
    progress = Column(Float, default=0.0)
    result_data = Column(Text)  # JSON数据
    error_message = Column(Text)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # 关系
    user = relationship("User", back_populates="analyses")
    results = relationship("AnalysisResult", back_populates="analysis", cascade="all, delete-orphan")

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("charging_analyses.id"))
    result_type = Column(String(50), nullable=False)
    title = Column(String(200), nullable=False)
    content = Column(Text)
    confidence_score = Column(Float)
    metadata = Column(Text)  # JSON数据
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    analysis = relationship("ChargingAnalysis", back_populates="results")
```

## 2. 用户认证模块详细设计

### 2.1 JWT 认证实现

```python
# src/services/auth_service.py
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.user import User
from ..repositories.user import UserRepository
from ..schemas.auth import LoginRequest, RegisterRequest, TokenResponse

# JWT配置
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 密码加密
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self, db: AsyncSession, user_repo: UserRepository):
        self.db = db
        self.user_repo = user_repo

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """获取密码哈希"""
        return pwd_context.hash(password)

    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """创建访问令牌"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """认证用户"""
        user = await self.user_repo.get_by_username(username)
        if not user or not self.verify_password(password, user.password_hash):
            return None
        return user

    async def register_user(self, user_data: RegisterRequest) -> User:
        """注册用户"""
        # 检查用户名是否已存在
        existing_user = await self.user_repo.get_by_username(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已存在"
            )
        
        # 检查邮箱是否已存在
        existing_email = await self.user_repo.get_by_email(user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已存在"
            )
        
        # 创建新用户
        hashed_password = self.get_password_hash(user_data.password)
        user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=hashed_password,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            role="user",  # 默认为普通用户
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        return await self.user_repo.create(user)

    async def login(self, login_data: LoginRequest) -> TokenResponse:
        """用户登录"""
        user = await self.authenticate_user(login_data.username, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户账户已被禁用"
            )
        
        access_token = self.create_access_token(
            data={"sub": user.username, "user_id": user.id, "role": user.role}
        )
        
        # 更新最后登录时间
        user.last_login = datetime.utcnow()
        await self.user_repo.update(user)
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user={
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
        )

    async def verify_token(self, token: str) -> Optional[User]:
        """验证令牌"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            user_id: int = payload.get("user_id")
            
            if username is None or user_id is None:
                return None
            
            user = await self.user_repo.get_by_id(user_id)
            if user is None or not user.is_active:
                return None
            
            return user
        except JWTError:
            return None
```

### 2.2 前端认证组件

```typescript
// src/components/auth/LoginForm.tsx
import React, { useState } from 'react';
import { Form, Input, Button, Card, Typography, Space, Alert } from 'antd';
import { UserOutlined, LockOutlined } from '@ant-design/icons';
import { useAuthStore } from '../../stores/authStore';
import { Link } from 'react-router-dom';

const { Title, Text } = Typography;

interface LoginFormData {
  username: string;
  password: string;
}

export const LoginForm: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { login } = useAuthStore();

  const onFinish = async (values: LoginFormData) => {
    setLoading(true);
    setError(null);

    try {
      await login(values);
    } catch (err: any) {
      setError(err.response?.data?.detail || '登录失败，请检查用户名和密码');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    }}>
      <Card style={{ width: 400, boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)' }}>
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <div style={{ textAlign: 'center' }}>
            <Title level={2}>登录系统</Title>
            <Text type="secondary">请输入您的账户信息</Text>
          </div>

          {error && (
            <Alert
              message={error}
              type="error"
              closable
              onClose={() => setError(null)}
            />
          )}

          <Form
            name="login"
            onFinish={onFinish}
            autoComplete="off"
            layout="vertical"
          >
            <Form.Item
              name="username"
              label="用户名"
              rules={[
                { required: true, message: '请输入用户名' },
                { min: 3, message: '用户名至少3个字符' }
              ]}
            >
              <Input 
                prefix={<UserOutlined />} 
                placeholder="用户名"
                size="large"
              />
            </Form.Item>

            <Form.Item
              name="password"
              label="密码"
              rules={[
                { required: true, message: '请输入密码' },
                { min: 6, message: '密码至少6个字符' }
              ]}
            >
              <Input.Password 
                prefix={<LockOutlined />} 
                placeholder="密码"
                size="large"
              />
            </Form.Item>

            <Form.Item>
              <Button 
                type="primary" 
                htmlType="submit" 
                loading={loading}
                size="large"
                block
              >
                登录
              </Button>
            </Form.Item>

            <div style={{ textAlign: 'center' }}>
              <Text>
                还没有账户？ 
                <Link to="/register">立即注册</Link>
              </Text>
            </div>
          </Form>
        </Space>
      </Card>
    </div>
  );
};
```

```typescript
// src/components/auth/ProtectedRoute.tsx
import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../../stores/authStore';
import { LoadingScreen } from '../common/LoadingScreen';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: string;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredRole
}) => {
  const { isAuthenticated, user, isLoading } = useAuthStore();
  const location = useLocation();

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (requiredRole && user?.role !== requiredRole) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
};
```

## 3. RAG管理模块详细设计

### 3.1 知识库管理组件

```typescript
// src/components/rag/KnowledgeBaseManager.tsx
import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Table, 
  Button, 
  Space, 
  Modal, 
  Form, 
  Input, 
  Select, 
  Upload, 
  message, 
  Popconfirm,
  Tag,
  Progress,
  Typography,
  Row,
  Col
} from 'antd';
import { 
  PlusOutlined, 
  UploadOutlined, 
  DeleteOutlined, 
  SearchOutlined,
  EyeOutlined,
  EditOutlined
} from '@ant-design/icons';
import { useRAGStore } from '../../stores/ragStore';
import type { KnowledgeCollection, KnowledgeDocument } from '../../types/rag';

const { TextArea } = Input;
const { Title, Text } = Typography;

export const KnowledgeBaseManager: React.FC = () => {
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [uploadModalVisible, setUploadModalVisible] = useState(false);
  const [selectedCollection, setSelectedCollection] = useState<KnowledgeCollection | null>(null);
  const [searchText, setSearchText] = useState('');

  const {
    collections,
    documents,
    isLoading,
    loadCollections,
    loadDocuments,
    createCollection,
    uploadDocument,
    deleteCollection,
    searchKnowledge
  } = useRAGStore();

  useEffect(() => {
    loadCollections();
  }, [loadCollections]);

  const handleCreateCollection = async (values: any) => {
    try {
      await createCollection(values);
      message.success('知识库创建成功');
      setCreateModalVisible(false);
      loadCollections();
    } catch (error) {
      message.error('创建知识库失败');
    }
  };

  const handleUploadDocument = async (values: any) => {
    if (!selectedCollection || !values.file) {
      message.error('请选择文件');
      return;
    }

    try {
      await uploadDocument({
        collectionId: selectedCollection.id,
        file: values.file,
        metadata: {
          category: values.category,
          tags: values.tags
        }
      });
      message.success('文档上传成功');
      setUploadModalVisible(false);
      loadDocuments(selectedCollection.id);
    } catch (error) {
      message.error('文档上传失败');
    }
  };

  const handleSearch = async () => {
    if (!selectedCollection || !searchText) return;

    try {
      const results = await searchKnowledge({
        collectionId: selectedCollection.id,
        query: searchText,
        topK: 10
      });
      console.log('搜索结果:', results);
      message.success(`找到 ${results.length} 个相关文档`);
    } catch (error) {
      message.error('搜索失败');
    }
  };

  const collectionColumns = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: KnowledgeCollection) => (
        <Space>
          <Text strong>{text}</Text>
          <Tag color={record.is_active ? 'green' : 'red'}>
            {record.is_active ? '启用' : '禁用'}
          </Tag>
        </Space>
      )
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description'
    },
    {
      title: '文档数量',
      dataIndex: 'document_count',
      key: 'document_count',
      render: (count: number) => (
        <Tag color="blue">{count}</Tag>
      )
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => new Date(date).toLocaleString()
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: KnowledgeCollection) => (
        <Space>
          <Button 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedCollection(record);
              loadDocuments(record.id);
            }}
          >
            查看
          </Button>
          <Button 
            size="small" 
            icon={<EditOutlined />}
            onClick={() => {
              setSelectedCollection(record);
              setUploadModalVisible(true);
            }}
          >
            上传
          </Button>
          <Popconfirm
            title="确定要删除这个知识库吗？"
            onConfirm={() => deleteCollection(record.id)}
          >
            <Button size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      )
    }
  ];

  const documentColumns = [
    {
      title: '文件名',
      dataIndex: 'filename',
      key: 'filename'
    },
    {
      title: '类型',
      dataIndex: 'file_type',
      key: 'file_type',
      render: (type: string) => (
        <Tag>{type}</Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'upload_status',
      key: 'upload_status',
      render: (status: string) => {
        const statusConfig = {
          uploading: { color: 'processing', text: '上传中' },
          processing: { color: 'processing', text: '处理中' },
          completed: { color: 'success', text: '已完成' },
          failed: { color: 'error', text: '失败' }
        };
        const config = statusConfig[status as keyof typeof statusConfig];
        return <Tag color={config.color}>{config.text}</Tag>;
      }
    },
    {
      title: '分块数',
      dataIndex: 'chunk_count',
      key: 'chunk_count'
    },
    {
      title: '上传时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => new Date(date).toLocaleString()
    }
  ];

  return (
    <div className="knowledge-base-manager">
      <Row gutter={[16, 16]}>
        <Col span={selectedCollection ? 12 : 24}>
          <Card
            title="知识库列表"
            extra={
              <Button 
                type="primary" 
                icon={<PlusOutlined />}
                onClick={() => setCreateModalVisible(true)}
              >
                创建知识库
              </Button>
            }
          >
            <Table
              dataSource={collections}
              columns={collectionColumns}
              rowKey="id"
              loading={isLoading}
              pagination={{ pageSize: 10 }}
              onRow={(record) => ({
                onClick: () => {
                  setSelectedCollection(record);
                  loadDocuments(record.id);
                }
              })}
            />
          </Card>
        </Col>

        {selectedCollection && (
          <Col span={12}>
            <Card
              title={
                <Space>
                  <Text strong>{selectedCollection.name}</Text>
                  <Button
                    size="small"
                    icon={<UploadOutlined />}
                    onClick={() => setUploadModalVisible(true)}
                  >
                    上传文档
                  </Button>
                </Space>
              }
              extra={
                <Space>
                  <Input.Search
                    placeholder="搜索知识..."
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                    onSearch={handleSearch}
                    style={{ width: 200 }}
                  />
                </Space>
              }
            >
              <Table
                dataSource={documents}
                columns={documentColumns}
                rowKey="id"
                pagination={{ pageSize: 5 }}
              />
            </Card>
          </Col>
        )}
      </Row>

      {/* 创建知识库对话框 */}
      <Modal
        title="创建知识库"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        footer={null}
      >
        <Form
          layout="vertical"
          onFinish={handleCreateCollection}
        >
          <Form.Item
            name="name"
            label="名称"
            rules={[{ required: true, message: '请输入知识库名称' }]}
          >
            <Input placeholder="知识库名称" />
          </Form.Item>

          <Form.Item
            name="description"
            label="描述"
          >
            <TextArea 
              rows={3} 
              placeholder="知识库描述（可选）"
            />
          </Form.Item>

          <Form.Item
            name="collection_type"
            label="类型"
            initialValue="document"
          >
            <Select>
              <Select.Option value="document">文档</Select.Option>
              <Select.Option value="guide">指南</Select.Option>
              <Select.Option value="faq">常见问题</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                创建
              </Button>
              <Button onClick={() => setCreateModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 上传文档对话框 */}
      <Modal
        title={`上传文档到 ${selectedCollection?.name}`}
        open={uploadModalVisible}
        onCancel={() => setUploadModalVisible(false)}
        footer={null}
      >
        <Form
          layout="vertical"
          onFinish={handleUploadDocument}
        >
          <Form.Item
            name="file"
            label="选择文件"
            rules={[{ required: true, message: '请选择文件' }]}
          >
            <Upload
              accept=".pdf,.doc,.docx,.txt"
              beforeUpload={() => false}
              maxCount={1}
            >
              <Button icon={<UploadOutlined />}>选择文件</Button>
            </Upload>
          </Form.Item>

          <Form.Item
            name="category"
            label="分类"
          >
            <Select placeholder="选择文档分类">
              <Select.Option value="troubleshooting">故障排除</Select.Option>
              <Select.Option value="technical">技术文档</Select.Option>
              <Select.Option value="manual">用户手册</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="tags"
            label="标签"
          >
            <Select
              mode="tags"
              placeholder="输入标签"
            />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                上传
              </Button>
              <Button onClick={() => setUploadModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};
```

### 3.2 RAG 服务实现

```python
# src/services/rag_service.py
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import json

from ..models.rag import KnowledgeCollection, KnowledgeDocument
from ..repositories.rag import RAGRepository
from ..services.vector_store_service import VectorStoreService
from ..schemas.rag import CreateCollectionRequest, UploadDocumentRequest

class RAGService:
    def __init__(
        self,
        db: AsyncSession,
        rag_repo: RAGRepository,
        vector_store: VectorStoreService
    ):
        self.db = db
        self.rag_repo = rag_repo
        self.vector_store = vector_store

    async def create_collection(self, data: CreateCollectionRequest, user_id: int) -> KnowledgeCollection:
        """创建知识库集合"""
        # 在向量数据库中创建集合
        chroma_collection = await self.vector_store.create_collection(
            name=f"collection_{hash(data.name)}",
            metadata={
                "name": data.name,
                "description": data.description,
                "type": data.collection_type,
                "created_by": user_id
            }
        )
        
        # 在关系数据库中记录
        collection = KnowledgeCollection(
            name=data.name,
            description=data.description,
            collection_type=data.collection_type,
            chroma_collection_id=chroma_collection.name,
            document_count=0,
            embedding_model="bge-base-zh-v1.5",
            is_active=True,
            created_by=user_id
        )
        
        return await self.rag_repo.create_collection(collection)

    async def upload_document(self, data: UploadDocumentRequest, user_id: int) -> KnowledgeDocument:
        """上传文档到知识库"""
        collection = await self.rag_repo.get_collection(data.collection_id)
        if not collection:
            raise ValueError("知识库不存在")
        
        # 保存文档文件
        file_info = await self._save_uploaded_file(data.file)
        
        # 创建文档记录
        document = KnowledgeDocument(
            collection_id=data.collection_id,
            filename=data.file.filename,
            file_path=file_info['path'],
            file_size=file_info['size'],
            file_type=data.file.content_type or 'unknown',
            upload_status='uploading',
            uploaded_by=user_id
        )
        
        document = await self.rag_repo.create_document(document)
        
        # 异步处理文档
        asyncio.create_task(self._process_document(document.id, collection, data.metadata or {}))
        
        return document

    async def _process_document(self, document_id: int, collection: KnowledgeCollection, metadata: Dict[str, Any]):
        """异步处理文档"""
        try:
            # 获取文档信息
            document = await self.rag_repo.get_document(document_id)
            if not document:
                return
            
            # 文档内容提取和分块
            chunks = await self._extract_and_chunk_document(document.file_path)
            
            # 批量添加到向量数据库
            await self.vector_store.add_documents(
                collection_name=collection.chroma_collection_id,
                documents=[
                    {
                        "content": chunk['content'],
                        "metadata": {
                            **metadata,
                            "document_id": document_id,
                            "chunk_index": i,
                            "filename": document.filename
                        }
                    }
                    for i, chunk in enumerate(chunks)
                ]
            )
            
            # 更新文档状态
            document.upload_status = 'completed'
            document.chunk_count = len(chunks)
            document.upload_processing_time = time.time() - start_time
            
            await self.rag_repo.update_document(document)
            
            # 更新集合统计
            collection.document_count += 1
            await self.rag_repo.update_collection(collection)
            
        except Exception as e:
            # 处理失败
            document = await self.rag_repo.get_document(document_id)
            if document:
                document.upload_status = 'failed'
                document.processing_error = str(e)
                await self.rag_repo.update_document(document)

    async def search_knowledge(
        self,
        collection_id: int,
        query: str,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """搜索知识"""
        collection = await self.rag_repo.get_collection(collection_id)
        if not collection:
            raise ValueError("知识库不存在")
        
        # 在向量数据库中搜索
        results = await self.vector_store.search(
            collection_name=collection.chroma_collection_id,
            query=query,
            top_k=top_k
        )
        
        # 过滤低质量结果
        filtered_results = [
            {
                "content": result["content"],
                "metadata": result["metadata"],
                "score": result["score"],
                "chunk_index": result["metadata"].get("chunk_index", 0)
            }
            for result in results
            if result["score"] >= min_score
        ]
        
        # 记录检索历史
        await self._record_query_history(collection_id, query, len(filtered_results))
        
        return filtered_results

    async def delete_collection(self, collection_id: int, user_id: int) -> bool:
        """删除知识库集合"""
        collection = await self.rag_repo.get_collection(collection_id)
        if not collection or collection.created_by != user_id:
            raise ValueError("知识库不存在或无权限删除")
        
        # 从向量数据库中删除
        await self.vector_store.delete_collection(collection.chroma_collection_id)
        
        # 从关系数据库中删除
        await self.rag_repo.delete_collection(collection_id)
        
        return True

    async def _save_uploaded_file(self, file) -> Dict[str, Any]:
        """保存上传文件"""
        import uuid
        from pathlib import Path
        
        # 生成唯一文件名
        extension = Path(file.filename).suffix.lower()
        filename = f"{uuid.uuid4()}{extension}"
        file_path = f"uploads/rag/{filename}"
        
        # 保存文件
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "path": file_path,
            "size": len(content)
        }

    async def _extract_and_chunk_document(self, file_path: str) -> List[Dict[str, str]]:
        """提取文档内容并分块"""
        # 根据文件类型选择处理器
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            chunks = await self._process_pdf(file_path)
        elif file_ext in ['.doc', '.docx']:
            chunks = await self._process_docx(file_path)
        elif file_ext == '.txt':
            chunks = await self._process_txt(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_ext}")
        
        return chunks

    async def _process_pdf(self, file_path: str) -> List[Dict[str, str]]:
        """处理PDF文件"""
        import PyPDF2
        
        chunks = []
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # 分块处理
            chunk_size = 1000
            overlap = 200
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size].strip()
                if chunk:
                    chunks.append({
                        "content": chunk,
                        "page": i // chunk_size + 1
                    })
        
        return chunks

    async def _record_query_history(self, collection_id: int, query: str, result_count: int):
        """记录检索历史"""
        # 这里可以实现检索历史的记录逻辑
        pass
```

由于篇幅限制，这里展示的是核心模块的部分实现。完整的模块实现文档还包括：

## 4. 训练管理模块
- 训练任务管理组件
- 数据集管理组件
- 模型版本管理
- 训练进度监控
- 指标可视化

## 5. 日志管理模块
- 日志查看组件
- 过滤器组件
- 日志导出功能
- 统计图表组件

## 6. 扩展接口设计
- 热管理Agent接口
- 能量管理Agent接口
- 行驶状态Agent接口
- Agent协同工作机制

每个模块都包含完整的前端组件设计、后端服务实现、数据模型定义和API接口设计，为后续的代码生成提供了详细的技术规范。