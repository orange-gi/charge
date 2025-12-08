import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link } from 'react-router-dom';
import { ConfigProvider, Layout as AntLayout, Menu, Button, message, Select, Spin, Card, Space, Typography, Progress } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { UserOutlined, FileTextOutlined, DatabaseOutlined, LogoutOutlined, UploadOutlined, FileOutlined, CloseOutlined, MenuFoldOutlined, MenuUnfoldOutlined, ThunderboltOutlined, PlusOutlined, MessageOutlined, CheckCircleOutlined, ReloadOutlined, DeleteOutlined, PlayCircleOutlined, CheckOutlined, CodeOutlined, ControlOutlined, SearchOutlined, ToolOutlined, RobotOutlined } from '@ant-design/icons';
import { useAuthStore } from './stores/authStore';
import './styles/globals.css';

const { Header, Content, Sider } = AntLayout;

// Simple Login Page
const LoginPage = () => {
  const [email, setEmail] = React.useState('');
  const [password, setPassword] = React.useState('');
  const { login, isLoading } = useAuthStore();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await login({ email, password });
      message.success('登录成功');
    } catch (error: any) {
      message.error(error.message || '登录失败');
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f0f2f5' }}>
      <div style={{ background: 'white', padding: '40px', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)', width: '400px' }}>
        <h1 style={{ marginBottom: '24px', textAlign: 'center' }}>充电分析系统</h1>
        <form onSubmit={handleLogin}>
          <div style={{ marginBottom: '16px' }}>
            <input
              type="email"
              placeholder="邮箱"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              style={{ width: '100%', padding: '10px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
              required
            />
          </div>
          <div style={{ marginBottom: '16px' }}>
            <input
              type="password"
              placeholder="密码"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={{ width: '100%', padding: '10px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
              required
            />
          </div>
          <Button type="primary" htmlType="submit" loading={isLoading} block size="large">
            登录
          </Button>
          <div style={{ marginTop: '16px', textAlign: 'center' }}>
            <Link to="/register">没有账号？立即注册</Link>
          </div>
        </form>
      </div>
    </div>
  );
};

// Simple Register Page
const RegisterPage = () => {
  const [email, setEmail] = React.useState('');
  const [password, setPassword] = React.useState('');
  const [username, setUsername] = React.useState('');
  const { register, isLoading } = useAuthStore();

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await register({ email, password, username });
      message.success('注册成功');
    } catch (error: any) {
      message.error(error.message || '注册失败');
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f0f2f5' }}>
      <div style={{ background: 'white', padding: '40px', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)', width: '400px' }}>
        <h1 style={{ marginBottom: '24px', textAlign: 'center' }}>用户注册</h1>
        <form onSubmit={handleRegister}>
          <div style={{ marginBottom: '16px' }}>
            <input
              type="text"
              placeholder="用户名"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              style={{ width: '100%', padding: '10px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
              required
            />
          </div>
          <div style={{ marginBottom: '16px' }}>
            <input
              type="email"
              placeholder="邮箱"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              style={{ width: '100%', padding: '10px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
              required
            />
          </div>
          <div style={{ marginBottom: '16px' }}>
            <input
              type="password"
              placeholder="密码"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={{ width: '100%', padding: '10px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
              required
              minLength={6}
            />
          </div>
          <Button type="primary" htmlType="submit" loading={isLoading} block size="large">
            注册
          </Button>
          <div style={{ marginTop: '16px', textAlign: 'center' }}>
            <Link to="/login">已有账号？立即登录</Link>
          </div>
        </form>
      </div>
    </div>
  );
};

// Main Dashboard
const Dashboard = () => {
  const { user, logout } = useAuthStore();
  const [selectedKey, setSelectedKey] = React.useState('home');

  return (
    <AntLayout style={{ minHeight: '100vh' }}>
      <Header style={{ 
        background: '#fff', 
        borderBottom: '1px solid #e8e8e8',
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between', 
        padding: '0 24px',
        height: '64px',
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 1000,
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '32px', flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', gap: '8px', flex: 1, minWidth: 0 }}>
            <Button 
              type={selectedKey === 'home' ? 'default' : 'text'}
              onClick={() => setSelectedKey('home')}
              style={{ 
                height: '40px',
                border: selectedKey === 'home' ? '1px solid #d9d9d9' : 'none',
                background: selectedKey === 'home' ? '#fafafa' : 'transparent'
              }}
            >
              <FileTextOutlined /> 首页
            </Button>
            <Button 
              type={selectedKey === 'charging' ? 'default' : 'text'}
              onClick={() => setSelectedKey('charging')}
              style={{ 
                height: '40px',
                border: selectedKey === 'charging' ? '1px solid #d9d9d9' : 'none',
                background: selectedKey === 'charging' ? '#fafafa' : 'transparent'
              }}
            >
              <FileTextOutlined /> 充电分析
            </Button>
            <Button 
              type={selectedKey === 'rag' ? 'default' : 'text'}
              onClick={() => setSelectedKey('rag')}
              style={{ 
                height: '40px',
                border: selectedKey === 'rag' ? '1px solid #d9d9d9' : 'none',
                background: selectedKey === 'rag' ? '#fafafa' : 'transparent'
              }}
            >
              <DatabaseOutlined /> RAG管理
            </Button>
            <Button 
              type={selectedKey === 'training' ? 'default' : 'text'}
              onClick={() => setSelectedKey('training')}
              style={{ 
                height: '40px',
                border: selectedKey === 'training' ? '1px solid #d9d9d9' : 'none',
                background: selectedKey === 'training' ? '#fafafa' : 'transparent'
              }}
            >
              <DatabaseOutlined /> 训练管理
            </Button>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span style={{ color: '#1a1a1a', fontSize: '14px' }}>
            <UserOutlined style={{ marginRight: '6px' }} />
          {user?.username || user?.email}
          </span>
          <Button type="text" onClick={() => logout()} style={{ color: '#1a1a1a' }}>
            <LogoutOutlined /> 退出
          </Button>
        </div>
      </Header>
      <Content style={{ padding: '24px', background: '#fff', minHeight: 'calc(100vh - 64px)', marginTop: '64px' }}>
          {selectedKey === 'home' && <HomePage />}
          {selectedKey === 'charging' && <ChargingPage />}
          {selectedKey === 'rag' && <RAGPage />}
          {selectedKey === 'training' && <TrainingPage />}
        </Content>
    </AntLayout>
  );
};

const HomePage = () => {
  const { user, token } = useAuthStore();
  const [stats, setStats] = React.useState({
    totalAnalyses: 0,
    completedAnalyses: 0,
    activeUsers: 0,
    knowledgeDocuments: 0
  });

  const [recentActivities, setRecentActivities] = React.useState<any[]>([]);
  const [loading, setLoading] = React.useState(true);

  // Load real data on mount
  React.useEffect(() => {
    if (token) {
      loadDashboardData();
    }
  }, [token, user]);

  const loadDashboardData = async () => {
    if (!token) {
      return;
    }
    try {
      const { statsService } = await import('./services/statsService');
      
      // Load stats
      const statsData = await statsService.getSystemStats(token);
      setStats(statsData);
      
      // Load recent activities
      const activities = await statsService.getRecentActivities(
        token,
        user?.username || user?.email || '当前用户',
        4
      );
      setRecentActivities(activities);
    } catch (error) {
      console.error('加载仪表板数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      <h1 style={{ fontSize: '32px', fontWeight: '300', color: '#1a1a1a', marginBottom: '12px' }}>欢迎使用充电分析系统</h1>
      <p style={{ marginTop: '16px', fontSize: '15px', color: '#8c8c8c', lineHeight: '1.6' }}>
        智能充电数据分析平台，提供专业的数据处理和AI辅助决策服务
      </p>

      {/* 统计卡片 */}
      <div style={{ 
        marginTop: '48px', 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
        gap: '16px' 
      }}>
        <div style={{ 
          padding: '20px', 
          background: 'white',
          border: '1px solid #e8e8e8',
          borderLeft: '3px solid #2c5aa0',
          borderRadius: '4px',
          color: '#1a1a1a'
        }}>
          <div style={{ fontSize: '12px', color: '#8c8c8c', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>总分析次数</div>
          <div style={{ fontSize: '32px', fontWeight: '300', marginTop: '4px', color: '#1a1a1a' }}>{stats.totalAnalyses}</div>
          <div style={{ fontSize: '11px', marginTop: '12px', color: '#8c8c8c', borderTop: '1px solid #f0f0f0', paddingTop: '12px' }}>本月增长 23%</div>
        </div>

        <div style={{ 
          padding: '20px', 
          background: 'white',
          border: '1px solid #e8e8e8',
          borderLeft: '3px solid #6b7280',
          borderRadius: '4px',
          color: '#1a1a1a'
        }}>
          <div style={{ fontSize: '12px', color: '#8c8c8c', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>已完成分析</div>
          <div style={{ fontSize: '32px', fontWeight: '300', marginTop: '4px', color: '#1a1a1a' }}>{stats.completedAnalyses}</div>
          <div style={{ fontSize: '11px', marginTop: '12px', color: '#8c8c8c', borderTop: '1px solid #f0f0f0', paddingTop: '12px' }}>成功率 91%</div>
        </div>

        <div style={{ 
          padding: '20px', 
          background: 'white',
          border: '1px solid #e8e8e8',
          borderLeft: '3px solid #6b7280',
          borderRadius: '4px',
          color: '#1a1a1a'
        }}>
          <div style={{ fontSize: '12px', color: '#8c8c8c', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>活跃用户</div>
          <div style={{ fontSize: '32px', fontWeight: '300', marginTop: '4px', color: '#1a1a1a' }}>{stats.activeUsers}</div>
          <div style={{ fontSize: '11px', marginTop: '12px', color: '#8c8c8c', borderTop: '1px solid #f0f0f0', paddingTop: '12px' }}>本周在线</div>
        </div>

        <div style={{ 
          padding: '20px', 
          background: 'white',
          border: '1px solid #e8e8e8',
          borderLeft: '3px solid #6b7280',
          borderRadius: '4px',
          color: '#1a1a1a'
        }}>
          <div style={{ fontSize: '12px', color: '#8c8c8c', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>知识库文档</div>
          <div style={{ fontSize: '32px', fontWeight: '300', marginTop: '4px', color: '#1a1a1a' }}>{stats.knowledgeDocuments}</div>
          <div style={{ fontSize: '11px', marginTop: '12px', color: '#8c8c8c', borderTop: '1px solid #f0f0f0', paddingTop: '12px' }}>最近更新</div>
        </div>
      </div>

      {/* 快捷操作 */}
      <div style={{ marginTop: '48px' }}>
        <h3 style={{ marginBottom: '24px', fontSize: '18px', fontWeight: '400', color: '#1a1a1a' }}>快捷操作</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
          <Button 
            size="large" 
            style={{ 
              height: '72px', 
              fontSize: '15px',
              background: '#2c5aa0',
              borderColor: '#2c5aa0',
              color: 'white',
              fontWeight: '400'
            }}
            onClick={() => {}}
          >
            新建充电分析
          </Button>
          <Button 
            size="large" 
            style={{ 
              height: '72px', 
              fontSize: '15px',
              background: 'white',
              borderColor: '#d9d9d9',
              color: '#1a1a1a',
              fontWeight: '400'
            }}
            onClick={() => {}}
          >
            查询知识库
          </Button>
          <Button 
            size="large" 
            style={{ 
              height: '72px', 
              fontSize: '15px',
              background: 'white',
              borderColor: '#d9d9d9',
              color: '#1a1a1a',
              fontWeight: '400'
            }}
            onClick={() => {}}
          >
            创建训练任务
          </Button>
        </div>
      </div>

      {/* 最近活动 */}
      <div style={{ marginTop: '48px' }}>
        <h3 style={{ marginBottom: '24px', fontSize: '18px', fontWeight: '400', color: '#1a1a1a' }}>最近活动</h3>
        <div style={{ background: 'white', border: '1px solid #e8e8e8', borderRadius: '4px', padding: '0' }}>
          {recentActivities.length === 0 ? (
            <div style={{ padding: '48px', textAlign: 'center', color: '#8c8c8c' }}>
              <div style={{ fontSize: '14px' }}>暂无活动记录</div>
            </div>
          ) : (
            recentActivities.map((activity, index) => (
            <div 
              key={activity.id}
              style={{ 
                  padding: '20px 24px', 
                  borderBottom: index < recentActivities.length - 1 ? '1px solid #f0f0f0' : 'none',
                display: 'flex',
                justifyContent: 'space-between',
                  alignItems: 'center'
              }}
            >
              <div>
                <span style={{ 
                    padding: '4px 10px', 
                    background: '#f0f4f8',
                    color: '#2c5aa0',
                    borderRadius: '2px',
                    fontSize: '11px',
                    marginRight: '12px',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    fontWeight: '400'
                }}>
                  {activity.type}
                </span>
                  <strong style={{ color: '#1a1a1a', fontWeight: '500' }}>{activity.user}</strong>
                  <span style={{ color: '#8c8c8c', marginLeft: '8px' }}>{activity.action}</span>
              </div>
                <span style={{ fontSize: '12px', color: '#8c8c8c' }}>{activity.time}</span>
            </div>
            ))
          )}
        </div>
      </div>

      {/* 功能介绍 */}
      <div style={{ marginTop: '48px', padding: '32px', background: 'white', border: '1px solid #e8e8e8', borderRadius: '4px' }}>
        <h3 style={{ marginBottom: '32px', fontSize: '18px', fontWeight: '400', color: '#1a1a1a' }}>系统功能</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '32px' }}>
          <div>
            <h4 style={{ color: '#2c5aa0', marginBottom: '12px', fontSize: '16px', fontWeight: '500' }}>充电数据分析</h4>
            <p style={{ fontSize: '14px', color: '#8c8c8c', lineHeight: '1.8' }}>
              上传BLF、CSV或Excel格式的充电数据，系统自动进行异常检测、趋势分析和风险评估，生成专业的诊断报告。
            </p>
          </div>
          <div>
            <h4 style={{ color: '#1a1a1a', marginBottom: '12px', fontSize: '16px', fontWeight: '500' }}>RAG知识库</h4>
            <p style={{ fontSize: '14px', color: '#8c8c8c', lineHeight: '1.8' }}>
              管理技术文档和知识库，支持智能检索和语义查询，快速获取相关技术信息和解决方案。
            </p>
          </div>
          <div>
            <h4 style={{ color: '#1a1a1a', marginBottom: '12px', fontSize: '16px', fontWeight: '500' }}>训练管理</h4>
            <p style={{ fontSize: '14px', color: '#8c8c8c', lineHeight: '1.8' }}>
              上传训练数据集，创建和管理AI模型训练任务，实时监控训练进度和性能指标。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

const ChargingPage = () => {
  const [file, setFile] = React.useState<File | null>(null);
  const [analysisId, setAnalysisId] = React.useState<number | null>(null);
  const [status, setStatus] = React.useState<string>('idle');
  const [results, setResults] = React.useState<any[]>([]);
  const [analysisHistory, setAnalysisHistory] = React.useState<any[]>([]);
  const [availableSignals, setAvailableSignals] = React.useState<any[]>([]);
  const [selectedSignals, setSelectedSignals] = React.useState<string[]>([]);
  const [loadingSignals, setLoadingSignals] = React.useState(false);
  const [analysisProgress, setAnalysisProgress] = React.useState(0);
  const [progressMessage, setProgressMessage] = React.useState('');
  const [pollInterval, setPollInterval] = React.useState<ReturnType<typeof setInterval> | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = React.useState(false);
  const [selectedStep, setSelectedStep] = React.useState<string | null>(null);
  const [analysisData, setAnalysisData] = React.useState<any>(null);
  const { user, token } = useAuthStore();
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  // 定义流程节点
  const workflowSteps = [
    { id: 'file_upload', name: '文件上传', icon: FileOutlined, progressRange: [0, 10] },
    { id: 'file_validation', name: '文件验证', icon: CheckOutlined, progressRange: [10, 20] },
    { id: 'message_parsing', name: '报文解析', icon: CodeOutlined, progressRange: [20, 50] },
    { id: 'flow_control', name: '流程控制', icon: ControlOutlined, progressRange: [50, 60] },
    { id: 'rag_retrieval', name: 'RAG检索', icon: SearchOutlined, progressRange: [60, 80] },
    { id: 'detailed_analysis', name: '细化分析', icon: ToolOutlined, progressRange: [80, 90] },
    { id: 'llm_analysis', name: 'LLM分析', icon: RobotOutlined, progressRange: [90, 95] },
    { id: 'report_generation', name: '报告生成', icon: FileTextOutlined, progressRange: [95, 100] },
  ];

  // 根据进度获取当前步骤
  const getCurrentSteps = () => {
    if (!file) return [];
    
    const steps: any[] = [];
    
    // 文件上传步骤（始终显示）
    steps.push({
      ...workflowSteps[0],
      status: 'completed',
      data: { fileName: file?.name, fileSize: file?.size }
    });

    // 如果已上传，显示文件验证
    if (status === 'uploaded' || status === 'analyzing' || status === 'completed') {
      steps.push({
        ...workflowSteps[1],
        status: status === 'uploaded' ? 'completed' : 'completed'
      });
    }

    // 根据进度显示其他步骤
    if (status === 'analyzing' || status === 'completed') {
      const currentProgress = analysisProgress;
      
      for (let i = 2; i < workflowSteps.length; i++) {
        const step = workflowSteps[i];
        const [min, max] = step.progressRange;
        
        if (currentProgress >= min) {
          let stepStatus: 'pending' | 'active' | 'completed' = 'pending';
          if (currentProgress >= max) {
            stepStatus = 'completed';
          } else if (currentProgress >= min) {
            stepStatus = 'active';
          }
          
          steps.push({
            ...step,
            status: stepStatus
          });
        }
      }
    }

    return steps;
  };

  // 获取步骤的详细信息
  const getStepDetails = (stepId: string) => {
    // 文件上传步骤不需要 analysisData
    if (stepId === 'file_upload') {
      return {
        title: '文件上传',
        data: {
          fileName: file?.name,
          fileSize: file?.size ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : '',
          fileType: file?.name.split('.').pop()?.toUpperCase()
        }
      };
    }
    
    // 其他步骤需要 analysisData，如果没有则返回空数据
    if (!analysisData) {
      return {
        title: stepId === 'message_parsing' ? '报文解析' : 
              stepId === 'report_generation' ? '报告生成' : '详细信息',
        data: {}
      };
    }
    
    const details: any = {
      file_upload: {
        title: '文件上传',
        data: {
          fileName: file?.name,
          fileSize: file?.size ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : '',
          fileType: file?.name.split('.').pop()?.toUpperCase()
        }
      },
      file_validation: {
        title: '文件验证',
        data: analysisData.analysis_status || {}
      },
      message_parsing: {
        title: '报文解析',
        data: analysisData.data_stats ? {
          dataStats: analysisData.data_stats,
          signalCount: analysisData.data_stats.signal_count || 0,
          totalRecords: analysisData.data_stats.total_records || 0,
          signalStats: analysisData.data_stats.signal_stats || {},
          timeRange: analysisData.data_stats.time_range || {}
        } : {}
      },
      flow_control: {
        title: '流程控制',
        data: analysisData.flow_analysis || {}
      },
      rag_retrieval: {
        title: 'RAG检索',
        data: {
          retrievedDocuments: analysisData.retrieved_documents || [],
          retrievalContext: analysisData.retrieval_context || ''
        }
      },
      detailed_analysis: {
        title: '细化分析',
        data: {
          refinedSignals: analysisData.refined_signals || [],
          signalValidation: analysisData.signal_validation || {}
        }
      },
      llm_analysis: {
        title: 'LLM分析',
        data: analysisData.llm_analysis || {}
      },
      report_generation: {
        title: '报告生成',
        data: {
          finalReport: analysisData.final_report || {},
          visualizations: analysisData.visualizations || [],
          llmAnalysis: analysisData.llm_analysis || {},
          results: results || []
        }
      }
    };

    return details[stepId] || null;
  };

  // Load analysis history on mount
  React.useEffect(() => {
    if (user && token) {
      loadAnalysisHistory();
    }
  }, [user, token]);

  const loadAnalysisHistory = async () => {
    if (!token) return;
    try {
      const { chargingService } = await import('./services/chargingService');
      const history = await chargingService.getUserAnalyses(token);
      setAnalysisHistory(history.slice(0, 10));
    } catch (error) {
      console.error('加载历史记录失败:', error);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      // 如果当前状态是错误或失败，重置状态以允许重新上传
      if (status === 'error' || status === 'failed' || status === 'cancelled') {
        setStatus('idle');
        setAnalysisId(null);
        setResults([]);
        setSelectedSignals([]);
        setAnalysisProgress(0);
        setProgressMessage('');
      }
    }
  };

  const handleReset = () => {
    setFile(null);
    setAnalysisId(null);
    setStatus('idle');
    setResults([]);
    setSelectedSignals([]);
    setAnalysisProgress(0);
    setProgressMessage('');
    setSelectedStep(null);
    setAnalysisData(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    // 停止轮询
    if (pollInterval) {
      clearInterval(pollInterval);
      setPollInterval(null);
    }
  };

  const loadAvailableSignals = async () => {
    if (!token) return;
    setLoadingSignals(true);
    try {
      const { chargingService } = await import('./services/chargingService');
      const response = await chargingService.getAvailableSignals(token);
      setAvailableSignals(response.signals);
    } catch (error: any) {
      console.error('加载信号列表失败:', error);
      message.warning('加载信号列表失败，将使用默认信号');
    } finally {
      setLoadingSignals(false);
    }
  };

  const handleUpload = async () => {
    if (!file || !user || !token) return;

    setStatus('uploading');
    setResults([]);
    setSelectedSignals([]);
    try {
      const { chargingService } = await import('./services/chargingService');
      
      message.loading('上传中...', 0);
      const analysis = await chargingService.uploadFile(file, token, file.name);
      message.destroy();
      message.success('文件上传成功');
      
      setAnalysisId(analysis.id);
      setStatus('uploaded');
      
      // Load available signals for selection
      await loadAvailableSignals();
      
    } catch (error: any) {
      message.destroy();
      message.error(error.message || '上传失败');
      setStatus('error');
    }
  };

  const handleStartAnalysis = async () => {
    if (!analysisId || !token) return;

    setStatus('analyzing');
    try {
      const { chargingService } = await import('./services/chargingService');
      
      message.loading('开始分析...', 0);
      // Pass selected signals (empty array means all signals)
      const signalsToUse = selectedSignals.length > 0 ? selectedSignals : undefined;
      await chargingService.startAnalysis(analysisId, token, signalsToUse);
      message.destroy();
      message.success(selectedSignals.length > 0 
        ? `分析已开始（已选择 ${selectedSignals.length} 个信号）`
        : '分析已开始（将解析所有信号）'
      );
      
      // Poll for results
      pollAnalysisStatus(analysisId);
      
    } catch (error: any) {
      message.destroy();
      message.error(error.message || '启动分析失败');
      setStatus('error');
    }
  };

  const pollAnalysisStatus = async (id: number) => {
    const authToken = useAuthStore.getState().token;
    if (!authToken) return;
    const { chargingService } = await import('./services/chargingService');
    
    const interval = setInterval(async () => {
      try {
        const analysis = await chargingService.getAnalysis(id, authToken);
        
        // 更新进度
        setAnalysisProgress(analysis.progress || 0);
        
        if (analysis.status === 'completed') {
          clearInterval(interval);
          setPollInterval(null);
          setStatus('completed');
          setAnalysisProgress(100);
          setProgressMessage('分析完成');
          message.success('分析完成');
          
          // Load results and display them
          const analysisResults = await chargingService.getAnalysisResults(id, authToken);
          setResults(analysisResults);
          
          // Load analysis data for workflow display
          if (analysis.resultData) {
            try {
              const data = JSON.parse(analysis.resultData);
              setAnalysisData(data);
            } catch (e) {
              console.error('解析分析数据失败:', e);
            }
          }
          
          // Refresh history
          loadAnalysisHistory();
        } else if (analysis.status === 'failed') {
          clearInterval(interval);
          setPollInterval(null);
          setStatus('failed');
          // 检查是否是被取消的
          if (analysis.errorMessage?.includes('取消')) {
            setStatus('cancelled');
            setProgressMessage('分析已取消');
            message.info('分析已取消');
          } else {
            message.error('分析失败：' + (analysis.errorMessage || '未知错误'));
          }
          // Refresh history
          loadAnalysisHistory();
        } else if (analysis.status === 'processing') {
          // 分析进行中，更新进度消息
          setProgressMessage(`分析进行中 (${analysis.progress?.toFixed(1)}%)`);
          
          // 如果分析数据已存在，尝试解析并更新
          if (analysis.resultData) {
            try {
              const data = JSON.parse(analysis.resultData);
              setAnalysisData(data);
            } catch (e) {
              console.error('解析分析数据失败:', e);
            }
          }
          
          // 定期刷新历史记录以更新进度（10%概率）
          if (Math.random() < 0.1) {
            loadAnalysisHistory();
          }
        }
      } catch (error) {
        console.error('轮询状态失败：', error);
      }
    }, 2000);  // 每2秒轮询一次
    
    setPollInterval(interval);
    
    // Stop after 10 minutes
    setTimeout(() => {
      clearInterval(interval);
      setPollInterval(null);
    }, 600000);
  };

  const handleCancelAnalysis = async () => {
    if (!analysisId || !token) {
      message.error('分析ID或token无效');
      console.error('取消分析失败: analysisId=', analysisId, 'token=', token ? '存在' : '不存在');
      return;
    }
    
    try {
      const { chargingService } = await import('./services/chargingService');
      console.log('发送取消分析请求:', analysisId);
      await chargingService.cancelAnalysis(analysisId, token);
      
      // 停止轮询
      if (pollInterval) {
        clearInterval(pollInterval);
        setPollInterval(null);
      }
      
      setStatus('cancelled');
      setProgressMessage('分析已取消');
      message.success('分析已取消');
    } catch (error: any) {
      console.error('取消分析错误:', error);
      message.error(error.message || '取消分析失败');
    }
  };

  const loadHistoryResults = async (id: number) => {
    if (!token) {
      message.error('请先登录');
      return;
    }
    try {
      const { chargingService } = await import('./services/chargingService');
      const analysis = await chargingService.getAnalysis(id, token);
      const analysisResults = await chargingService.getAnalysisResults(id, token);
      setResults(analysisResults);
      setAnalysisId(id);
      setStatus(analysis.status);
      setAnalysisProgress(analysis.progress || 100);
      
      // Load analysis data for workflow display
      if (analysis.resultData) {
        try {
          const data = JSON.parse(analysis.resultData);
          setAnalysisData(data);
        } catch (e) {
          console.error('解析分析数据失败:', e);
        }
      }
    } catch (error) {
      message.error('加载结果失败');
    }
  };

  const getResultLabel = (type: string) => {
    switch (type) {
      case 'summary': return '概要';
      case 'finding': return '发现';
      case 'recommendation': return '建议';
      case 'technical': return '技术';
      case 'chart': return '图表';
      default: return '结果';
    }
  };

  const getResultColor = (type: string) => {
    switch (type) {
      case 'summary': return '#1890ff';
      case 'finding': return '#faad14';
      case 'recommendation': return '#52c41a';
      case 'technical': return '#722ed1';
      default: return '#666';
    }
  };

  return (
    <AntLayout style={{ background: '#fff' }}>
      {/* 侧边栏 - 历史分析记录 */}
      <Sider 
        width={sidebarCollapsed ? 80 : 280} 
        style={{ 
          background: '#fff',
          borderRight: sidebarCollapsed ? 'none' : '1px solid #e8e8e8',
          height: 'calc(100vh - 64px)',
          position: 'fixed',
          left: 0,
          top: '64px',
          overflowY: 'auto',
          overflowX: 'hidden',
          zIndex: 10,
          transition: 'width 0.2s'
        }}
      >
        {sidebarCollapsed ? (
          <div style={{ 
            padding: '8px 4px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'flex-start',
            height: '100%'
          }}>
            {/* 操作按钮组 - 悬浮白色容器，左右半圆 */}
            <div style={{
              background: 'white',
              border: '1px solid #e0e0e0',
              borderRadius: '50px',
              padding: '10px 24px',
              display: 'flex',
              flexDirection: 'row',
              gap: '16px',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
              width: 'calc(100% - 8px)',
              minWidth: '64px',
              marginTop: '8px'
            }}>
              {/* 展开侧边栏按钮 */}
              <Button
                type="text"
                icon={<MenuUnfoldOutlined style={{ fontSize: '18px', color: '#8c8c8c' }} />}
                onClick={() => setSidebarCollapsed(false)}
                style={{
                  width: 'auto',
                  height: 'auto',
                  minWidth: 'auto',
                  padding: '0',
                  margin: '0',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: 'none',
                  background: 'transparent',
                  boxShadow: 'none',
                  flex: '0 0 auto'
                }}
              />
              
              {/* 新建分析按钮 - 对话框图标带加号 */}
              <Button
                type="text"
                onClick={() => {
                  setFile(null);
                  setAnalysisId(null);
                  setStatus('idle');
                  setResults([]);
                  setSelectedSignals([]);
                  setAnalysisProgress(0);
                  setProgressMessage('');
                  if (fileInputRef.current) {
                    fileInputRef.current.value = '';
                  }
                }}
                style={{
                  width: 'auto',
                  height: 'auto',
                  minWidth: 'auto',
                  padding: '0',
                  margin: '0',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: 'none',
                  background: 'transparent',
                  boxShadow: 'none',
                  position: 'relative',
                  flex: '0 0 auto'
                }}
              >
                <MessageOutlined style={{ fontSize: '18px', color: '#8c8c8c' }} />
                <PlusOutlined style={{ 
                  fontSize: '9px', 
                  position: 'absolute',
                  bottom: '0px',
                  right: '0px',
                  background: '#1a1a1a',
                  color: 'white',
                  borderRadius: '50%',
                  width: '12px',
                  height: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  lineHeight: '12px',
                  padding: 0,
                  border: '1px solid white'
                }} />
              </Button>
            </div>
          </div>
        ) : (
          <div style={{ padding: '16px' }}>
          {/* 侧边栏标题栏 */}
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between',
            marginBottom: '16px',
            paddingBottom: '12px',
            borderBottom: '1px solid #e8e8e8'
          }}>
            <div style={{ fontSize: '16px', fontWeight: '500', color: '#1a1a1a' }}>充电分析</div>
            <Button
              type="text"
              icon={sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              style={{
                padding: '4px 8px',
                color: '#8c8c8c',
                minWidth: 'auto',
                height: 'auto'
              }}
            />
          </div>
          
          {/* 新建分析按钮 */}
          <Button
            block
            size="large"
            onClick={() => {
              setFile(null);
              setAnalysisId(null);
              setStatus('idle');
              setResults([]);
              setSelectedSignals([]);
              setAnalysisProgress(0);
              setProgressMessage('');
              if (fileInputRef.current) {
                fileInputRef.current.value = '';
              }
            }}
            style={{
              height: '44px',
              borderRadius: '8px',
              border: '1px solid #e8e8e8',
              background: 'white',
              color: '#1a1a1a',
              fontWeight: '400',
              marginBottom: '24px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px'
            }}
          >
            <span style={{ fontSize: '16px' }}>+</span>
            新建分析
          </Button>

          {/* 历史记录列表 */}
          {analysisHistory.length === 0 ? (
            <div style={{ padding: '40px 20px', textAlign: 'center', color: '#8c8c8c' }}>
              <div style={{ fontSize: '14px' }}>暂无历史记录</div>
            </div>
          ) : (
            (() => {
              // 按日期分组
              const now = new Date();
              const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
              const yesterday = new Date(today);
              yesterday.setDate(yesterday.getDate() - 1);

              const groups: { [key: string]: any[] } = {
                '今天': [],
                '昨天': [],
                '更早': []
              };

              analysisHistory.forEach((item) => {
                const itemDate = new Date(item.createdAt);
                const itemDateOnly = new Date(itemDate.getFullYear(), itemDate.getMonth(), itemDate.getDate());
                
                if (itemDateOnly.getTime() === today.getTime()) {
                  groups['今天'].push(item);
                } else if (itemDateOnly.getTime() === yesterday.getTime()) {
                  groups['昨天'].push(item);
                } else {
                  groups['更早'].push(item);
                }
              });

              return (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                  {Object.entries(groups).map(([groupName, items]) => {
                    if (items.length === 0) return null;
                    return (
                      <div key={groupName}>
                        <div style={{ 
                          fontSize: '12px', 
                          color: '#8c8c8c', 
                          marginBottom: '12px',
                          paddingLeft: '4px'
                        }}>
                          {groupName}
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                          {items.map((item) => (
                            <div 
                              key={item.id}
                              style={{ 
                                padding: '10px 12px', 
                                background: analysisId === item.id ? '#e6f7ff' : 'transparent',
                                borderRadius: '6px',
                                cursor: item.status === 'completed' ? 'pointer' : 'default',
                                transition: 'all 0.2s',
                                border: 'none'
                              }}
                              onClick={(e) => {
                                if (item.status === 'completed' && (e.target as HTMLElement).tagName !== 'BUTTON') {
                                  loadHistoryResults(item.id);
                                }
                              }}
                              onMouseEnter={(e) => {
                                if (analysisId !== item.id) {
                                  e.currentTarget.style.background = '#f0f0f0';
                                }
                              }}
                              onMouseLeave={(e) => {
                                if (analysisId !== item.id) {
                                  e.currentTarget.style.background = 'transparent';
                                }
                              }}
                            >
                              <div style={{ 
                                fontSize: '14px',
                                color: '#1a1a1a',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                                lineHeight: '1.4',
                                marginBottom: item.status === 'processing' ? '4px' : '0'
                              }}>
                                {item.name}
                              </div>
                              {item.status === 'processing' && item.progress !== undefined && (
                                <div style={{ 
                                  display: 'flex', 
                                  alignItems: 'center', 
                                  justifyContent: 'space-between',
                                  marginTop: '4px',
                                  gap: '8px'
                                }}>
                                  <div style={{ fontSize: '11px', color: '#8c8c8c', flex: 1 }}>
                                    {item.progress.toFixed(1)}%
                                  </div>
                                  <Button
                                    size="small"
                                    danger
                                    onClick={async (e) => {
                                      e.stopPropagation();
                                      if (!token) {
                                        message.error('请先登录');
                                        return;
                                      }
                                      try {
                                        const { chargingService } = await import('./services/chargingService');
                                        await chargingService.cancelAnalysis(item.id, token);
                                        message.success('分析已停止');
                                        await loadAnalysisHistory();
                                        if (analysisId === item.id) {
                                          setStatus('cancelled');
                                          setProgressMessage('分析已取消');
                                          if (pollInterval) {
                                            clearInterval(pollInterval);
                                            setPollInterval(null);
                                          }
                                        }
                                      } catch (error: any) {
                                        console.error('停止分析失败:', error);
                                        message.error(error.message || '停止分析失败');
                                      }
                                    }}
                                    style={{ 
                                      fontSize: '11px', 
                                      padding: '0 6px', 
                                      height: '20px',
                                      lineHeight: '20px'
                                    }}
                                  >
                                    停止
                                  </Button>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            })()
          )}
        </div>
        )}
      </Sider>

      {/* 主内容区域 */}
      <Content style={{ 
        marginLeft: sidebarCollapsed ? '80px' : '280px', 
        padding: '32px',
        paddingRight: '48px',
        transition: 'margin-left 0.2s',
        position: 'relative',
        overflow: 'visible'
      }}>
        {/* 内容容器 */}
        <div style={{ 
          display: 'flex', 
          flexDirection: 'column',
          width: '100%',
          maxWidth: '1400px',
          margin: '0 auto',
          overflow: 'visible'
        }}>
        {/* 流程卡片区域 - 左上角 */}
        {file && (
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '12px',
            marginTop: '0',
            marginBottom: '24px',
            marginLeft: '0',
            marginRight: '-16px',
            paddingTop: '8px',
            paddingBottom: '8px',
            paddingLeft: '0',
            paddingRight: '32px',
            flexWrap: 'nowrap',
            overflowX: 'auto',
            overflowY: 'visible',
            minWidth: '100%',
            boxSizing: 'border-box'
          }}>
            {/* 文件卡片 */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: (status === 'analyzing' || status === 'completed') ? '8px' : '12px 16px',
              background: 'white',
              border: '2px solid #2c5aa0',
              borderRadius: '8px',
              boxShadow: '0 2px 8px rgba(44, 90, 160, 0.1)',
              cursor: selectedStep === 'file_upload' ? 'default' : 'pointer',
              transition: 'all 0.2s',
              opacity: selectedStep === 'file_upload' ? 1 : 0.9,
              flexShrink: 0
            }}
            onClick={() => setSelectedStep('file_upload')}
            onMouseEnter={(e) => {
              if (selectedStep !== 'file_upload') {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(44, 90, 160, 0.15)';
              }
            }}
            onMouseLeave={(e) => {
              if (selectedStep !== 'file_upload') {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 2px 8px rgba(44, 90, 160, 0.1)';
              }
            }}
            >
              {/* 分析开始后隐藏图标和文字，只显示删除按钮 */}
              {(status === 'analyzing' || status === 'completed') ? (
                <Button
                  type="text"
                  size="small"
                  icon={<DeleteOutlined />}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleReset();
                  }}
                  style={{ padding: '4px', minWidth: 'auto', color: '#ff4d4f' }}
                />
              ) : (
                <>
                  <FileOutlined style={{ fontSize: '20px', color: '#2c5aa0' }} />
                  <span style={{ fontSize: '14px', color: '#1a1a1a', fontWeight: '500', whiteSpace: 'nowrap' }}>
                    文件上传
                  </span>
                  <Button
                    type="text"
                    size="small"
                    icon={<DeleteOutlined />}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleReset();
                    }}
                    style={{ padding: '4px', minWidth: 'auto', color: '#ff4d4f', marginLeft: '4px' }}
                  />
                </>
              )}
            </div>

            {/* 上传按钮（如果文件还未上传） */}
            {file && status !== 'uploaded' && status !== 'uploading' && status !== 'analyzing' && status !== 'completed' && (
              <Button
                type="primary"
                icon={<UploadOutlined />}
                onClick={handleUpload}
                loading={status === 'uploading'}
                style={{
                  background: '#2c5aa0',
                  borderColor: '#2c5aa0',
                  height: '44px',
                  padding: '0 20px',
                  borderRadius: '8px',
                  fontWeight: '500',
                  flexShrink: 0
                }}
              >
                {status === 'uploading' ? '上传中...' : '上传文件'}
              </Button>
            )}

            {/* 开始分析按钮（如果文件已上传） */}
            {status === 'uploaded' && (
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={handleStartAnalysis}
                style={{
                  background: '#2c5aa0',
                  borderColor: '#2c5aa0',
                  height: '44px',
                  padding: '0 20px',
                  borderRadius: '8px',
                  fontWeight: '500',
                  flexShrink: 0
                }}
              >
                开始分析
              </Button>
            )}

            {/* 动态流程卡片 */}
            {getCurrentSteps().slice(1).map((step, index) => {
              const StepIcon = step.icon;
              const isActive = step.status === 'active';
              const isCompleted = step.status === 'completed';
              const isSelected = selectedStep === step.id;
              
              return (
                <React.Fragment key={step.id}>
                  {/* 连接线 */}
                  <div style={{
                    width: '24px',
                    height: '2px',
                    background: isCompleted ? '#52c41a' : isActive ? '#2c5aa0' : '#e8e8e8',
                    transition: 'all 0.3s',
                    flexShrink: 0
                  }} />
                  
                  {/* 流程卡片 */}
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '12px 16px',
                    background: isSelected ? '#e6f7ff' : 'white',
                    border: `2px solid ${isCompleted ? '#52c41a' : isActive ? '#2c5aa0' : '#e8e8e8'}`,
                    borderRadius: '8px',
                    boxShadow: isSelected ? '0 4px 12px rgba(44, 90, 160, 0.15)' : '0 2px 8px rgba(0,0,0,0.08)',
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    position: 'relative',
                    flexShrink: 0
                  }}
                  onClick={() => setSelectedStep(step.id)}
                  onMouseEnter={(e) => {
                    if (!isSelected) {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 4px 12px rgba(44, 90, 160, 0.15)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) {
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)';
                    }
                  }}
                  >
                    <StepIcon style={{ 
                      fontSize: '20px', 
                      color: isCompleted ? '#52c41a' : isActive ? '#2c5aa0' : '#8c8c8c'
                    }} />
                    <span style={{ 
                      fontSize: '14px', 
                      color: isCompleted ? '#52c41a' : isActive ? '#2c5aa0' : '#8c8c8c',
                      fontWeight: isActive ? '500' : '400',
                      whiteSpace: 'nowrap'
                    }}>
                      {step.name}
                    </span>
                    {isCompleted && (
                      <CheckCircleOutlined style={{ 
                        fontSize: '16px', 
                        color: '#52c41a',
                        position: 'absolute',
                        top: '-6px',
                        right: '-6px',
                        background: 'white',
                        borderRadius: '50%'
                      }} />
                    )}
                  </div>
                </React.Fragment>
              );
            })}
          </div>
        )}

        {/* Upload Section - 优雅的文件选择区域（仅在无文件时显示） */}
        {!file && (
        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', marginBottom: '32px' }}>
          <div 
            style={{ 
              border: '2px dashed #2c5aa0',
              borderRadius: '12px',
              padding: '32px 40px',
              textAlign: 'center',
              background: '#f0f4f8',
              transition: 'all 0.3s',
              cursor: 'pointer',
              width: '480px',
              maxWidth: '100%'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = '#1e3f73';
              e.currentTarget.style.background = '#e6f0ff';
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(44, 90, 160, 0.15)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = '#2c5aa0';
              e.currentTarget.style.background = '#f0f4f8';
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = 'none';
            }}
            onClick={() => fileInputRef.current?.click()}
          >
        <input
          ref={fileInputRef}
          type="file"
          accept=".blf,.csv,.xlsx"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
            <div style={{ marginBottom: '16px' }}>
              <UploadOutlined style={{ fontSize: '48px', color: '#2c5aa0' }} />
            </div>
            <div style={{ fontSize: '16px', color: '#1a1a1a', marginBottom: '6px', fontWeight: '500' }}>
              点击或拖拽文件到此处上传
            </div>
            <div style={{ fontSize: '13px', color: '#595959', marginBottom: '20px' }}>
              支持 BLF、CSV、XLSX 格式
            </div>
            <Button 
              size="large"
              style={{
                background: '#2c5aa0',
                borderColor: '#2c5aa0',
                color: 'white',
                height: '40px',
                padding: '0 32px',
                borderRadius: '8px',
                fontWeight: '500',
                fontSize: '15px'
              }}
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              <UploadOutlined /> 选择文件
            </Button>
          </div>
          </div>
        )}

        {/* 详细信息展示区域 */}
        {selectedStep && (
          <Card 
            title={getStepDetails(selectedStep)?.title || '详细信息'}
            style={{ marginTop: '24px', width: '100%' }}
            extra={
              <Button 
                type="text" 
                icon={<CloseOutlined />}
                onClick={() => setSelectedStep(null)}
              />
            }
          >
            {(() => {
              const details = getStepDetails(selectedStep);
              if (!details) {
                return (
                  <div style={{ textAlign: 'center', padding: '40px', color: '#8c8c8c' }}>
                    <Typography.Text>暂无详细信息</Typography.Text>
                    {!analysisData && (
                      <div style={{ marginTop: '12px', fontSize: '12px' }}>
                        数据可能还在加载中，请稍候...
                      </div>
                    )}
                  </div>
                );
              }

              switch (selectedStep) {
                case 'file_upload':
                  return (
                    <div>
                      <div style={{ marginBottom: '16px' }}>
                        <Typography.Text strong>文件名：</Typography.Text>
                        <Typography.Text>{details.data.fileName}</Typography.Text>
      </div>
                      <div style={{ marginBottom: '16px' }}>
                        <Typography.Text strong>文件大小：</Typography.Text>
                        <Typography.Text>{details.data.fileSize}</Typography.Text>
                      </div>
                      <div>
                        <Typography.Text strong>文件类型：</Typography.Text>
                        <Typography.Text>{details.data.fileType}</Typography.Text>
                      </div>
                    </div>
                  );
                
                case 'message_parsing':
                  // 检查数据是否存在
                  if (!details.data || !details.data.dataStats) {
                    return (
                      <div style={{ textAlign: 'center', padding: '40px', color: '#8c8c8c' }}>
                        <Typography.Text>数据正在加载中，请稍候...</Typography.Text>
                      </div>
                    );
                  }
                  
                  const dataStats = details.data.dataStats;
                  const signalStats = details.data.signalStats || dataStats.signal_stats || {};
                  const timeRange = details.data.timeRange || dataStats.time_range || {};
                  
                  return (
                    <div>
                      <div style={{ marginBottom: '20px' }}>
                        <Typography.Title level={5} style={{ marginBottom: '12px' }}>数据概览</Typography.Title>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '16px' }}>
                          <div style={{ padding: '16px', background: '#f5f5f5', borderRadius: '8px' }}>
                            <Typography.Text type="secondary" style={{ display: 'block', fontSize: '12px', marginBottom: '8px' }}>总记录数</Typography.Text>
                            <Typography.Text strong style={{ fontSize: '24px', color: '#1a1a1a' }}>
                              {(details.data.totalRecords || dataStats.total_records || 0).toLocaleString()}
                            </Typography.Text>
                          </div>
                          <div style={{ padding: '16px', background: '#f5f5f5', borderRadius: '8px' }}>
                            <Typography.Text type="secondary" style={{ display: 'block', fontSize: '12px', marginBottom: '8px' }}>解析信号数</Typography.Text>
                            <Typography.Text strong style={{ fontSize: '24px', color: '#1a1a1a' }}>
                              {details.data.signalCount || dataStats.signal_count || 0}
                            </Typography.Text>
                          </div>
                          {timeRange.start && (
                            <div style={{ padding: '16px', background: '#f5f5f5', borderRadius: '8px' }}>
                              <Typography.Text type="secondary" style={{ display: 'block', fontSize: '12px', marginBottom: '8px' }}>时间范围</Typography.Text>
                              <Typography.Text style={{ fontSize: '13px', color: '#1a1a1a' }}>
                                {new Date(timeRange.start).toLocaleString('zh-CN')}
                              </Typography.Text>
                              <Typography.Text style={{ display: 'block', fontSize: '13px', color: '#1a1a1a', marginTop: '4px' }}>
                                至 {new Date(timeRange.end).toLocaleString('zh-CN')}
                              </Typography.Text>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      {signalStats && Object.keys(signalStats).length > 0 && (
                        <div>
                          <Typography.Title level={5} style={{ marginBottom: '12px' }}>信号详细统计</Typography.Title>
                          <div style={{ maxHeight: '500px', overflowY: 'auto', border: '1px solid #e8e8e8', borderRadius: '8px', padding: '16px' }}>
                            {Object.entries(signalStats).map(([signal, stats]: [string, any]) => (
                              <div key={signal} style={{ 
                                padding: '12px', 
                                marginBottom: '12px', 
                                background: '#fafafa', 
                                borderRadius: '6px',
                                border: '1px solid #f0f0f0'
                              }}>
                                <Typography.Text strong style={{ fontSize: '14px', color: '#1a1a1a', display: 'block', marginBottom: '8px' }}>
                                  {signal}
                                </Typography.Text>
                                {stats.type === 'numeric' ? (
                                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '8px', fontSize: '13px' }}>
                                    <div>
                                      <Typography.Text type="secondary">均值：</Typography.Text>
                                      <Typography.Text>{stats.mean?.toFixed(4) || 'N/A'}</Typography.Text>
                                    </div>
                                    <div>
                                      <Typography.Text type="secondary">标准差：</Typography.Text>
                                      <Typography.Text>{stats.std?.toFixed(4) || 'N/A'}</Typography.Text>
                                    </div>
                                    <div>
                                      <Typography.Text type="secondary">最小值：</Typography.Text>
                                      <Typography.Text>{stats.min?.toFixed(4) || 'N/A'}</Typography.Text>
                                    </div>
                                    <div>
                                      <Typography.Text type="secondary">最大值：</Typography.Text>
                                      <Typography.Text>{stats.max?.toFixed(4) || 'N/A'}</Typography.Text>
                                    </div>
                                  </div>
                                ) : stats.type === 'categorical' ? (
                                  <div>
                                    <Typography.Text type="secondary" style={{ fontSize: '12px', display: 'block', marginBottom: '4px' }}>
                                      唯一值数量：{stats.unique_values?.length || Object.keys(stats.distribution || {}).length}
                                    </Typography.Text>
                                    {stats.distribution && (
                                      <div style={{ marginTop: '8px' }}>
                                        <Typography.Text type="secondary" style={{ fontSize: '12px', display: 'block', marginBottom: '4px' }}>分布：</Typography.Text>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                                          {Object.entries(stats.distribution).slice(0, 10).map(([value, count]: [string, any]) => (
                                            <span key={value} style={{
                                              padding: '4px 8px',
                                              background: '#e6f7ff',
                                              borderRadius: '4px',
                                              fontSize: '12px'
                                            }}>
                                              {value}: {count}
                                            </span>
                                          ))}
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                ) : (
                                  <Typography.Text type="secondary" style={{ fontSize: '12px' }}>
                                    唯一值数量：{stats.unique_count || 'N/A'}
                                  </Typography.Text>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                
                case 'flow_control':
                  return (
                    <div>
                      <div style={{ marginBottom: '16px' }}>
                        <Typography.Text strong>问题方向：</Typography.Text>
                        <Typography.Text>{details.data.problem_direction || '未确定'}</Typography.Text>
                      </div>
                      <div style={{ marginBottom: '16px' }}>
                        <Typography.Text strong>置信度：</Typography.Text>
                        <Typography.Text>{(details.data.confidence || 0) * 100}%</Typography.Text>
                      </div>
                      {details.data.reasoning && (
                        <div>
                          <Typography.Text strong style={{ display: 'block', marginBottom: '8px' }}>分析推理：</Typography.Text>
                          <Typography.Text>{details.data.reasoning}</Typography.Text>
                        </div>
                      )}
                    </div>
                  );
                
                case 'rag_retrieval':
                  return (
                    <div>
                      <div style={{ marginBottom: '16px' }}>
                        <Typography.Text strong>检索到文档数：</Typography.Text>
                        <Typography.Text>{details.data.retrievedDocuments?.length || 0}</Typography.Text>
                      </div>
                      {details.data.retrievedDocuments && details.data.retrievedDocuments.length > 0 && (
                        <div>
                          <Typography.Text strong style={{ display: 'block', marginBottom: '12px' }}>相关文档：</Typography.Text>
                          {details.data.retrievedDocuments.map((doc: any, index: number) => (
                            <div key={index} style={{ 
                              padding: '12px', 
                              marginBottom: '12px', 
                              background: '#f5f5f5', 
                              borderRadius: '4px' 
                            }}>
                              <div style={{ marginBottom: '8px' }}>
                                <Typography.Text strong>来源：</Typography.Text>
                                <Typography.Text>{doc.metadata?.source || '未知'}</Typography.Text>
                                <Typography.Text type="secondary" style={{ marginLeft: '12px' }}>
                                  相似度：{(doc.score * 100).toFixed(1)}%
                                </Typography.Text>
                              </div>
                              <Typography.Text>{doc.content}</Typography.Text>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                
                case 'detailed_analysis':
                  return (
                    <div>
                      <div style={{ marginBottom: '16px' }}>
                        <Typography.Text strong>细化信号数：</Typography.Text>
                        <Typography.Text>{details.data.refinedSignals?.length || 0}</Typography.Text>
                      </div>
                      {details.data.refinedSignals && details.data.refinedSignals.length > 0 && (
                        <div style={{ marginBottom: '16px' }}>
                          <Typography.Text strong style={{ display: 'block', marginBottom: '8px' }}>细化信号列表：</Typography.Text>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                            {details.data.refinedSignals.map((signal: string) => (
                              <span key={signal} style={{
                                padding: '4px 12px',
                                background: '#e6f7ff',
                                borderRadius: '4px',
                                fontSize: '13px'
                              }}>
                                {signal}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      {details.data.signalValidation && (
                        <div>
                          <Typography.Text strong style={{ display: 'block', marginBottom: '8px' }}>信号验证结果：</Typography.Text>
                          <div style={{ padding: '12px', background: '#f5f5f5', borderRadius: '4px' }}>
                            <div style={{ marginBottom: '8px' }}>
                              <Typography.Text strong>验证状态：</Typography.Text>
                              <Typography.Text style={{ color: details.data.signalValidation.validated ? '#52c41a' : '#ff4d4f' }}>
                                {details.data.signalValidation.validated ? '通过' : '未通过'}
                              </Typography.Text>
                            </div>
                            <div style={{ marginBottom: '8px' }}>
                              <Typography.Text strong>平均质量分数：</Typography.Text>
                              <Typography.Text>{(details.data.signalValidation.average_quality * 100).toFixed(1)}%</Typography.Text>
                            </div>
                            <div>
                              <Typography.Text strong>有效信号数：</Typography.Text>
                              <Typography.Text>{details.data.signalValidation.signal_count}</Typography.Text>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                
                case 'llm_analysis':
                  return (
                    <div>
                      {details.data.summary && (
                        <div style={{ marginBottom: '16px' }}>
                          <Typography.Text strong style={{ display: 'block', marginBottom: '8px' }}>诊断总结：</Typography.Text>
                          <Typography.Text>{details.data.summary}</Typography.Text>
                        </div>
                      )}
                      {details.data.findings && details.data.findings.length > 0 && (
                        <div style={{ marginBottom: '16px' }}>
                          <Typography.Text strong style={{ display: 'block', marginBottom: '8px' }}>关键发现：</Typography.Text>
                          <ul style={{ paddingLeft: '20px' }}>
                            {details.data.findings.map((finding: string, index: number) => (
                              <li key={index} style={{ marginBottom: '4px' }}>
                                <Typography.Text>{finding}</Typography.Text>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {details.data.risk_assessment && (
                        <div style={{ marginBottom: '16px' }}>
                          <Typography.Text strong>风险评估：</Typography.Text>
                          <Typography.Text style={{ 
                            color: details.data.risk_assessment === '高' ? '#ff4d4f' : 
                                   details.data.risk_assessment === '中等' ? '#faad14' : '#52c41a'
                          }}>
                            {details.data.risk_assessment}
                          </Typography.Text>
                        </div>
                      )}
                      {details.data.recommendations && details.data.recommendations.length > 0 && (
                        <div>
                          <Typography.Text strong style={{ display: 'block', marginBottom: '8px' }}>建议措施：</Typography.Text>
                          <ul style={{ paddingLeft: '20px' }}>
                            {details.data.recommendations.map((rec: string, index: number) => (
                              <li key={index} style={{ marginBottom: '4px' }}>
                                <Typography.Text>{rec}</Typography.Text>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  );
                
                case 'report_generation':
                  if (!details.data) {
                    return (
                      <div style={{ textAlign: 'center', padding: '40px', color: '#8c8c8c' }}>
                        <Typography.Text>报告正在生成中，请稍候...</Typography.Text>
                      </div>
                    );
                  }
                  
                  const llmData = details.data.llmAnalysis || {};
                  const reportResults = details.data.results || [];
                  
                  // 如果没有任何数据，显示提示
                  if (!llmData.summary && !llmData.findings && !llmData.risk_assessment && !llmData.recommendations && reportResults.length === 0) {
                    return (
                      <div style={{ textAlign: 'center', padding: '40px', color: '#8c8c8c' }}>
                        <Typography.Text>报告数据正在加载中，请稍候...</Typography.Text>
                      </div>
                    );
                  }
                  
                  return (
                    <div>
                      {/* LLM 分析总结 */}
                      {llmData.summary && (
                        <div style={{ marginBottom: '24px', padding: '20px', background: '#f0f4f8', borderRadius: '8px', borderLeft: '4px solid #2c5aa0' }}>
                          <Typography.Title level={5} style={{ marginBottom: '12px', color: '#2c5aa0' }}>分析总结</Typography.Title>
                          <Typography.Text style={{ fontSize: '15px', lineHeight: '1.8', color: '#1a1a1a' }}>
                            {llmData.summary}
                          </Typography.Text>
                        </div>
                      )}
                      
                      {/* 关键发现 */}
                      {llmData.findings && llmData.findings.length > 0 && (
                        <div style={{ marginBottom: '24px' }}>
                          <Typography.Title level={5} style={{ marginBottom: '12px' }}>关键发现</Typography.Title>
                          <div style={{ display: 'grid', gap: '12px' }}>
                            {llmData.findings.map((finding: string, index: number) => (
                              <div key={index} style={{
                                padding: '12px 16px',
                                background: 'white',
                                border: '1px solid #e8e8e8',
                                borderRadius: '6px',
                                borderLeft: '3px solid #52c41a'
                              }}>
                                <Typography.Text style={{ fontSize: '14px', lineHeight: '1.6' }}>
                                  {finding}
                                </Typography.Text>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {/* 风险评估 */}
                      {llmData.risk_assessment && (
                        <div style={{ marginBottom: '24px', padding: '16px', background: '#fff7e6', borderRadius: '8px', border: '1px solid #ffe58f' }}>
                          <Typography.Title level={5} style={{ marginBottom: '8px' }}>风险评估</Typography.Title>
                          <Typography.Text style={{ 
                            fontSize: '16px', 
                            fontWeight: '500',
                            color: llmData.risk_assessment === '高' ? '#ff4d4f' : 
                                   llmData.risk_assessment === '中等' ? '#faad14' : '#52c41a'
                          }}>
                            {llmData.risk_assessment}
                          </Typography.Text>
                        </div>
                      )}
                      
                      {/* 建议措施 */}
                      {llmData.recommendations && llmData.recommendations.length > 0 && (
                        <div style={{ marginBottom: '24px' }}>
                          <Typography.Title level={5} style={{ marginBottom: '12px' }}>建议措施</Typography.Title>
                          <div style={{ display: 'grid', gap: '10px' }}>
                            {llmData.recommendations.map((rec: string, index: number) => (
                              <div key={index} style={{
                                padding: '12px 16px',
                                background: '#f5f5f5',
                                borderRadius: '6px',
                                display: 'flex',
                                alignItems: 'flex-start',
                                gap: '12px'
                              }}>
                                <span style={{
                                  display: 'inline-flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  width: '20px',
                                  height: '20px',
                                  borderRadius: '50%',
                                  background: '#2c5aa0',
                                  color: 'white',
                                  fontSize: '12px',
                                  fontWeight: 'bold',
                                  flexShrink: 0
                                }}>
                                  {index + 1}
                                </span>
                                <Typography.Text style={{ fontSize: '14px', lineHeight: '1.6', flex: 1 }}>
                                  {rec}
                                </Typography.Text>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {/* 分析结果详情 */}
                      {reportResults.length > 0 && (
                        <div style={{ marginBottom: '24px' }}>
                          <Typography.Title level={5} style={{ marginBottom: '12px' }}>分析结果详情</Typography.Title>
                          <div style={{ display: 'grid', gap: '16px' }}>
                            {reportResults.map((result: any, index: number) => (
                              <div 
                                key={result.id || index} 
                                style={{ 
                                  padding: '20px', 
                                  background: 'white', 
                                  border: '1px solid #e8e8e8',
                                  borderLeft: `4px solid ${getResultColor(result.resultType)}`,
                                  borderRadius: '8px',
                                  boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
                                }}
                              >
                                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '12px' }}>
                                  <span style={{ 
                                    padding: '4px 12px',
                                    background: getResultColor(result.resultType) + '20',
                                    color: getResultColor(result.resultType),
                                    borderRadius: '4px',
                                    fontSize: '12px',
                                    fontWeight: 'bold',
                                    marginRight: '12px'
                                  }}>
                                    {getResultLabel(result.resultType)}
                                  </span>
                                  <h3 style={{ margin: 0, color: getResultColor(result.resultType), fontSize: '16px' }}>
                                    {result.title}
                                  </h3>
                                </div>
                                <div style={{ 
                                  color: '#333', 
                                  lineHeight: '1.8',
                                  whiteSpace: 'pre-wrap',
                                  fontSize: '14px'
                                }}>
                                  {result.content}
                                </div>
                                {result.confidenceScore && (
                                  <div style={{ marginTop: '12px', fontSize: '12px', color: '#999' }}>
                                    置信度: {(result.confidenceScore * 100).toFixed(1)}%
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {/* 可视化图表 */}
                      {details.data.visualizations && details.data.visualizations.length > 0 && (
                        <div>
                          <Typography.Title level={5} style={{ marginBottom: '12px' }}>可视化图表</Typography.Title>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
                            {details.data.visualizations.map((viz: any, index: number) => (
                              <div key={index} style={{
                                padding: '12px',
                                background: '#f5f5f5',
                                borderRadius: '4px',
                                minWidth: '200px'
                              }}>
                                <Typography.Text strong>{viz.title}</Typography.Text>
                                <div style={{ marginTop: '8px', fontSize: '12px', color: '#8c8c8c' }}>
                                  类型：{viz.type}
                                </div>
                                {viz.signals && (
                                  <div style={{ marginTop: '4px', fontSize: '12px', color: '#8c8c8c' }}>
                                    信号：{viz.signals.join(', ')}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                
                default:
                  return (
                    <div>
                      <pre style={{ 
                        background: '#f5f5f5', 
                        padding: '16px', 
                        borderRadius: '4px',
                        overflow: 'auto',
                        maxHeight: '400px'
                      }}>
                        {JSON.stringify(details.data, null, 2)}
                      </pre>
                    </div>
                  );
              }
            })()}
          </Card>
        )}

        {/* 信号选择区域（仅在已上传且未开始分析时显示） */}
      {status === 'uploaded' && (
        <Card 
          title="选择要解析的信号" 
            style={{ marginTop: '24px', width: '100%' }}
          extra={
            <Space>
              <Button 
                size="small" 
                onClick={() => {
                  // Select common charging signals
                  const commonSignals = availableSignals
                    .filter(s => 
                      s.name.includes('Battery') || 
                      s.name.includes('Charge') || 
                      s.name.includes('Voltage') ||
                      s.name.includes('Current') ||
                      s.name.includes('Temp') ||
                      s.name.includes('SOC')
                    )
                    .map(s => s.name)
                    .slice(0, 10);
                  setSelectedSignals(commonSignals);
                }}
              >
                快速选择（常用信号）
              </Button>
              <Button 
                size="small" 
                onClick={() => setSelectedSignals([])}
              >
                清空
              </Button>
            </Space>
          }
        >
          {loadingSignals ? (
            <div style={{ textAlign: 'center', padding: '40px' }}>
              <Spin size="large" />
              <p style={{ marginTop: '16px' }}>加载信号列表...</p>
            </div>
          ) : (
            <div>
              <Typography.Text type="secondary" style={{ display: 'block', marginBottom: '16px' }}>
                选择需要解析的信号（留空则解析所有信号）。选择少量关键信号可以显著提升解析速度。
              </Typography.Text>
              <Select
                mode="multiple"
                placeholder="选择信号（留空则解析所有信号）"
                value={selectedSignals}
                onChange={setSelectedSignals}
                style={{ width: '100%' }}
                showSearch
                filterOption={(input, option) =>
                  (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                }
                options={availableSignals.map(signal => ({
                  label: `${signal.name} (${signal.messageName})`,
                  value: signal.name,
                  title: `${signal.name} - 消息: ${signal.messageName}, ID: ${signal.messageId}`
                }))}
                maxTagCount={10}
                listHeight={300}
              />
              <div style={{ marginTop: '16px', padding: '12px', background: '#f5f5f5', borderRadius: '4px' }}>
                <Typography.Text strong>
                  已选择 {selectedSignals.length} 个信号
                  {selectedSignals.length === 0 && '（将解析所有信号）'}
                </Typography.Text>
              </div>
              <div style={{ marginTop: '16px', textAlign: 'right' }}>
                <Space>
                  <Button onClick={() => setStatus('idle')}>
                    取消
                  </Button>
                  <Button 
                    type="primary" 
                    size="large"
                    onClick={handleStartAnalysis}
                  >
                    开始分析
                  </Button>
                </Space>
              </div>
            </div>
          )}
        </Card>
      )}

      {/* 分析进度提示（简化版，主要信息在流程卡片中） */}
      {status === 'analyzing' && !selectedStep && (
        <Card style={{ marginTop: '24px', width: '100%' }}>
          <div style={{ textAlign: 'center', padding: '20px' }}>
            <Progress 
              percent={analysisProgress} 
              status="active"
              strokeColor={{
                '0%': '#108ee9',
                '100%': '#87d068',
              }}
            />
            <p style={{ marginTop: '12px', color: '#666' }}>
              {progressMessage || '分析进行中...'}
            </p>
            <div style={{ marginTop: '16px' }}>
              <Button 
                danger 
                onClick={handleCancelAnalysis}
                disabled={!analysisId}
              >
                停止分析
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* Status Section - shown for other statuses */}
      {status !== 'idle' && status !== 'analyzing' && (
          <div style={{ marginTop: '24px', padding: '16px', background: '#f5f5f5', borderRadius: '8px', width: '100%' }}>
          <p><strong>状态：</strong>
            <span style={{ 
              color: status === 'completed' ? '#52c41a' : 
                     status === 'failed' || status === 'error' ? '#ff4d4f' : 
                     status === 'cancelled' ? '#faad14' : '#666'
            }}>
              {status === 'uploading' ? '上传中...' :
               status === 'uploaded' ? '已上传' :
               status === 'completed' ? '已完成' :
               status === 'failed' ? '失败' :
               status === 'error' ? '错误' : 
               status === 'cancelled' ? '已取消' : status}
            </span>
          </p>
          {analysisId && <p><strong>分析ID：</strong>{analysisId}</p>}
        </div>
      )}

      {/* Results Section - 已移至报告生成卡片中，此处不再显示 */}
                  </div>
      </Content>
    </AntLayout>
  );
};

const RAGPage = () => {
  const [query, setQuery] = React.useState('');
  const [answer, setAnswer] = React.useState('');
  const [isQuerying, setIsQuerying] = React.useState(false);
  const [documents, setDocuments] = React.useState<any[]>([]);
  const [uploadFile, setUploadFile] = React.useState<File | null>(null);
  const [queryHistory, setQueryHistory] = React.useState<any[]>([]);
  const { user, token } = useAuthStore();
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const [collections, setCollections] = React.useState<any[]>([]);
  const [collectionId, setCollectionId] = React.useState<number | null>(null);
  const [isLoadingCollections, setIsLoadingCollections] = React.useState(false);

  React.useEffect(() => {
    if (token) {
      loadCollections();
    }
  }, [token]);

  React.useEffect(() => {
    if (collectionId && token) {
      loadDocuments(collectionId);
      loadQueryHistory(collectionId);
    }
  }, [collectionId, token]);

  const loadCollections = async () => {
    if (!token) return;
    try {
      setIsLoadingCollections(true);
      const { ragService } = await import('./services/ragService');
      let data = await ragService.getCollections(token);
      if (data.length === 0) {
        const created = await ragService.createCollection('默认知识库', '系统自动创建', token);
        data = [created];
      }
      setCollections(data);
      setCollectionId((prev) => prev ?? data[0]?.id ?? null);
    } catch (error) {
      console.error('加载知识库失败:', error);
    } finally {
      setIsLoadingCollections(false);
    }
  };

  const loadDocuments = async (targetId: number) => {
    if (!token) return;
    try {
      const { ragService } = await import('./services/ragService');
      const docs = await ragService.getDocuments(targetId, token);
      setDocuments(docs);
    } catch (error) {
      console.error('加载文档失败:', error);
    }
  };

  const loadQueryHistory = async (targetId: number) => {
    if (!token) return;
    try {
      const { ragService } = await import('./services/ragService');
      const history = await ragService.getQueryHistory(targetId, token, 10);
      setQueryHistory(history);
    } catch (error) {
      console.error('加载查询历史失败:', error);
    }
  };

  const handleCreateCollection = async () => {
    if (!token) {
      message.warning('请先登录');
      return;
    }
    try {
      message.loading('创建知识库...', 0);
      const { ragService } = await import('./services/ragService');
      const name = `知识库-${collections.length + 1}`;
      const collection = await ragService.createCollection(name, '自动创建', token);
      message.destroy();
      message.success('知识库创建成功');
      setCollections((prev: any[]) => [collection, ...prev]);
      setCollectionId(collection.id);
    } catch (error: any) {
      message.destroy();
      message.error(error.message || '创建失败');
    }
  };

  const handleQuery = async () => {
    if (!query.trim()) {
      message.warning('请输入查询内容');
      return;
    }
    if (!token || !collectionId) {
      message.warning('请选择知识库');
      return;
    }

    setIsQuerying(true);
    try {
      const { ragService } = await import('./services/ragService');
      
      // Call real RAG query
      const response = await ragService.query(collectionId, query, token);
      
      setAnswer(response.response);
      message.success('查询完成');
      
      // Reload query history
      loadQueryHistory(collectionId);
      
    } catch (error: any) {
      message.error(error.message || '查询失败');
    } finally {
      setIsQuerying(false);
    }
  };

  const handleUploadDocument = async () => {
    if (!uploadFile) {
      message.warning('请选择要上传的文件');
      return;
    }
    if (!token || !collectionId) {
      message.warning('请选择知识库');
      return;
    }

    try {
      message.loading('上传中...', 0);
      const { ragService } = await import('./services/ragService');
      
      await ragService.uploadDocument(collectionId, uploadFile, token);
      
      message.destroy();
      message.success('文档上传成功');
      setUploadFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      
      // Reload documents
      loadDocuments(collectionId);
    } catch (error: any) {
      message.destroy();
      message.error(error.message || '上传失败');
    }
  };

  return (
    <div>
      <h1>RAG知识库管理</h1>
      <p style={{ marginTop: '8px', color: '#666' }}>
        智能检索技术文档，快速获取相关知识和解决方案
      </p>
      <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <span style={{ fontWeight: 500 }}>当前知识库：</span>
        <select
          value={collectionId ?? ''}
          onChange={(e) => setCollectionId(e.target.value ? Number(e.target.value) : null)}
          style={{ padding: '6px 12px', borderRadius: '6px', border: '1px solid #d9d9d9' }}
          disabled={isLoadingCollections || collections.length === 0}
        >
          {collections.length === 0 && (
            <option value="" disabled>
              暂无知识库
            </option>
          )}
          {collections.map((collection: any) => (
            <option key={collection.id} value={collection.id}>
              {collection.name}
            </option>
          ))}
        </select>
        <Button size="small" onClick={handleCreateCollection} disabled={isLoadingCollections}>
          新建知识库
        </Button>
      </div>

      {/* 查询区域 */}
      <div style={{ marginTop: '24px', padding: '24px', background: '#f6f8fa', borderRadius: '12px' }}>
        <h3 style={{ marginBottom: '16px' }}>知识库查询</h3>
        <div style={{ display: 'flex', gap: '12px' }}>
          <input
            type="text"
            placeholder="请输入您的问题，例如：充电过程中电压异常如何处理？"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleQuery()}
            style={{ 
              flex: 1, 
              padding: '12px 16px', 
              border: '1px solid #d9d9d9', 
              borderRadius: '8px',
              fontSize: '14px'
            }}
          />
          <Button 
            type="primary" 
            size="large" 
            onClick={handleQuery}
            loading={isQuerying}
            style={{ minWidth: '100px' }}
          >
            查询
          </Button>
        </div>

        {/* 查询结果 */}
        {answer && (
          <div style={{ 
            marginTop: '20px', 
            padding: '20px', 
            background: 'white', 
            borderRadius: '8px',
            border: '1px solid #e8e8e8'
          }}>
            <h4 style={{ marginBottom: '12px', color: '#1890ff' }}>查询结果</h4>
            <div style={{ 
              whiteSpace: 'pre-wrap', 
              lineHeight: '1.8',
              color: '#333'
            }}>
              {answer}
            </div>
          </div>
        )}
      </div>

      {/* 查询历史 */}
      <div style={{ marginTop: '24px' }}>
        <h3 style={{ marginBottom: '16px' }}>查询历史</h3>
        <div style={{ display: 'grid', gap: '8px' }}>
          {queryHistory.length === 0 ? (
            <div style={{ padding: '20px', textAlign: 'center', color: '#999' }}>暂无查询历史</div>
          ) : (
            queryHistory.map(item => (
              <div 
                key={item.id}
                style={{ 
                  padding: '12px 16px', 
                  background: '#fafafa',
                  borderRadius: '8px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  cursor: 'pointer'
                }}
                onClick={() => setQuery(item.queryText)}
              >
                <span style={{ color: '#333' }}>{item.queryText}</span>
                <span style={{ fontSize: '12px', color: '#999' }}>
                  {new Date(item.createdAt).toLocaleString('zh-CN')}
                </span>
              </div>
            ))
          )}
        </div>
      </div>

      {/* 文档上传 */}
      <div style={{ marginTop: '32px' }}>
        <h3 style={{ marginBottom: '16px' }}>上传知识库文档</h3>
        <div style={{ padding: '24px', border: '2px dashed #d9d9d9', borderRadius: '12px', textAlign: 'center' }}>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.doc,.docx,.txt"
            onChange={(e) => e.target.files && setUploadFile(e.target.files[0])}
            style={{ display: 'none' }}
          />
          <Button 
            size="large" 
            onClick={() => fileInputRef.current?.click()}
          >
            选择文档
          </Button>
          {uploadFile && (
            <div style={{ marginTop: '16px' }}>
              <p style={{ color: '#333' }}>已选择：{uploadFile.name}</p>
              <Button 
                type="primary" 
                size="large" 
                onClick={handleUploadDocument}
                style={{ marginTop: '12px' }}
              >
                上传文档
              </Button>
            </div>
          )}
          <p style={{ marginTop: '16px', fontSize: '12px', color: '#999' }}>
            支持PDF、Word、TXT格式，文件大小不超过10MB
          </p>
        </div>
      </div>

      {/* 文档列表 */}
      <div style={{ marginTop: '32px' }}>
        <h3 style={{ marginBottom: '16px' }}>已索引文档 ({documents.length})</h3>
        <div style={{ background: '#fafafa', borderRadius: '8px', padding: '16px' }}>
          <div style={{ display: 'grid', gap: '8px' }}>
            {documents.length === 0 ? (
              <div style={{ padding: '20px', textAlign: 'center', color: '#999' }}>暂无文档</div>
            ) : (
              documents.map(doc => (
                <div 
                  key={doc.id}
                  style={{ 
                    padding: '16px', 
                    background: 'white',
                    borderRadius: '8px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    boxShadow: '0 1px 4px rgba(0,0,0,0.05)'
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 'bold', color: '#333' }}>{doc.filename}</div>
                    <div style={{ fontSize: '12px', color: '#999', marginTop: '4px' }}>
                      上传时间: {new Date(doc.createdAt).toLocaleDateString('zh-CN')} | 
                      大小: {(doc.fileSize / 1024 / 1024).toFixed(2)} MB
                    </div>
                  </div>
                  <span style={{ 
                    padding: '4px 12px', 
                    background: doc.uploadStatus === 'completed' ? '#f6ffed' : '#e6f7ff',
                    color: doc.uploadStatus === 'completed' ? '#52c41a' : '#1890ff',
                    borderRadius: '4px',
                    fontSize: '12px'
                  }}>
                    {doc.uploadStatus === 'completed' ? '已索引' : 
                     doc.uploadStatus === 'processing' ? '处理中' : doc.uploadStatus}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const TrainingPage = () => {
  const [datasetFile, setDatasetFile] = React.useState<File | null>(null);
  const [datasetName, setDatasetName] = React.useState('');
  const [uploadStatus, setUploadStatus] = React.useState<string>('idle');
  const [datasetId, setDatasetId] = React.useState<number | null>(null);
  const [taskName, setTaskName] = React.useState('');
  const [taskStatus, setTaskStatus] = React.useState<string>('idle');
  const { user, token } = useAuthStore();
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleDatasetUpload = async () => {
    if (!datasetFile || !datasetName || !user || !token) return;

    setUploadStatus('uploading');
    try {
      const { trainingService } = await import('./services/trainingService');
      
      message.loading('上传数据集...', 0);
      const dataset = await trainingService.uploadDataset(
        datasetName,
        datasetFile,
        token,
        {
          description: '训练数据集',
          datasetType: 'standard'
        }
      );
      message.destroy();
      message.success('数据集上传成功');
      
      setDatasetId(dataset.id);
      setUploadStatus('uploaded');
      
    } catch (error: any) {
      message.destroy();
      message.error(error.message || '上传失败');
      setUploadStatus('error');
    }
  };

  const handleCreateTask = async () => {
    if (!taskName || !datasetId || !user || !token) return;

    setTaskStatus('creating');
    try {
      const { trainingService } = await import('./services/trainingService');
      
      message.loading('创建训练任务...', 0);
      const task = await trainingService.createTrainingTask(
        taskName,
        datasetId,
        'flow_control',
        { epochs: 10, batch_size: 32, learning_rate: 0.001 },
        token,
        '流程控制模型训练'
      );
      message.destroy();
      message.success('训练任务创建成功');
      
      // Start training
      message.loading('启动训练...', 0);
      await trainingService.startTraining(task.id, token);
      message.destroy();
      message.success('训练已启动');
      
      setTaskStatus('training');
      
      // Poll training status
      pollTrainingStatus(task.id);
      
    } catch (error: any) {
      message.destroy();
      message.error(error.message || '创建任务失败');
      setTaskStatus('error');
    }
  };

  const pollTrainingStatus = async (taskId: number) => {
    const authToken = useAuthStore.getState().token;
    if (!authToken) return;
    const { trainingService } = await import('./services/trainingService');
    
    const interval = setInterval(async () => {
      try {
        const task = await trainingService.getTrainingStatus(taskId, authToken);
        
        if (task.status === 'completed') {
          clearInterval(interval);
          setTaskStatus('completed');
          message.success('训练完成');
        } else if (task.status === 'failed') {
          clearInterval(interval);
          setTaskStatus('failed');
          message.error('训练失败：' + ((task as any).error_message || '未知错误'));
        }
      } catch (error) {
        console.error('轮询训练状态失败：', error);
      }
    }, 5000);
    
    // Stop after 30 minutes
    setTimeout(() => clearInterval(interval), 1800000);
  };

  return (
    <div>
      <h1>训练管理</h1>
      
      <div style={{ marginTop: '24px' }}>
        <h3>1. 上传训练数据集</h3>
        <div style={{ marginTop: '16px', padding: '24px', border: '1px dashed #d9d9d9', borderRadius: '8px' }}>
          <input
            type="text"
            placeholder="数据集名称"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
            style={{ width: '100%', padding: '10px', border: '1px solid #d9d9d9', borderRadius: '4px', marginBottom: '16px' }}
          />
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx"
            onChange={(e) => e.target.files && setDatasetFile(e.target.files[0])}
            style={{ display: 'none' }}
          />
          <Button onClick={() => fileInputRef.current?.click()}>
            选择数据集文件
          </Button>
          {datasetFile && (
            <div style={{ marginTop: '16px' }}>
              <p>已选择：{datasetFile.name}</p>
              <Button type="primary" onClick={handleDatasetUpload} disabled={!datasetName} style={{ marginTop: '12px' }}>
                上传数据集
              </Button>
            </div>
          )}
          {uploadStatus !== 'idle' && (
            <div style={{ marginTop: '16px', padding: '12px', background: '#f5f5f5', borderRadius: '4px' }}>
              <p><strong>状态：</strong>{uploadStatus}</p>
              {datasetId && <p><strong>数据集ID：</strong>{datasetId}</p>}
            </div>
          )}
        </div>
      </div>

      {datasetId && (
        <div style={{ marginTop: '32px' }}>
          <h3>2. 创建训练任务</h3>
          <div style={{ marginTop: '16px', padding: '24px', border: '1px dashed #d9d9d9', borderRadius: '8px' }}>
            <input
              type="text"
              placeholder="任务名称"
              value={taskName}
              onChange={(e) => setTaskName(e.target.value)}
              style={{ width: '100%', padding: '10px', border: '1px solid #d9d9d9', borderRadius: '4px', marginBottom: '16px' }}
            />
            <Button type="primary" onClick={handleCreateTask} disabled={!taskName}>
              创建并启动训练
            </Button>
            {taskStatus !== 'idle' && (
              <div style={{ marginTop: '16px', padding: '12px', background: '#f5f5f5', borderRadius: '4px' }}>
                <p><strong>状态：</strong>{taskStatus}</p>
                {taskStatus === 'training' && <p>训练进行中，请稍候...</p>}
              </div>
            )}
          </div>
        </div>
      )}
      
      <div style={{ marginTop: '32px', padding: '16px', background: '#e6f7ff', borderRadius: '8px' }}>
        <h4>功能说明：</h4>
        <ul style={{ marginTop: '12px', lineHeight: '1.8' }}>
          <li>上传CSV或Excel格式的训练数据集</li>
          <li>创建训练任务并配置超参数</li>
          <li>实时监控训练进度和指标</li>
          <li>管理模型版本和评估结果</li>
        </ul>
      </div>
    </div>
  );
};

const App = () => {
  const { isAuthenticated, initialize } = useAuthStore();

  React.useEffect(() => {
    initialize();
  }, [initialize]);

  return (
    <ConfigProvider locale={zhCN}>
      <Router>
        <Routes>
          <Route path="/login" element={!isAuthenticated ? <LoginPage /> : <Navigate to="/" replace />} />
          <Route path="/register" element={!isAuthenticated ? <RegisterPage /> : <Navigate to="/" replace />} />
          <Route path="/" element={isAuthenticated ? <Dashboard /> : <Navigate to="/login" replace />} />
        </Routes>
      </Router>
    </ConfigProvider>
  );
};

export default App;
