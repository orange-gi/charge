import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link } from 'react-router-dom';
import { ConfigProvider, Layout as AntLayout, Menu, Button, message, Modal, Select, Spin, Card, Space, Typography, Progress, Tabs, Table, Tag, Empty } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { UserOutlined, FileTextOutlined, DatabaseOutlined, LogoutOutlined, UploadOutlined, FileOutlined, CloseOutlined, MenuFoldOutlined, MenuUnfoldOutlined, ThunderboltOutlined, PlusOutlined, MessageOutlined, CheckCircleOutlined, ReloadOutlined, DeleteOutlined, CheckOutlined, CodeOutlined, ControlOutlined, SearchOutlined, ToolOutlined, RobotOutlined, CloseCircleOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { useAuthStore } from './stores/authStore';
import './styles/globals.css';
import TrainingCenter from './pages/training/TrainingCenter';
import RagCenter from './pages/rag/RagCenter';

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
      <Header
        style={{
          background: 'rgba(255,255,255,0.92)',
          backdropFilter: 'blur(6px)',
          borderBottom: '1px solid #eef2f7',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 28px',
          height: '68px',
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          zIndex: 1000,
          boxShadow: '0 12px 30px rgba(15,23,42,0.08)'
        }}
      >
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
          {selectedKey === 'training' && <TrainingCenter />}
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
  const [dbcModalOpen, setDbcModalOpen] = React.useState(false);
  const [dbcFile, setDbcFile] = React.useState<File | null>(null);
  const [dbcUploading, setDbcUploading] = React.useState(false);
  const [dbcInfo, setDbcInfo] = React.useState<any>(null);
  const [loadingDbcInfo, setLoadingDbcInfo] = React.useState(false);
  const { user, token } = useAuthStore();
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const dbcFileInputRef = React.useRef<HTMLInputElement>(null);
  const workflowTrace = analysisData?.workflow_trace || {};

  const toDisplayText = React.useCallback((value: any): string => {
    if (value === null || value === undefined) return '';
    if (typeof value === 'string') return value;
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
    if (Array.isArray(value)) {
      return value.map((v) => toDisplayText(v)).filter(Boolean).join('\n');
    }
    if (typeof value === 'object') {
      // 常见结构：finding / evidence
      const v: any = value;
      const parts: string[] = [];
      if (v.signal) parts.push(`信号：${v.signal}`);
      if (v.description) parts.push(`描述：${v.description}`);
      if (v.finding) parts.push(String(v.finding));
      if (v.evidence) parts.push(`证据：${v.evidence}`);
      if (parts.length) return parts.join('\n');
      try {
        return JSON.stringify(value, null, 2);
      } catch {
        return String(value);
      }
    }
    return String(value);
  }, []);

  const getRiskLevel = React.useCallback((risk: any): string => {
    if (!risk) return '';
    if (typeof risk === 'string') return risk;
    if (typeof risk === 'object') {
      return String(risk.risk_level || risk.level || risk.severity || '');
    }
    return String(risk);
  }, []);

  // 后端“流程控制”特殊处理的 5 个关键信号（大小写不敏感）
  const KEY_SIGNALS = React.useMemo(
    () => ["BMS_DCChrgSt", "BMS_ChrgEndNum", "BMS_FaultNum1", "VIU0_FaultNum1", "CHM_ComVersion"],
    []
  );

  const pickDefaultSignals = React.useCallback(
    (signals: any[]) => {
      const norm = (s: any) => String(s ?? "").trim().toLowerCase();
      const byLower = new Map<string, string>();
      (signals || []).forEach((s: any) => {
        if (s?.name) byLower.set(norm(s.name), String(s.name));
      });

      const selected: string[] = [];
      const selectedLower = new Set<string>();

      // 1) 优先选择后端特殊处理的 5 个信号（不区分大小写）
      for (const k of KEY_SIGNALS) {
        const hit = byLower.get(norm(k));
        if (hit && !selectedLower.has(norm(hit))) {
          selected.push(hit);
          selectedLower.add(norm(hit));
        }
        if (selected.length >= 5) break;
      }

      // 2) 不足 5 个则补齐：优先常用信号，再用任意信号
      const isCommon = (name: string) => {
        const n = name.toLowerCase();
        return (
          n.includes("battery") ||
          n.includes("charge") ||
          n.includes("voltage") ||
          n.includes("current") ||
          n.includes("temp") ||
          n.includes("soc")
        );
      };

      const pool = [...(signals || [])]
        .filter((s: any) => s?.name)
        .sort((a: any, b: any) => Number(isCommon(String(b.name))) - Number(isCommon(String(a.name))));

      for (const s of pool) {
        if (selected.length >= 5) break;
        const name = String(s.name);
        const key = norm(name);
        if (!selectedLower.has(key)) {
          selected.push(name);
          selectedLower.add(key);
        }
      }

      return selected.slice(0, 5);
    },
    [KEY_SIGNALS]
  );

  const loadDbcInfo = React.useCallback(async () => {
    if (!token) return;
    setLoadingDbcInfo(true);
    try {
      const { chargingService } = await import('./services/chargingService');
      const info = await chargingService.getCurrentDbc(token);
      setDbcInfo(info);
    } catch (e: any) {
      // 不影响主流程：静默失败，仅在控制台记录
      console.warn('加载 DBC 配置失败:', e);
    } finally {
      setLoadingDbcInfo(false);
    }
  }, [token]);

  React.useEffect(() => {
    loadDbcInfo();
  }, [loadDbcInfo]);

  const handleUploadDbc = async () => {
    if (!token) {
      message.warning('请先登录');
      return;
    }
    if (!dbcFile) {
      message.warning('请选择 .dbc 文件');
      return;
    }
    if (!dbcFile.name.toLowerCase().endsWith('.dbc')) {
      message.warning('仅支持 .dbc 文件');
      return;
    }

    setDbcUploading(true);
    try {
      message.loading({ content: '正在上传 DBC...', key: 'dbcUpload' });
      const { chargingService } = await import('./services/chargingService');
      const info = await chargingService.uploadDbc(dbcFile, token);
      setDbcInfo(info);
      message.success({ content: 'DBC 配置成功，后续解析将使用该 DBC', key: 'dbcUpload' });
      setDbcModalOpen(false);
      setDbcFile(null);
      if (dbcFileInputRef.current) {
        dbcFileInputRef.current.value = '';
      }
    } catch (e: any) {
      message.error({ content: e?.message || '上传 DBC 失败', key: 'dbcUpload' });
    } finally {
      setDbcUploading(false);
    }
  };

  const formatDateTime = (value?: string) => {
    if (!value) return '--';
    try {
      return new Date(value).toLocaleString('zh-CN');
    } catch (error) {
      return value;
    }
  };

  const renderTraceInfo = (trace?: any) => {
    if (!trace) return null;
    const statusMap: Record<string, { label: string; color: string }> = {
      completed: { label: '已完成', color: '#52c41a' },
      running: { label: '进行中', color: '#2c5aa0' },
      failed: { label: '失败', color: '#ff4d4f' },
      pending: { label: '未开始', color: '#8c8c8c' },
    };
    const status = statusMap[trace.status] || statusMap.pending;
    return (
      <div style={{ marginBottom: '16px', padding: '12px', background: '#fafafa', borderRadius: '6px', border: '1px solid #f0f0f0' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '12px' }}>
          <Tag color={status.color} style={{ marginRight: 0 }}>{status.label}</Tag>
          {Array.isArray(trace.runs) && trace.runs.length > 1 && (
            <Tag color="#2c5aa0" style={{ marginRight: 0 }}>
              执行次数：{trace.runs.length}
            </Tag>
          )}
          {trace.started_at && (
            <Typography.Text type="secondary">
              开始：{formatDateTime(trace.started_at)}
            </Typography.Text>
          )}
          {trace.ended_at && (
            <Typography.Text type="secondary">
              结束：{formatDateTime(trace.ended_at)}
            </Typography.Text>
          )}
        </div>
        {trace.description && (
          <Typography.Text style={{ display: 'block', marginTop: '8px', color: '#595959' }}>
            {trace.description}
          </Typography.Text>
        )}
        {Array.isArray(trace.runs) && trace.runs.length > 1 && (
          <div style={{ marginTop: '10px' }}>
            <Typography.Text type="secondary" style={{ display: 'block', marginBottom: '6px' }}>
              历次执行：
            </Typography.Text>
            <div style={{ display: 'grid', gap: '6px' }}>
              {trace.runs.map((r: any, idx: number) => (
                <div key={idx} style={{ padding: '8px 10px', background: '#fff', border: '1px solid #f0f0f0', borderRadius: '6px' }}>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', alignItems: 'center' }}>
                    <Tag color={r.status === 'completed' ? '#52c41a' : r.status === 'failed' ? '#ff4d4f' : '#2c5aa0'} style={{ marginRight: 0 }}>
                      {r.status || 'unknown'}
                    </Tag>
                    {r.started_at && <Typography.Text type="secondary">开始：{formatDateTime(r.started_at)}</Typography.Text>}
                    {r.ended_at && <Typography.Text type="secondary">结束：{formatDateTime(r.ended_at)}</Typography.Text>}
                  </div>
                  {r.description && (
                    <Typography.Text style={{ display: 'block', marginTop: '6px', color: '#595959' }}>
                      {r.description}
                    </Typography.Text>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const getTraceStatus = (stepId: string): 'pending' | 'active' | 'completed' | 'failed' | null => {
    // stepId 可能是 nodeId 或 nodeId::run:N
    const parsed = parseStepKey(stepId);
    const trace = workflowTrace?.[parsed.nodeId];
    if (!trace) return null;
    const run = parsed.runIndex ? (Array.isArray(trace.runs) ? trace.runs[parsed.runIndex - 1] : null) : null;
    const status = String(run?.status || trace.status || '');
    if (status === 'completed') return 'completed';
    if (status === 'failed') return 'failed';
    if (status === 'running') return 'active';
    if (status === 'pending') return 'pending';
    return null;
  };

  const parseStepKey = (stepId: string): { nodeId: string; runIndex: number | null } => {
    const raw = String(stepId || '');
    const parts = raw.split('::run:');
    if (parts.length === 2) {
      const n = Number(parts[1]);
      return { nodeId: parts[0], runIndex: Number.isFinite(n) ? n : null };
    }
    return { nodeId: raw, runIndex: null };
  };

  const getTraceEntry = (stepId: string) => {
    const parsed = parseStepKey(stepId);
    const entry = workflowTrace?.[parsed.nodeId];
    if (!entry) return null;

    // 若选择的是某一轮 run，则把该 run“投影”为一个可展示的 trace（兼容 renderTraceInfo）
    if (parsed.runIndex && Array.isArray(entry.runs) && entry.runs[parsed.runIndex - 1]) {
      const run = entry.runs[parsed.runIndex - 1];
      return {
        ...entry,
        // 让详情卡片与渲染逻辑聚焦“这一轮”
        runs: [run],
        started_at: run?.started_at || entry.started_at,
        ended_at: run?.ended_at || entry.ended_at,
        status: run?.status || entry.status,
        description: run?.description || entry.description,
        metadata: run?.metadata || entry.metadata,
        output: run?.output || entry.output,
        error: run?.error || entry.error
      };
    }

    return entry;
  };

  // --- 流程展示（动态适配后端 langgraph 变化） ---
  // 1) fallbackWorkflowSteps：当后端尚未返回 workflow_trace 时，仍用“进度区间”做粗略展示（避免空白）
  // 2) 一旦 workflow_trace 可用：步骤列表完全来自 workflow_trace（节点增删/回环都能自动适配）
  const fallbackWorkflowSteps = [
    { id: 'file_upload', name: '文件上传', icon: FileOutlined, progressRange: [0, 10] },
    { id: 'file_validation', name: '文件验证', icon: CheckOutlined, progressRange: [10, 20] },
    { id: 'message_parsing', name: '报文解析', icon: CodeOutlined, progressRange: [20, 50] },
    { id: 'flow_control', name: '流程控制', icon: ControlOutlined, progressRange: [50, 60] },
    { id: 'rag_retrieval', name: 'RAG检索', icon: SearchOutlined, progressRange: [60, 80] },
    { id: 'detailed_analysis', name: '细化分析', icon: ToolOutlined, progressRange: [80, 90] },
    { id: 'llm_analysis', name: 'LLM分析', icon: RobotOutlined, progressRange: [90, 95] },
    { id: 'report_generation', name: '报告生成', icon: FileTextOutlined, progressRange: [95, 100] }
  ];

  // 已知节点的图标映射；后端新增节点时自动回退到默认图标（不影响展示）
  const workflowIconMap: Record<string, any> = React.useMemo(
    () => ({
      file_upload: FileOutlined,
      file_validation: CheckOutlined,
      message_parsing: CodeOutlined,
      flow_control: ControlOutlined,
      rag_retrieval: SearchOutlined,
      detailed_analysis: ToolOutlined,
      llm_analysis: RobotOutlined,
      report_generation: FileTextOutlined
    }),
    []
  );

  const getWorkflowIcon = React.useCallback(
    (nodeId: string) => workflowIconMap[nodeId] || ThunderboltOutlined,
    [workflowIconMap]
  );

  const safeParseTime = (value?: string): number => {
    if (!value) return Number.POSITIVE_INFINITY;
    const t = new Date(value).getTime();
    return Number.isFinite(t) ? t : Number.POSITIVE_INFINITY;
  };

  const getEntryFirstStartedAt = (entry: any): string | undefined => {
    // 后端支持循环：优先用 runs[0].started_at 作为“首次出现时间”
    const runs = Array.isArray(entry?.runs) ? entry.runs : [];
    return runs?.[0]?.started_at || entry?.started_at;
  };

  const isReactLikeStep = (entry: any): boolean => {
    // 识别 ReAct：优先看 metadata.phase，其次看 description 文案
    const desc = String(entry?.description || '');
    const meta = entry?.metadata || {};
    return Boolean(meta?.phase) || desc.toLowerCase().includes('react');
  };

  const buildWorkflowStepsFromTrace = React.useCallback(() => {
    const raw = workflowTrace || {};
    const entries = Object.entries(raw).map(([key, value]: [string, any]) => {
      const v = value || {};
      const id = String(v.node_id || key);
      return {
        id,
        name: String(v.name || id),
        trace: v
      };
    });

    // 排序规则：
    // - file_upload 永远最前（符合用户直觉）
    // - 其余按首次 started_at 升序（更接近真实执行顺序）
    // - 没有 started_at 的（pending）排在最后
    const sorted = entries.sort((a, b) => {
      if (a.id === 'file_upload') return -1;
      if (b.id === 'file_upload') return 1;
      const ta = safeParseTime(getEntryFirstStartedAt(a.trace));
      const tb = safeParseTime(getEntryFirstStartedAt(b.trace));
      if (ta !== tb) return ta - tb;
      return a.name.localeCompare(b.name);
    });

    return sorted.map((item) => {
      const status = getTraceStatus(item.id) || 'pending';
      return {
        id: item.id,
        name: item.name,
        icon: getWorkflowIcon(item.id),
        status,
        isReact: isReactLikeStep(item.trace),
        trace: item.trace
      };
    });
  }, [workflowTrace, getTraceStatus, getWorkflowIcon]);

  // 把 workflow_trace 展开为“卡片实例列表”：
  // - 同一节点多次执行（runs）会生成多张卡片（nodeId::run:N）
  // - 卡片按 started_at 排序，更贴近真实执行顺序（尤其是 ReAct 回环）
  // - 没有 started_at 的 pending 节点会被追加到末尾（保持可预期的流程感）
  const buildWorkflowCardInstances = React.useCallback(() => {
    const raw = workflowTrace || {};
    const startedCards: any[] = [];
    const pendingCards: any[] = [];

    Object.entries(raw).forEach(([key, value]: [string, any]) => {
      const entry = value || {};
      const nodeId = String(entry.node_id || key);
      const nodeName = String(entry.name || nodeId);
      if (nodeId === 'file_upload') return; // 顶部单独的文件卡片已经覆盖

      const runs = Array.isArray(entry.runs) ? entry.runs : [];
      if (runs.length > 0) {
        runs.forEach((r: any, idx: number) => {
          const startedAt = r?.started_at || entry?.started_at;
          const status = String(r?.status || entry?.status || 'pending');
          const stepId = `${nodeId}::run:${idx + 1}`;
          const card = {
            id: stepId,
            nodeId,
            runIndex: idx + 1,
            name: nodeName,
            // 卡片标题尽量用“真实描述”，否则用默认 name
            displayName: r?.description ? String(r.description) : nodeName,
            icon: getWorkflowIcon(nodeId),
            status: status === 'running' ? 'active' : status,
            startedAt
          };
          if (startedAt) {
            startedCards.push(card);
          } else {
            // 极端情况：run 没有 started_at，就当 pending 放后面
            pendingCards.push({ ...card, status: 'pending' });
          }
        });
      } else {
        // 没有 runs：单次执行节点，生成一个卡片（nodeId）
        const startedAt = entry?.started_at;
        const status = String(entry?.status || 'pending');
        const card = {
          id: nodeId,
          nodeId,
          runIndex: null,
          name: nodeName,
          displayName: entry?.description ? String(entry.description) : nodeName,
          icon: getWorkflowIcon(nodeId),
          status: status === 'running' ? 'active' : status,
          startedAt
        };
        if (startedAt) startedCards.push(card);
        else pendingCards.push({ ...card, status: 'pending' });
      }
    });

    startedCards.sort((a, b) => safeParseTime(a.startedAt) - safeParseTime(b.startedAt));
    // pendingCards 按 name 排序即可（不强行定义拓扑顺序，避免写死流程）
    pendingCards.sort((a, b) => String(a.name).localeCompare(String(b.name)));

    // 若 file_validation/message_parsing 等在 trace 内但尚未开始，会出现在 pendingCards 里
    return [...startedCards, ...pendingCards];
  }, [workflowTrace, getWorkflowIcon]);

  // 根据进度获取当前步骤
  const getCurrentSteps = () => {
    if (!file) return [];
    
    const steps: any[] = [];
    const hasTrace = analysisData && Object.keys(workflowTrace || {}).length > 0;
    
    // 文件上传步骤（始终显示）
    steps.push({
      ...fallbackWorkflowSteps[0],
      status: 'completed',
      data: { fileName: file?.name, fileSize: file?.size }
    });

    // 优先：从 workflow_trace 动态构建步骤（自动适配 langgraph 节点增删/回环）
    if (hasTrace) {
      const cards = buildWorkflowCardInstances();
      for (const c of cards) {
        steps.push({
          id: c.id,
          // 卡片展示优先用 displayName（更贴近 ReAct 每轮做了什么）
          name: c.displayName || c.name,
          icon: c.icon,
          status: c.status,
          // 让 UI 更容易区分“同一节点的多轮”
          runIndex: c.runIndex,
          nodeId: c.nodeId
        });
      }
      return steps;
    }

    // fallback：无 trace 时仍用进度区间展示（避免“分析中但步骤空白”）
    if (status === 'uploaded' || status === 'analyzing' || status === 'completed') {
      // 文件验证（上传后默认认为已完成；开始分析后按进度活跃）
      const validationStatus = getTraceStatus('file_validation');
      steps.push({
        ...fallbackWorkflowSteps[1],
        status: validationStatus || (status === 'uploaded' ? 'completed' : status === 'analyzing' ? 'active' : 'pending')
      });
    }

    if (status === 'analyzing' || status === 'completed') {
      const currentProgress = analysisProgress;
      
      for (let i = 2; i < fallbackWorkflowSteps.length; i++) {
        const step = fallbackWorkflowSteps[i];
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

  // 执行过程时间线（按 started_at 排序，展示 ReAct 的回环 runs）
  const buildTraceEvents = React.useCallback(() => {
    const raw = workflowTrace || {};
    const events: any[] = [];

    Object.entries(raw).forEach(([key, value]: [string, any]) => {
      const entry = value || {};
      const nodeId = String(entry.node_id || key);
      const nodeName = String(entry.name || nodeId);
      const runs = Array.isArray(entry.runs) && entry.runs.length > 0 ? entry.runs : null;

      const pushEvent = (run: any, runIndex: number | null) => {
        const startedAt = run?.started_at || entry?.started_at;
        if (!startedAt) return; // 没有开始时间的（纯 pending）不进入时间线
        events.push({
          nodeId,
          nodeName,
          runIndex,
          status: String(run?.status || entry?.status || 'pending'),
          startedAt,
          endedAt: run?.ended_at || entry?.ended_at,
          description: run?.description || entry?.description,
          metadata: run?.metadata || entry?.metadata,
          output: run?.output || entry?.output,
          error: run?.error || entry?.error
        });
      };

      if (runs) {
        runs.forEach((r: any, idx: number) => pushEvent(r, idx + 1));
      } else {
        pushEvent(entry, null);
      }
    });

    return events.sort((a, b) => safeParseTime(a.startedAt) - safeParseTime(b.startedAt));
  }, [workflowTrace]);

  const renderExecutionTimeline = () => {
    const hasTrace = analysisData && Object.keys(workflowTrace || {}).length > 0;
    if (!hasTrace) return null;
    const events = buildTraceEvents();
    if (!events.length) return null;

    const statusColor = (s: string) => {
      if (s === 'completed') return '#52c41a';
      if (s === 'failed') return '#ff4d4f';
      if (s === 'running') return '#2c5aa0';
      return '#8c8c8c';
    };

    return (
      <Card title="执行过程（按时间排序，自动适配 ReAct 回环）" style={{ marginTop: '24px', width: '100%' }}>
        <div style={{ display: 'grid', gap: '10px' }}>
          {events.map((ev, idx) => {
            const isReact = Boolean(ev?.metadata?.phase) || String(ev?.description || '').toLowerCase().includes('react');
            const runLabel = ev.runIndex ? `#${ev.runIndex}` : '';
            return (
              <div
                key={`${ev.nodeId}-${ev.runIndex ?? 'single'}-${idx}`}
                style={{
                  border: `1px solid #f0f0f0`,
                  borderLeft: `4px solid ${statusColor(ev.status)}`,
                  borderRadius: '8px',
                  padding: '10px 12px',
                  background: '#fff'
                }}
              >
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', alignItems: 'center' }}>
                  <Tag color={statusColor(ev.status)} style={{ marginRight: 0 }}>
                    {ev.status}
                  </Tag>
                  {isReact && (
                    <Tag color="#2c5aa0" style={{ marginRight: 0 }}>
                      ReAct
                    </Tag>
                  )}
                  <Typography.Text strong style={{ cursor: 'pointer' }} onClick={() => setSelectedStep(ev.nodeId)}>
                    {ev.nodeName}{runLabel ? ` ${runLabel}` : ''}
                  </Typography.Text>
                  {ev.startedAt && (
                    <Typography.Text type="secondary">
                      开始：{formatDateTime(ev.startedAt)}
                    </Typography.Text>
                  )}
                  {ev.endedAt && (
                    <Typography.Text type="secondary">
                      结束：{formatDateTime(ev.endedAt)}
                    </Typography.Text>
                  )}
                </div>

                {ev.description && (
                  <Typography.Text style={{ display: 'block', marginTop: '6px', color: '#595959' }}>
                    {ev.description}
                  </Typography.Text>
                )}

                {/* 只展示“必要信息”：metadata/output/error 的精简版文本，避免铺满屏幕 */}
                {(ev.metadata || ev.output || ev.error) && (
                  <div style={{ marginTop: '8px', display: 'grid', gap: '6px' }}>
                    {ev.metadata && (
                      <div style={{ background: '#fafafa', border: '1px solid #f0f0f0', borderRadius: '6px', padding: '8px' }}>
                        <Typography.Text type="secondary" style={{ display: 'block', marginBottom: '4px' }}>
                          metadata
                        </Typography.Text>
                        <Typography.Text style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px' }}>
                          {toDisplayText(ev.metadata)}
                        </Typography.Text>
                      </div>
                    )}
                    {ev.output && (
                      <div style={{ background: '#fafafa', border: '1px solid #f0f0f0', borderRadius: '6px', padding: '8px' }}>
                        <Typography.Text type="secondary" style={{ display: 'block', marginBottom: '4px' }}>
                          output
                        </Typography.Text>
                        <Typography.Text style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px' }}>
                          {toDisplayText(ev.output)}
                        </Typography.Text>
                      </div>
                    )}
                    {ev.error && (
                      <div style={{ background: '#fff1f0', border: '1px solid #ffccc7', borderRadius: '6px', padding: '8px' }}>
                        <Typography.Text type="secondary" style={{ display: 'block', marginBottom: '4px' }}>
                          error
                        </Typography.Text>
                        <Typography.Text style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px' }}>
                          {toDisplayText(ev.error)}
                        </Typography.Text>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </Card>
    );
  };

  // 获取步骤的详细信息
  const getStepDetails = (stepId: string) => {
    // 若点击的是某一轮 run（nodeId::run:N），优先展示“该轮的 trace 信息”，避免被写死的分支覆盖
    const parsed = parseStepKey(stepId);
    if (parsed.runIndex) {
      const trace = getTraceEntry(stepId);
      return {
        title: `${trace?.name || parsed.nodeId}（第${parsed.runIndex}轮）`,
        data: {
          trace
        }
      };
    }

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
          fileType: file?.name.split('.').pop()?.toUpperCase(),
          trace: workflowTrace?.file_upload
        }
      },
      file_validation: {
        title: '文件验证',
        data: {
          validationStatus: analysisData?.validation_status,
          validationMessage: analysisData?.validation_message,
          trace: workflowTrace?.file_validation
        }
      },
      message_parsing: {
        title: '报文解析',
        data: analysisData.data_stats ? {
          dataStats: analysisData.data_stats,
          signalCount: analysisData.data_stats.signal_count || 0,
          totalRecords: analysisData.data_stats.total_records || 0,
          signalStats: analysisData.data_stats.signal_stats || {},
          timeRange: analysisData.data_stats.time_range || {},
          parsedRecords: analysisData.parsed_data || [],
          rawMessages: analysisData.raw_messages || [],  // 原始消息数据
          selectedSignals: analysisData.selected_signals || [],  // 选择的信号列表
          trace: workflowTrace?.message_parsing
        } : {}
      },
      flow_control: {
        title: '流程控制',
        data: {
          flowAnalysis: analysisData.flow_analysis || {},
          problemDirection: analysisData.problem_direction,
          confidenceScore: analysisData.confidence_score,
          signalWindowsList: analysisData.signal_windows_list || [],
          signalRuleMeta: analysisData.signal_rule_meta || {},
          ragQueries: analysisData.rag_queries || [],
          processedSignals: analysisData.processed_signals || [],
          trace: workflowTrace?.flow_control
        }
      },
      rag_retrieval: {
        title: 'RAG检索',
        data: {
          retrievedDocuments: analysisData.retrieved_documents || [],
          retrievalContext: analysisData.retrieval_context || '',
          retrievalByQuery: analysisData.retrieval_by_query || {},
          trace: workflowTrace?.rag_retrieval
        }
      },
      detailed_analysis: {
        title: '细化分析',
        data: {
          refinedSignals: analysisData.refined_signals || [],
          signalValidation: analysisData.signal_validation || {},
          refineResult: analysisData.refine_result || {},
          refineConfidence: analysisData.refine_confidence,
          additionalSignals: analysisData.additional_signals || [],
          trace: workflowTrace?.detailed_analysis
        }
      },
      llm_analysis: {
        title: 'LLM分析',
        data: {
          ...(analysisData.llm_analysis || {}),
          trace: workflowTrace?.llm_analysis
        }
      },
      report_generation: {
        title: '报告生成',
        data: {
          finalReport: analysisData.final_report || {},
          visualizations: analysisData.visualizations || [],
          llmAnalysis: analysisData.llm_analysis || {},
          results: results || [],
          trace: workflowTrace?.report_generation
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

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      // 如果当前状态是错误或失败，重置状态以允许重新上传
      if (status === 'error' || status === 'failed' || status === 'cancelled') {
        setStatus('idle');
        setAnalysisId(null);
        setResults([]);
        setSelectedSignals([]);
        setAnalysisProgress(0);
        setProgressMessage('');
      }
      
      await handleUpload(selectedFile);
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
      // 默认勾选：优先“流程控制特殊处理的5个信号”，不足则补齐5个
      const defaults = pickDefaultSignals(response.signals || []);
      if (status === 'uploaded') {
        setSelectedSignals(defaults);
      }
    } catch (error: any) {
      console.error('加载信号列表失败:', error);
      message.warning('加载信号列表失败，将使用默认信号');
    } finally {
      setLoadingSignals(false);
    }
  };

  const handleUpload = async (selectedFile?: File) => {
    const fileToUpload = selectedFile || file;
    if (!fileToUpload || !user || !token) return;

    setStatus('uploading');
    setResults([]);
    setSelectedSignals([]);
    try {
      const { chargingService } = await import('./services/chargingService');
      
      message.loading('上传中...', 0);
      const analysis = await chargingService.uploadFile(fileToUpload, token, fileToUpload.name);
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
          // 即使失败，也尽量加载 resultData，让前端能展示“已完成的节点”与“失败节点报错”
          if (analysis.resultData) {
            try {
              const data = JSON.parse(analysis.resultData);
              setAnalysisData(data);
            } catch (e) {
              console.error('解析失败时的分析数据失败:', e);
            }
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
      <Modal
        title="配置DBC（用于报文解析）"
        open={dbcModalOpen}
        onCancel={() => {
          setDbcModalOpen(false);
          setDbcFile(null);
          if (dbcFileInputRef.current) {
            dbcFileInputRef.current.value = '';
          }
        }}
        okText="上传并启用"
        cancelText="取消"
        onOk={handleUploadDbc}
        okButtonProps={{ loading: dbcUploading, disabled: !dbcFile }}
        destroyOnClose
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ fontSize: '13px', color: '#595959' }}>
            当前解析会优先使用你上传配置的 DBC（按用户生效）。如果未配置，则回退到系统内置 DBC。
          </div>

          <div style={{ padding: '10px 12px', background: '#fafafa', border: '1px solid #f0f0f0', borderRadius: '8px' }}>
            {loadingDbcInfo ? (
              <Spin size="small" />
            ) : dbcInfo?.configured ? (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
                <Tag color="black">已配置</Tag>
                <span style={{ fontWeight: 500 }}>{dbcInfo.filename || '-'}</span>
                {dbcInfo.uploadedAt && <span style={{ color: '#8c8c8c' }}>上传时间：{dbcInfo.uploadedAt}</span>}
              </div>
            ) : (
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <Tag>未配置</Tag>
                <span style={{ color: '#8c8c8c' }}>当前使用系统默认 DBC</span>
              </div>
            )}
          </div>

          <div>
            <input
              ref={dbcFileInputRef}
              type="file"
              accept=".dbc"
              onChange={(e) => setDbcFile(e.target.files?.[0] || null)}
            />
            <div style={{ marginTop: '8px', fontSize: '12px', color: '#8c8c8c' }}>
              提示：建议使用与日志采集一致的 DBC；若解析仍为空，请把后端日志中“首帧采样/未定义CAN ID/解码错误采样”贴出来排查。
            </div>
          </div>
        </div>
      </Modal>
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
          
          {/* 配置DBC按钮（优先使用用户上传的 DBC 解析） */}
          <Button
            block
            size="large"
            onClick={() => {
              setDbcModalOpen(true);
              // 打开弹窗时刷新一次状态（避免多端/多页面操作后不同步）
              loadDbcInfo();
            }}
            style={{
              height: '44px',
              borderRadius: '8px',
              border: '1px solid #1a1a1a',
              background: '#1a1a1a',
              color: 'white',
              fontWeight: '400',
              marginBottom: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px'
            }}
          >
            配置DBC
          </Button>

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
              // 新建分析时也刷新一次 DBC 信息，便于用户确认解析来源
              loadDbcInfo();
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
              padding: (status === 'uploaded' || status === 'analyzing' || status === 'completed') ? '8px' : '12px 16px',
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
              {/* 成功上传后隐藏图标和文字，只显示删除按钮 */}
              {(status === 'uploaded' || status === 'analyzing' || status === 'completed') ? (
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

            {/* 动态流程卡片 */}
            {getCurrentSteps().slice(1).map((step, index) => {
              const StepIcon = step.icon;
              const isActive = step.status === 'active';
              const isCompleted = step.status === 'completed';
              const isFailed = step.status === 'failed';
              const isSelected = selectedStep === step.id;
              
              return (
                <React.Fragment key={step.id}>
                  {/* 连接线 */}
                  <div style={{
                    width: '24px',
                    height: '2px',
                    background: isFailed ? '#ff4d4f' : isCompleted ? '#52c41a' : isActive ? '#2c5aa0' : '#e8e8e8',
                    transition: 'all 0.3s',
                    flexShrink: 0
                  }} />
                  
                  {/* 流程卡片 */}
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '12px 16px',
                    background: isFailed ? '#fff1f0' : isSelected ? '#e6f7ff' : 'white',
                    border: `2px solid ${isFailed ? '#ff4d4f' : isCompleted ? '#52c41a' : isActive ? '#2c5aa0' : '#e8e8e8'}`,
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
                      color: isFailed ? '#ff4d4f' : isCompleted ? '#52c41a' : isActive ? '#2c5aa0' : '#8c8c8c'
                    }} />
                    <span style={{ 
                      fontSize: '14px', 
                      color: isFailed ? '#ff4d4f' : isCompleted ? '#52c41a' : isActive ? '#2c5aa0' : '#8c8c8c',
                      fontWeight: isActive ? '500' : '400',
                      whiteSpace: 'nowrap'
                    }}>
                      {step.name}
                    </span>
                    {/* 多轮执行的徽标（例如 flow_control::run:2） */}
                    {step.runIndex && (
                      <Tag
                        color="#2c5aa0"
                        style={{
                          marginRight: 0,
                          fontSize: '11px',
                          lineHeight: '16px',
                          padding: '0 6px',
                          height: '18px'
                        }}
                      >
                        第{step.runIndex}轮
                      </Tag>
                    )}
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
                    {isFailed && (
                      <CloseCircleOutlined style={{
                        fontSize: '16px',
                        color: '#ff4d4f',
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
                      {renderTraceInfo(details.data.trace)}
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
                
                case 'file_validation': {
                  const validationStatus = details.data.validationStatus;
                  const statusLabel = validationStatus === 'passed' ? '通过' :
                    validationStatus === 'failed' ? '失败' :
                    validationStatus ? validationStatus : '未知';
                  const statusColor = validationStatus === 'passed' ? '#52c41a' :
                    validationStatus === 'failed' ? '#ff4d4f' : '#8c8c8c';
                  return (
                    <div>
                      {renderTraceInfo(details.data.trace)}
                      <div style={{ marginBottom: '16px' }}>
                        <Typography.Text strong>验证结果：</Typography.Text>
                        <Typography.Text style={{ color: statusColor, marginLeft: '8px' }}>
                          {statusLabel}
                        </Typography.Text>
                      </div>
                      {details.data.validationMessage && (
                        <div>
                          <Typography.Text strong>说明：</Typography.Text>
                          <Typography.Text style={{ marginLeft: '8px' }}>
                            {details.data.validationMessage}
                          </Typography.Text>
                        </div>
                      )}
                    </div>
                  );
                }
                
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
                  const parsedRecords = details.data.parsedRecords || [];
                  const parsedColumns = parsedRecords.length > 0
                    ? Object.keys(parsedRecords[0]).map((key) => ({
                        title: key,
                        dataIndex: key,
                        key
                      }))
                    : [];
                  const parsedTableData = parsedRecords.map((row: any, index: number) => ({
                    key: index,
                    ...row
                  }));
                  
                  // 原始消息数据
                  const rawMessages = details.data.rawMessages || [];
                  const rawMessagesColumns = rawMessages.length > 0
                    ? Object.keys(rawMessages[0]).map((key) => ({
                        title: key,
                        dataIndex: key,
                        key,
                        width: key === 'Data' ? 200 : undefined,
                        render: (text: any) => {
                          if (key === 'Time') {
                            return typeof text === 'number' ? text.toFixed(6) : text;
                          }
                          return text;
                        }
                      }))
                    : [];
                  const rawMessagesTableData = rawMessages.map((row: any, index: number) => ({
                    key: index,
                    ...row
                  }));

                  const overviewContent = (
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

                  const rawDataContent = parsedRecords.length > 0 ? (
                    <Table
                      size="small"
                      columns={parsedColumns}
                      dataSource={parsedTableData}
                      pagination={{ pageSize: 20, showSizeChanger: true, pageSizeOptions: ['20', '50', '100'] }}
                      scroll={{ x: 'max-content', y: 420 }}
                    />
                  ) : (
                    <div style={{ textAlign: 'center', padding: '40px', color: '#8c8c8c' }}>
                      暂无解析数据
                    </div>
                  );
                  
                  // 图表数据准备
                  const selectedSignalsList = details.data.selectedSignals || [];
                  const metadataColumns = ['timestamp', 'ts', 'time', 'Time', 'can_id', 'message_name', 'dlc'];
                  
                  // 获取要展示的信号列表（如果选择了信号，则只展示选择的；否则展示所有数值型信号）
                  // 如果用户选择了信号，严格只显示这些信号（即使数据中没有也要显示，这样用户知道哪些信号缺失）
                  const signalsToDisplay = selectedSignalsList.length > 0 
                    ? selectedSignalsList  // 直接使用选择的信号列表，不进行过滤
                    : parsedColumns
                        .filter((col: any) => !metadataColumns.includes(col.key))
                        .slice(0, 10)  // 最多展示10个信号
                        .map((col: any) => col.key);
                  
                  // 准备图表数据
                  const chartData: any[] = [];
                  const timeColumn = parsedColumns.find((col: any) => ['ts', 'timestamp', 'time', 'Time'].includes(col.key));
                  const timeKey = timeColumn ? timeColumn.key : null;
                  
                  signalsToDisplay.forEach((signalName: string) => {
                    if (!parsedRecords || parsedRecords.length === 0) return;
                    
                    // 检查信号是否在数据中存在
                    const signalExists = parsedRecords.some((record: any) => signalName in record);
                    if (!signalExists && selectedSignalsList.length > 0) {
                      // 如果用户选择了信号但数据中没有，创建一个空图表提示
                      chartData.push({
                        signalName,
                        timeData: [],
                        valueData: [],
                        missing: true
                      });
                      return;
                    }
                    
                    // 提取时间和信号值
                    const timeData: any[] = [];
                    const valueData: any[] = [];
                    
                    parsedRecords.forEach((record: any, index: number) => {
                      let timeValue = null;
                      if (timeKey && record[timeKey]) {
                        try {
                          const time = typeof record[timeKey] === 'string' 
                            ? new Date(record[timeKey]).getTime() 
                            : record[timeKey];
                          timeValue = time;
                        } catch (e) {
                          timeValue = index;  // 如果时间解析失败，使用索引
                        }
                      } else {
                        timeValue = index;
                      }
                      
                      const signalValue = record[signalName];
                      if (signalValue !== null && signalValue !== undefined && !isNaN(signalValue)) {
                        timeData.push(timeValue);
                        valueData.push(Number(signalValue));
                      }
                    });
                    
                    if (valueData.length > 0 || selectedSignalsList.length > 0) {
                      // 即使用户选择了信号但数据为空，也显示图表（用于提示）
                      chartData.push({
                        signalName,
                        timeData,
                        valueData,
                        missing: valueData.length === 0 && selectedSignalsList.length > 0
                      });
                    }
                  });
                  
                  // 创建图表配置（每个信号一个图表，垂直排列）
                  const graphContent = chartData.length > 0 ? (
                    <div style={{ maxHeight: '600px', overflowY: 'auto' }}>
                      {chartData.map((data, index) => {
                        // 如果信号缺失或数据为空，显示提示
                        if (data.missing || (data.valueData.length === 0 && selectedSignalsList.length > 0)) {
                          return (
                            <div key={data.signalName} style={{ marginBottom: '20px', border: '1px solid #ffccc7', borderRadius: '8px', padding: '12px', background: '#fff2f0' }}>
                              <Typography.Text strong style={{ color: '#ff4d4f' }}>
                                {data.signalName}
                              </Typography.Text>
                              <Typography.Text type="secondary" style={{ display: 'block', marginTop: '8px' }}>
                                该信号在解析数据中未找到，可能信号名称不匹配或数据中确实不包含此信号
                              </Typography.Text>
                            </div>
                          );
                        }
                        
                        const option = {
                          title: {
                            text: data.signalName,
                            left: 'center',
                            textStyle: { fontSize: 14, fontWeight: 'bold' }
                          },
                          tooltip: {
                            trigger: 'axis',
                            formatter: (params: any) => {
                              const param = params[0];
                              const timeStr = timeKey 
                                ? new Date(param.value[0]).toLocaleString('zh-CN')
                                : `索引: ${param.value[0]}`;
                              return `${data.signalName}<br/>${timeStr}<br/>值: ${param.value[1]}`;
                            }
                          },
                          grid: {
                            left: '3%',
                            right: '4%',
                            bottom: '3%',
                            containLabel: true,
                            height: '80px'  // 高度改为原来的1/3（200px -> 80px）
                          },
                          xAxis: {
                            type: 'category',
                            boundaryGap: false,
                            data: data.timeData.map((t: any) => {
                              if (timeKey) {
                                try {
                                  return new Date(t).toLocaleTimeString('zh-CN');
                                } catch {
                                  return t;
                                }
                              }
                              return t;
                            }),
                            axisLabel: {
                              rotate: 45,
                              fontSize: 10
                            }
                          },
                          yAxis: {
                            type: 'value',
                            axisLabel: {
                              fontSize: 10
                            }
                          },
                          series: [{
                            name: data.signalName,
                            type: 'line',
                            smooth: true,
                            data: data.valueData,  // 直接使用数值数组
                            symbol: 'circle',
                            symbolSize: 4,
                            lineStyle: {
                              width: 2
                            },
                            itemStyle: {
                              color: `hsl(${(index * 137.5) % 360}, 70%, 50%)`  // 使用不同颜色
                            }
                          }]
                        };
                        
                        return (
                          <div key={data.signalName} style={{ marginBottom: '20px', border: '1px solid #e8e8e8', borderRadius: '8px', padding: '12px' }}>
                            <ReactECharts 
                              option={option} 
                              style={{ height: '120px', width: '100%' }}  // 高度改为原来的1/3（250px -> 120px）
                              opts={{ renderer: 'canvas' }}
                            />
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center', padding: '40px', color: '#8c8c8c' }}>
                      暂无信号数据可用于图表展示
                      {selectedSignalsList.length > 0 && (
                        <div style={{ marginTop: '16px' }}>
                          <Typography.Text type="secondary">
                            已选择 {selectedSignalsList.length} 个信号，但数据中未找到这些信号
                          </Typography.Text>
                        </div>
                      )}
                    </div>
                  );
                  
                  return (
                    <div>
                      {renderTraceInfo(details.data.trace)}
                      <Tabs
                        defaultActiveKey="overview"
                        items={[
                          { key: 'overview', label: '数据概览', children: overviewContent },
                          { key: 'graph', label: signalsToDisplay.length > 0 ? `图表（${signalsToDisplay.length} 个信号）` : '图表', children: graphContent },
                          { key: 'parsed', label: parsedRecords.length ? `解析数据（${parsedRecords.length} 行）` : '解析数据', children: rawDataContent },
                          { key: 'raw', label: rawMessages.length ? `原始数据（${rawMessages.length} 行）` : '原始数据', children: rawMessages.length > 0 ? (
                            <Table
                              columns={rawMessagesColumns}
                              dataSource={rawMessagesTableData}
                              pagination={{ pageSize: 20, showSizeChanger: true, pageSizeOptions: ['20', '50', '100'] }}
                              scroll={{ x: 'max-content', y: 420 }}
                              size="small"
                            />
                          ) : (
                            <div style={{ textAlign: 'center', padding: '40px', color: '#8c8c8c' }}>
                              暂无原始消息数据
                            </div>
                          ) }
                        ]}
                      />
                    </div>
                  );
                
                case 'flow_control':
                  return (
                    <div>
                      {renderTraceInfo(details.data.trace)}
                      {(() => {
                        const flowAnalysis = details.data.flowAnalysis || {};
                        const signalWindowsList = details.data.signalWindowsList || [];
                        const ragQueries = details.data.ragQueries || [];
                        const signalRuleMeta = details.data.signalRuleMeta || {};
                        const processedSignals = details.data.processedSignals || [];

                        return (
                          <div>
                            <div style={{ display: 'grid', gap: '10px', marginBottom: '12px' }}>
                              <div>
                                <Typography.Text strong>问题方向：</Typography.Text>
                                <Typography.Text>{details.data.problemDirection || '未确定'}</Typography.Text>
                              </div>
                              <div>
                                <Typography.Text strong>当前置信度：</Typography.Text>
                                <Typography.Text>
                                  {typeof details.data.confidenceScore === 'number'
                                    ? `${(details.data.confidenceScore * 100).toFixed(1)}%`
                                    : '未知'}
                                </Typography.Text>
                              </div>
                              <div>
                                <Typography.Text strong>已规则化信号：</Typography.Text>
                                <Typography.Text>
                                  {Array.isArray(flowAnalysis.handled_signals)
                                    ? flowAnalysis.handled_signals.length
                                    : (flowAnalysis.handled_signal_count ?? signalWindowsList.length ?? 0)}
                                </Typography.Text>
                              </div>
                              <div>
                                <Typography.Text strong>RAG 查询数：</Typography.Text>
                                <Typography.Text>{ragQueries.length}</Typography.Text>
                              </div>
                            </div>

                            <Tabs
                              defaultActiveKey="signals"
                              items={[
                                {
                                  key: 'signals',
                                  label: signalWindowsList.length ? `规则化信号（${signalWindowsList.length}）` : '规则化信号',
                                  children: (
                                    <pre style={{ background: '#f5f5f5', padding: '12px', borderRadius: '6px', overflow: 'auto', maxHeight: '420px' }}>
                                      {JSON.stringify(signalWindowsList, null, 2)}
                                    </pre>
                                  )
                                },
                                {
                                  key: 'queries',
                                  label: ragQueries.length ? `RAG查询（${ragQueries.length}）` : 'RAG查询',
                                  children: ragQueries.length ? (
                                    <Table
                                      size="small"
                                      rowKey={(r: any, i: number) => `${r.query || ''}-${i}`}
                                      pagination={{ pageSize: 8, showSizeChanger: true, pageSizeOptions: ['8', '20', '50'] }}
                                      columns={[
                                        { title: '信号', dataIndex: 'signal', key: 'signal', width: 180 },
                                        { title: '值', dataIndex: 'value', key: 'value', width: 80 },
                                        { title: '查询', dataIndex: 'query', key: 'query' },
                                        { title: '时间段数', key: 'intervals', width: 90, render: (_: any, r: any) => (r.intervals?.length ?? 0) }
                                      ]}
                                      dataSource={ragQueries}
                                      scroll={{ x: 'max-content', y: 360 }}
                                    />
                                  ) : (
                                    <div style={{ textAlign: 'center', padding: '24px', color: '#8c8c8c' }}>
                                      未生成 RAG 查询（可能关键5信号在数据中未解析到）
                                    </div>
                                  )
                                },
                                {
                                  key: 'meta',
                                  label: '枚举/规则释义',
                                  children: (
                                    <pre style={{ background: '#f5f5f5', padding: '12px', borderRadius: '6px', overflow: 'auto', maxHeight: '420px' }}>
                                      {JSON.stringify(signalRuleMeta, null, 2)}
                                    </pre>
                                  )
                                },
                                {
                                  key: 'processed',
                                  label: processedSignals.length ? `已处理（${processedSignals.length}）` : '已处理',
                                  children: (
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                                      {processedSignals.map((s: string) => (
                                        <span key={s} style={{ padding: '4px 10px', background: '#f0f5ff', borderRadius: '4px', fontSize: '13px' }}>
                                          {s}
                                        </span>
                                      ))}
                                    </div>
                                  )
                                }
                              ]}
                            />
                          </div>
                        );
                      })()}
                    </div>
                  );
                
                case 'rag_retrieval':
                  return (
                    <div>
                      {renderTraceInfo(details.data.trace)}
                      <div style={{ marginBottom: '16px' }}>
                        <Typography.Text strong>检索到文档数：</Typography.Text>
                        <Typography.Text>{details.data.retrievedDocuments?.length || 0}</Typography.Text>
                      </div>
                      {(() => {
                        const retrievalByQuery = details.data.retrievalByQuery || {};
                        const queryEntries = Object.entries(retrievalByQuery);
                        if (!queryEntries.length) return null;
                        return (
                          <div style={{ marginBottom: '16px' }}>
                            <Typography.Text strong style={{ display: 'block', marginBottom: '8px' }}>
                              按查询分组（已过滤相似度≥70%）：
                            </Typography.Text>
                            <div style={{ display: 'grid', gap: '8px' }}>
                              {queryEntries.map(([q, docs]: any) => (
                                <div key={q} style={{ padding: '10px 12px', background: '#f5f5f5', borderRadius: '6px' }}>
                                  <Typography.Text strong>{q}</Typography.Text>
                                  <Typography.Text type="secondary" style={{ marginLeft: '10px' }}>
                                    命中：{Array.isArray(docs) ? docs.length : 0}
                                  </Typography.Text>
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })()}
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
                                <Typography.Text>
                                  {doc.filename || doc.metadata?.source_filename || doc.metadata?.source || '未知'}
                                  {doc.row_index != null && (
                                    <Typography.Text type="secondary">（第{doc.row_index}行）</Typography.Text>
                                  )}
                                </Typography.Text>
                                <Typography.Text type="secondary" style={{ marginLeft: '12px' }}>
                                  相似度：{((doc.score || 0) * 100).toFixed(1)}%
                                </Typography.Text>
                              </div>
                              <Typography.Text>{doc.snippet || doc.content}</Typography.Text>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                
                case 'detailed_analysis':
                  return (
                    <div>
                      {renderTraceInfo(details.data.trace)}
                      <div style={{ marginBottom: '16px' }}>
                        <Typography.Text strong>细化信号数：</Typography.Text>
                        <Typography.Text>{details.data.refinedSignals?.length || 0}</Typography.Text>
                      </div>
                      {(() => {
                        const refineResult = details.data.refineResult || {};
                        const refineConfidence = details.data.refineConfidence;
                        const additionalSignals = details.data.additionalSignals || [];
                        return (
                          <div>
                            <div style={{ marginBottom: '16px', padding: '12px', background: '#f6ffed', border: '1px solid #b7eb8f', borderRadius: '6px' }}>
                              <div style={{ marginBottom: '6px' }}>
                                <Typography.Text strong>细化结论：</Typography.Text>
                                <Typography.Text>{refineResult.conclusion || '暂无'}</Typography.Text>
                              </div>
                              <div>
                                <Typography.Text strong>细化置信度：</Typography.Text>
                                <Typography.Text>
                                  {typeof refineConfidence === 'number' ? `${(refineConfidence * 100).toFixed(1)}%` : '未知'}
                                </Typography.Text>
                              </div>
                            </div>
                            {additionalSignals.length > 0 && (
                              <div style={{ marginBottom: '16px' }}>
                                <Typography.Text strong style={{ display: 'block', marginBottom: '8px' }}>
                                  建议补充的信号（用于下一轮流程控制）：
                                </Typography.Text>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                                  {additionalSignals.map((signal: string) => (
                                    <span key={signal} style={{ padding: '4px 12px', background: '#fff1f0', border: '1px solid #ffa39e', borderRadius: '4px', fontSize: '13px' }}>
                                      {signal}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })()}
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
                      {renderTraceInfo(details.data.trace)}
                      {details.data.summary && (
                        <div style={{ marginBottom: '16px' }}>
                          <Typography.Text strong style={{ display: 'block', marginBottom: '8px' }}>诊断总结：</Typography.Text>
                          <Typography.Text>{toDisplayText(details.data.summary)}</Typography.Text>
                        </div>
                      )}
                      {details.data.findings && details.data.findings.length > 0 && (
                        <div style={{ marginBottom: '16px' }}>
                          <Typography.Text strong style={{ display: 'block', marginBottom: '8px' }}>关键发现：</Typography.Text>
                          <ul style={{ paddingLeft: '20px' }}>
                            {details.data.findings.map((finding: any, index: number) => (
                              <li key={index} style={{ marginBottom: '4px' }}>
                                <Typography.Text>{toDisplayText(finding)}</Typography.Text>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {details.data.risk_assessment && (
                        <div style={{ marginBottom: '16px' }}>
                          <Typography.Text strong>风险评估：</Typography.Text>
                          {(() => {
                            const lvl = getRiskLevel(details.data.risk_assessment);
                            const color =
                              lvl === '高' ? '#ff4d4f' :
                              lvl === '中等' ? '#faad14' : '#52c41a';
                            return (
                          <Typography.Text style={{ 
                            color
                          }}>
                            {toDisplayText(details.data.risk_assessment)}
                          </Typography.Text>
                            );
                          })()}
                        </div>
                      )}
                      {details.data.recommendations && details.data.recommendations.length > 0 && (
                        <div>
                          <Typography.Text strong style={{ display: 'block', marginBottom: '8px' }}>建议措施：</Typography.Text>
                          <ul style={{ paddingLeft: '20px' }}>
                            {details.data.recommendations.map((rec: any, index: number) => (
                              <li key={index} style={{ marginBottom: '4px' }}>
                                <Typography.Text>{toDisplayText(rec)}</Typography.Text>
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
                      {renderTraceInfo(details.data.trace)}
                      {/* LLM 分析总结 */}
                      {llmData.summary && (
                        <div style={{ marginBottom: '24px', padding: '20px', background: '#f0f4f8', borderRadius: '8px', borderLeft: '4px solid #2c5aa0' }}>
                          <Typography.Title level={5} style={{ marginBottom: '12px', color: '#2c5aa0' }}>分析总结</Typography.Title>
                          <Typography.Text style={{ fontSize: '15px', lineHeight: '1.8', color: '#1a1a1a' }}>
                            {toDisplayText(llmData.summary)}
                          </Typography.Text>
                        </div>
                      )}
                      
                      {/* 关键发现 */}
                      {llmData.findings && llmData.findings.length > 0 && (
                        <div style={{ marginBottom: '24px' }}>
                          <Typography.Title level={5} style={{ marginBottom: '12px' }}>关键发现</Typography.Title>
                          <div style={{ display: 'grid', gap: '12px' }}>
                            {llmData.findings.map((finding: any, index: number) => (
                              <div key={index} style={{
                                padding: '12px 16px',
                                background: 'white',
                                border: '1px solid #e8e8e8',
                                borderRadius: '6px',
                                borderLeft: '3px solid #52c41a'
                              }}>
                                <Typography.Text style={{ fontSize: '14px', lineHeight: '1.6' }}>
                                  {toDisplayText(finding)}
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
                          {(() => {
                            const lvl = getRiskLevel(llmData.risk_assessment);
                            const color =
                              lvl === '高' ? '#ff4d4f' :
                              lvl === '中等' ? '#faad14' : '#52c41a';
                            return (
                          <Typography.Text style={{ 
                            fontSize: '16px', 
                            fontWeight: '500',
                            color
                          }}>
                            {toDisplayText(llmData.risk_assessment)}
                          </Typography.Text>
                            );
                          })()}
                        </div>
                      )}
                      
                      {/* 建议措施 */}
                      {llmData.recommendations && llmData.recommendations.length > 0 && (
                        <div style={{ marginBottom: '24px' }}>
                          <Typography.Title level={5} style={{ marginBottom: '12px' }}>建议措施</Typography.Title>
                          <div style={{ display: 'grid', gap: '10px' }}>
                            {llmData.recommendations.map((rec: any, index: number) => (
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
                                  {toDisplayText(rec)}
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
                
                default: {
                  // 动态节点兜底：后端 langgraph 新增/调整节点时，前端无需改代码也能展示必要信息
                  const trace = getTraceEntry(selectedStep);
                  const lastRun =
                    trace && Array.isArray(trace.runs) && trace.runs.length
                      ? trace.runs[trace.runs.length - 1]
                      : null;
                  const meta = lastRun?.metadata || trace?.metadata;
                  const output = lastRun?.output || trace?.output;
                  const err = lastRun?.error || trace?.error;

                  return (
                    <div>
                      {renderTraceInfo(trace)}
                      {(meta || output || err) ? (
                        <div style={{ display: 'grid', gap: '12px' }}>
                          {meta && (
                            <div style={{ background: '#fafafa', border: '1px solid #f0f0f0', borderRadius: '8px', padding: '12px' }}>
                              <Typography.Text strong style={{ display: 'block', marginBottom: '6px' }}>
                                metadata
                              </Typography.Text>
                              <Typography.Text style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px' }}>
                                {toDisplayText(meta)}
                              </Typography.Text>
                            </div>
                          )}
                          {output && (
                            <div style={{ background: '#fafafa', border: '1px solid #f0f0f0', borderRadius: '8px', padding: '12px' }}>
                              <Typography.Text strong style={{ display: 'block', marginBottom: '6px' }}>
                                output
                              </Typography.Text>
                              <Typography.Text style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px' }}>
                                {toDisplayText(output)}
                              </Typography.Text>
                            </div>
                          )}
                          {err && (
                            <div style={{ background: '#fff1f0', border: '1px solid #ffccc7', borderRadius: '8px', padding: '12px' }}>
                              <Typography.Text strong style={{ display: 'block', marginBottom: '6px' }}>
                                error
                              </Typography.Text>
                              <Typography.Text style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px' }}>
                                {toDisplayText(err)}
                              </Typography.Text>
                            </div>
                          )}
                        </div>
                      ) : (
                        <Empty description="暂无可展示的节点信息" />
                      )}
                    </div>
                  );
                }
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
                  const defaults = pickDefaultSignals(availableSignals);
                  setSelectedSignals(defaults);
                }}
              >
                快速选择（关键5信号）
              </Button>
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

      {/* 执行过程时间线：展示后端 langgraph/ReAct 的动态过程（不依赖写死节点） */}
      {(status === 'analyzing' || status === 'completed' || status === 'failed' || status === 'error') && renderExecutionTimeline()}

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
  return <RagCenter />;
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
