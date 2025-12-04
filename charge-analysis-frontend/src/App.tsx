import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link } from 'react-router-dom';
import { ConfigProvider, Layout as AntLayout, Menu, Button, message } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { UserOutlined, FileTextOutlined, DatabaseOutlined, LogoutOutlined } from '@ant-design/icons';
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
      <Header style={{ background: '#001529', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px' }}>
        <div style={{ color: 'white', fontSize: '18px', fontWeight: 'bold' }}>充电分析系统</div>
        <div style={{ color: 'white' }}>
          <UserOutlined style={{ marginRight: '8px' }} />
          {user?.username || user?.email}
          <Button type="link" onClick={() => logout()} style={{ color: 'white', marginLeft: '16px' }}>
            <LogoutOutlined /> 退出
          </Button>
        </div>
      </Header>
      <AntLayout>
        <Sider width={200} style={{ background: '#fff' }}>
          <Menu
            mode="inline"
            selectedKeys={[selectedKey]}
            onClick={({ key }) => setSelectedKey(key)}
            style={{ height: '100%', borderRight: 0 }}
            items={[
              { key: 'home', icon: <FileTextOutlined />, label: '首页' },
              { key: 'charging', icon: <FileTextOutlined />, label: '充电分析' },
              { key: 'rag', icon: <DatabaseOutlined />, label: 'RAG管理' },
              { key: 'training', icon: <DatabaseOutlined />, label: '训练管理' },
            ]}
          />
        </Sider>
        <Content style={{ padding: '24px', background: '#fff', minHeight: 280 }}>
          {selectedKey === 'home' && <HomePage />}
          {selectedKey === 'charging' && <ChargingPage />}
          {selectedKey === 'rag' && <RAGPage />}
          {selectedKey === 'training' && <TrainingPage />}
        </Content>
      </AntLayout>
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
    <div>
      <h1>欢迎使用充电分析系统</h1>
      <p style={{ marginTop: '16px', fontSize: '16px', color: '#666' }}>
        智能充电数据分析平台，提供专业的数据处理和AI辅助决策服务
      </p>

      {/* 统计卡片 */}
      <div style={{ 
        marginTop: '32px', 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', 
        gap: '16px' 
      }}>
        <div style={{ 
          padding: '24px', 
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
          borderRadius: '12px',
          color: 'white',
          boxShadow: '0 4px 12px rgba(102, 126, 234, 0.4)'
        }}>
          <div style={{ fontSize: '14px', opacity: 0.9 }}>总分析次数</div>
          <div style={{ fontSize: '36px', fontWeight: 'bold', marginTop: '8px' }}>{stats.totalAnalyses}</div>
          <div style={{ fontSize: '12px', marginTop: '8px', opacity: 0.8 }}>本月增长 23%</div>
        </div>

        <div style={{ 
          padding: '24px', 
          background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', 
          borderRadius: '12px',
          color: 'white',
          boxShadow: '0 4px 12px rgba(245, 87, 108, 0.4)'
        }}>
          <div style={{ fontSize: '14px', opacity: 0.9 }}>已完成分析</div>
          <div style={{ fontSize: '36px', fontWeight: 'bold', marginTop: '8px' }}>{stats.completedAnalyses}</div>
          <div style={{ fontSize: '12px', marginTop: '8px', opacity: 0.8 }}>成功率 91%</div>
        </div>

        <div style={{ 
          padding: '24px', 
          background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', 
          borderRadius: '12px',
          color: 'white',
          boxShadow: '0 4px 12px rgba(79, 172, 254, 0.4)'
        }}>
          <div style={{ fontSize: '14px', opacity: 0.9 }}>活跃用户</div>
          <div style={{ fontSize: '36px', fontWeight: 'bold', marginTop: '8px' }}>{stats.activeUsers}</div>
          <div style={{ fontSize: '12px', marginTop: '8px', opacity: 0.8 }}>本周在线</div>
        </div>

        <div style={{ 
          padding: '24px', 
          background: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)', 
          borderRadius: '12px',
          color: 'white',
          boxShadow: '0 4px 12px rgba(67, 233, 123, 0.4)'
        }}>
          <div style={{ fontSize: '14px', opacity: 0.9 }}>知识库文档</div>
          <div style={{ fontSize: '36px', fontWeight: 'bold', marginTop: '8px' }}>{stats.knowledgeDocuments}</div>
          <div style={{ fontSize: '12px', marginTop: '8px', opacity: 0.8 }}>最近更新</div>
        </div>
      </div>

      {/* 快捷操作 */}
      <div style={{ marginTop: '32px' }}>
        <h3 style={{ marginBottom: '16px', fontSize: '18px' }}>快捷操作</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
          <Button 
            type="primary" 
            size="large" 
            style={{ height: '80px', fontSize: '16px' }}
            onClick={() => {}}
          >
            新建充电分析
          </Button>
          <Button 
            size="large" 
            style={{ height: '80px', fontSize: '16px' }}
            onClick={() => {}}
          >
            查询知识库
          </Button>
          <Button 
            size="large" 
            style={{ height: '80px', fontSize: '16px' }}
            onClick={() => {}}
          >
            创建训练任务
          </Button>
        </div>
      </div>

      {/* 最近活动 */}
      <div style={{ marginTop: '32px' }}>
        <h3 style={{ marginBottom: '16px', fontSize: '18px' }}>最近活动</h3>
        <div style={{ background: '#fafafa', borderRadius: '8px', padding: '16px' }}>
          {recentActivities.map(activity => (
            <div 
              key={activity.id}
              style={{ 
                padding: '12px 16px', 
                background: 'white',
                borderRadius: '8px',
                marginBottom: '8px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                boxShadow: '0 1px 4px rgba(0,0,0,0.05)'
              }}
            >
              <div>
                <span style={{ 
                  padding: '2px 8px', 
                  background: '#e6f7ff',
                  color: '#1890ff',
                  borderRadius: '4px',
                  fontSize: '12px',
                  marginRight: '8px'
                }}>
                  {activity.type}
                </span>
                <strong>{activity.user}</strong>
                <span style={{ color: '#666', marginLeft: '8px' }}>{activity.action}</span>
              </div>
              <span style={{ fontSize: '12px', color: '#999' }}>{activity.time}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 功能介绍 */}
      <div style={{ marginTop: '32px', padding: '24px', background: '#f6f8fa', borderRadius: '8px' }}>
        <h3 style={{ marginBottom: '16px', fontSize: '18px' }}>系统功能</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '16px' }}>
          <div>
            <h4 style={{ color: '#1890ff', marginBottom: '8px' }}>充电数据分析</h4>
            <p style={{ fontSize: '14px', color: '#666', lineHeight: '1.6' }}>
              上传BLF、CSV或Excel格式的充电数据，系统自动进行异常检测、趋势分析和风险评估，生成专业的诊断报告。
            </p>
          </div>
          <div>
            <h4 style={{ color: '#52c41a', marginBottom: '8px' }}>RAG知识库</h4>
            <p style={{ fontSize: '14px', color: '#666', lineHeight: '1.6' }}>
              管理技术文档和知识库，支持智能检索和语义查询，快速获取相关技术信息和解决方案。
            </p>
          </div>
          <div>
            <h4 style={{ color: '#722ed1', marginBottom: '8px' }}>训练管理</h4>
            <p style={{ fontSize: '14px', color: '#666', lineHeight: '1.6' }}>
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
  const { user, token } = useAuthStore();
  const fileInputRef = React.useRef<HTMLInputElement>(null);

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
    }
  };

  const handleUpload = async () => {
    if (!file || !user || !token) return;

    setStatus('uploading');
    setResults([]);
    try {
      const { chargingService } = await import('./services/chargingService');
      
      message.loading('上传中...', 0);
      const analysis = await chargingService.uploadFile(file, token, file.name);
      message.destroy();
      message.success('文件上传成功');
      
      setAnalysisId(analysis.id);
      setStatus('uploaded');
      
      // Start analysis
      message.loading('开始分析...', 0);
      await chargingService.startAnalysis(analysis.id, token);
      message.destroy();
      message.success('分析已开始');
      
      setStatus('analyzing');
      
      // Poll for results
      pollAnalysisStatus(analysis.id);
      
    } catch (error: any) {
      message.destroy();
      message.error(error.message || '上传失败');
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
        
        if (analysis.status === 'completed') {
          clearInterval(interval);
          setStatus('completed');
          message.success('分析完成');
          
          // Load results and display them
          const analysisResults = await chargingService.getAnalysisResults(id, authToken);
          setResults(analysisResults);
          
          // Refresh history
          loadAnalysisHistory();
        } else if (analysis.status === 'failed') {
          clearInterval(interval);
          setStatus('failed');
          message.error('分析失败：' + (analysis.errorMessage || '未知错误'));
        }
      } catch (error) {
        console.error('轮询状态失败：', error);
      }
    }, 3000);
    
    // Stop after 5 minutes
    setTimeout(() => clearInterval(interval), 300000);
  };

  const loadHistoryResults = async (id: number) => {
    if (!token) {
      message.error('请先登录');
      return;
    }
    try {
      const { chargingService } = await import('./services/chargingService');
      const analysisResults = await chargingService.getAnalysisResults(id, token);
      setResults(analysisResults);
      setAnalysisId(id);
      setStatus('completed');
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
    <div>
      <h1>充电数据分析</h1>
      
      {/* Upload Section */}
      <div style={{ marginTop: '24px', padding: '24px', border: '1px dashed #d9d9d9', borderRadius: '8px', textAlign: 'center' }}>
        <input
          ref={fileInputRef}
          type="file"
          accept=".blf,.csv,.xlsx"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <Button size="large" onClick={() => fileInputRef.current?.click()}>
          选择文件
        </Button>
        {file && (
          <div style={{ marginTop: '16px' }}>
            <p>已选择文件：{file.name}</p>
            <Button type="primary" size="large" onClick={handleUpload} style={{ marginTop: '12px' }}>
              开始分析
            </Button>
          </div>
        )}
      </div>

      {/* Status Section */}
      {status !== 'idle' && (
        <div style={{ marginTop: '24px', padding: '16px', background: '#f5f5f5', borderRadius: '8px' }}>
          <p><strong>状态：</strong>
            <span style={{ 
              color: status === 'completed' ? '#52c41a' : 
                     status === 'failed' || status === 'error' ? '#ff4d4f' : 
                     status === 'analyzing' ? '#1890ff' : '#666'
            }}>
              {status === 'idle' ? '空闲' :
               status === 'uploading' ? '上传中...' :
               status === 'uploaded' ? '已上传' :
               status === 'analyzing' ? '分析中...' :
               status === 'completed' ? '已完成' :
               status === 'failed' ? '失败' :
               status === 'error' ? '错误' : status}
            </span>
          </p>
          {analysisId && <p><strong>分析ID：</strong>{analysisId}</p>}
        </div>
      )}

      {/* Results Section */}
      {results.length > 0 && (
        <div style={{ marginTop: '24px' }}>
          <h2 style={{ marginBottom: '16px', borderBottom: '2px solid #1890ff', paddingBottom: '8px' }}>
            分析结果
          </h2>
          <div style={{ display: 'grid', gap: '16px' }}>
            {results.map((result, index) => (
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
                  <h3 style={{ margin: 0, color: getResultColor(result.resultType) }}>
                    {result.title}
                  </h3>
                  <span style={{ 
                    marginLeft: 'auto', 
                    padding: '2px 8px', 
                    background: '#f0f0f0', 
                    borderRadius: '4px',
                    fontSize: '12px',
                    color: '#666'
                  }}>
                    {result.resultType}
                  </span>
                </div>
                <div style={{ 
                  color: '#333', 
                  lineHeight: '1.8',
                  whiteSpace: 'pre-wrap'
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

      {/* History Section */}
      {analysisHistory.length > 0 && (
        <div style={{ marginTop: '32px' }}>
          <h2 style={{ marginBottom: '16px', borderBottom: '1px solid #d9d9d9', paddingBottom: '8px' }}>
            历史分析记录
          </h2>
          <div style={{ display: 'grid', gap: '12px' }}>
            {analysisHistory.map((item) => (
              <div 
                key={item.id}
                style={{ 
                  padding: '12px 16px', 
                  background: '#fafafa', 
                  borderRadius: '4px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  cursor: item.status === 'completed' ? 'pointer' : 'default'
                }}
                onClick={() => item.status === 'completed' && loadHistoryResults(item.id)}
              >
                <div>
                  <strong>{item.name}</strong>
                  <div style={{ fontSize: '12px', color: '#999', marginTop: '4px' }}>
                    {new Date(item.createdAt).toLocaleString()}
                  </div>
                </div>
                <span style={{ 
                  padding: '2px 8px', 
                  borderRadius: '4px',
                  fontSize: '12px',
                  background: item.status === 'completed' ? '#f6ffed' : 
                             item.status === 'failed' ? '#fff2f0' : '#e6f7ff',
                  color: item.status === 'completed' ? '#52c41a' : 
                         item.status === 'failed' ? '#ff4d4f' : '#1890ff'
                }}>
                  {item.status === 'completed' ? '已完成' :
                   item.status === 'failed' ? '失败' :
                   item.status === 'processing' ? '处理中' : '待处理'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
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
          message.error('训练失败：' + task.error_message);
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
