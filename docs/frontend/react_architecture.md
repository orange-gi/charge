# 前端 React 架构设计

## 1. 项目概述

本系统前端采用 React 18 + TypeScript 构建，提供现代化的单页应用体验，包含用户认证、充电分析、RAG管理、训练管理和日志管理等核心功能。

### 1.1 核心特性
- **组件化架构**: 高度可复用的组件设计
- **类型安全**: 完整的 TypeScript 类型定义
- **状态管理**: 基于 Zustand 的轻量级状态管理
- **实时通信**: WebSocket 支持实时进度更新
- **响应式设计**: 适配各种设备尺寸
- **权限控制**: 基于角色的访问控制

## 2. 项目结构

### 2.1 目录结构
```
charge-analysis-frontend/
├── public/
│   ├── index.html
│   └── favicon.ico
├── src/
│   ├── components/           # 可复用组件
│   │   ├── common/          # 通用组件
│   │   ├── auth/            # 认证组件
│   │   ├── charging/        # 充电分析组件
│   │   ├── rag/             # RAG管理组件
│   │   ├── training/        # 训练管理组件
│   │   └── layout/          # 布局组件
│   ├── pages/               # 页面组件
│   │   ├── auth/            # 认证页面
│   │   ├── home/            # 首页
│   │   ├── charging/        # 充电分析页面
│   │   ├── rag/             # RAG管理页面
│   │   ├── training/        # 训练管理页面
│   │   └── logs/            # 日志管理页面
│   ├── hooks/               # 自定义Hook
│   ├── stores/              # 状态管理
│   ├── services/            # API服务
│   ├── utils/               # 工具函数
│   ├── types/               # 类型定义
│   ├── constants/           # 常量定义
│   ├── styles/              # 样式文件
│   └── App.tsx              # 主应用组件
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

### 2.2 核心文件说明

#### 2.2.1 App.tsx
```typescript
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { useAuthStore } from './stores/authStore';
import { Layout } from './components/layout/Layout';
import { LoadingScreen } from './components/common/LoadingScreen';
import { ErrorBoundary } from './components/common/ErrorBoundary';

// 页面组件
import { LoginPage } from './pages/auth/LoginPage';
import { RegisterPage } from './pages/auth/RegisterPage';
import { HomePage } from './pages/home/HomePage';
import { ChargingAnalysisPage } from './pages/charging/ChargingAnalysisPage';
import { RAGManagementPage } from './pages/rag/RAGManagementPage';
import { TrainingManagementPage } from './pages/training/TrainingManagementPage';
import { LogManagementPage } from './pages/logs/LogManagementPage';

import 'antd/dist/reset.css';
import './styles/globals.css';

const App: React.FC = () => {
  const { isLoading, isAuthenticated, initialize } = useAuthStore();

  React.useEffect(() => {
    initialize();
  }, [initialize]);

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <ErrorBoundary>
      <ConfigProvider locale={zhCN}>
        <Router>
          <Routes>
            {/* 公开路由 */}
            <Route path="/login" element={
              isAuthenticated ? <Navigate to="/" replace /> : <LoginPage />
            } />
            <Route path="/register" element={
              isAuthenticated ? <Navigate to="/" replace /> : <RegisterPage />
            } />
            
            {/* 受保护的路由 */}
            <Route path="/" element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }>
              <Route index element={<HomePage />} />
              <Route path="charging" element={<ChargingAnalysisPage />} />
              <Route 
                path="rag" 
                element={
                  <AdminRoute>
                    <RAGManagementPage />
                  </AdminRoute>
                } 
              />
              <Route 
                path="training" 
                element={
                  <AdminRoute>
                    <TrainingManagementPage />
                  </AdminRoute>
                } 
              />
              <Route 
                path="logs" 
                element={
                  <AdminRoute>
                    <LogManagementPage />
                  </AdminRoute>
                } 
              />
            </Route>
            
            {/* 404 页面 */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Router>
      </ConfigProvider>
    </ErrorBoundary>
  );
};

// 路由保护组件
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated } = useAuthStore();
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
};

// 管理员路由保护组件
const AdminRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user } = useAuthStore();
  
  if (!user || user.role !== 'admin') {
    return <Navigate to="/" replace />;
  }
  
  return <>{children}</>;
};

export default App;
```

## 3. 组件架构

### 3.1 组件分类

#### 3.1.1 布局组件
```typescript
// src/components/layout/Layout.tsx
import React from 'react';
import { Outlet } from 'react-router-dom';
import { Layout as AntLayout } from 'antd';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { useLayoutStore } from '../../stores/layoutStore';

const { Content } = AntLayout;

export const Layout: React.FC = () => {
  const { isCollapsed } = useLayoutStore();

  return (
    <AntLayout style={{ minHeight: '100vh' }}>
      <Sidebar />
      <AntLayout 
        style={{ 
          marginLeft: isCollapsed ? 80 : 200,
          transition: 'margin-left 0.2s'
        }}
      >
        <Header />
        <Content 
          style={{ 
            margin: '24px 16px', 
            padding: 24, 
            minHeight: 280,
            background: '#fff',
            borderRadius: '8px'
          }}
        >
          <Outlet />
        </Content>
      </AntLayout>
    </AntLayout>
  );
};
```

#### 3.1.2 通用组件
```typescript
// src/components/common/LoadingScreen.tsx
import React from 'react';
import { Spin } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';

const loadingIcon = <LoadingOutlined style={{ fontSize: 24 }} spin />;

export const LoadingScreen: React.FC = () => {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      background: '#f5f5f5'
    }}>
      <Spin indicator={loadingIcon} size="large" />
    </div>
  );
};

// src/components/common/ErrorBoundary.tsx
import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Result, Button } from 'antd';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <Result
          status="500"
          title="系统错误"
          subTitle="抱歉，系统出现了错误。请刷新页面重试。"
          extra={
            <Button 
              type="primary" 
              onClick={() => window.location.reload()}
            >
              刷新页面
            </Button>
          }
        />
      );
    }

    return this.props.children;
  }
}
```

### 3.2 页面组件

#### 3.2.1 充电分析页面
```typescript
// src/pages/charging/ChargingAnalysisPage.tsx
import React, { useState } from 'react';
import { Card, Row, Col, Button, message } from 'antd';
import { UploadOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { FileUploader } from '../../components/charging/FileUploader';
import { SignalChart } from '../../components/charging/SignalChart';
import { AnalysisResults } from '../../components/charging/AnalysisResults';
import { useChargingStore } from '../../stores/chargingStore';
import { useWebSocket } from '../../hooks/useWebSocket';

export const ChargingAnalysisPage: React.FC = () => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const { 
    currentAnalysis, 
    analysisProgress, 
    analysisResults,
    startAnalysis,
    createAnalysis 
  } = useChargingStore();

  const { connect, disconnect } = useWebSocket();

  React.useEffect(() => {
    if (currentAnalysis) {
      connect(`/ws/analysis/${currentAnalysis.id}`);
    }
    
    return () => {
      disconnect();
    };
  }, [currentAnalysis, connect, disconnect]);

  const handleFileUpload = (file: File) => {
    setUploadedFile(file);
    message.success(`文件 ${file.name} 上传成功`);
  };

  const handleStartAnalysis = async () => {
    if (!uploadedFile) {
      message.error('请先上传文件');
      return;
    }

    try {
      const analysis = await createAnalysis({
        name: uploadedFile.name,
        file: uploadedFile
      });
      
      await startAnalysis(analysis.id);
      message.success('分析任务已启动');
    } catch (error) {
      message.error('启动分析失败');
    }
  };

  return (
    <div className="charging-analysis-page">
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="充电数据分析">
            <Row gutter={[16, 16]}>
              <Col xs={24} lg={12}>
                <FileUploader 
                  onUpload={handleFileUpload}
                  uploadedFile={uploadedFile}
                />
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={handleStartAnalysis}
                  disabled={!uploadedFile}
                  size="large"
                  style={{ marginTop: 16 }}
                >
                  开始分析
                </Button>
              </Col>
              <Col xs={24} lg={12}>
                {analysisProgress && (
                  <div className="progress-section">
                    <h3>分析进度</h3>
                    <div className="progress-info">
                      <div className="progress-bar">
                        <div 
                          className="progress-fill"
                          style={{ width: `${analysisProgress.progress}%` }}
                        />
                      </div>
                      <p>{analysisProgress.currentStep}</p>
                    </div>
                  </div>
                )}
              </Col>
            </Row>
          </Card>
        </Col>
        
        {analysisResults && (
          <>
            <Col span={24}>
              <SignalChart 
                data={analysisResults.visualizations}
                signals={analysisResults.signals}
              />
            </Col>
            <Col span={24}>
              <AnalysisResults results={analysisResults.results} />
            </Col>
          </>
        )}
      </Row>
    </div>
  );
};
```

## 4. 状态管理架构

### 4.1 Zustand Store 设计

#### 4.1.1 认证状态管理
```typescript
// src/stores/authStore.ts
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { authService } from '../services/authService';
import { User, LoginCredentials, RegisterData } from '../types/auth';

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  
  // Actions
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<void>;
  initialize: () => Promise<void>;
  updateProfile: (data: Partial<User>) => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isLoading: false,
      isAuthenticated: false,

      login: async (credentials: LoginCredentials) => {
        set({ isLoading: true });
        try {
          const response = await authService.login(credentials);
          set({
            user: response.user,
            token: response.access_token,
            isAuthenticated: true,
            isLoading: false
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      register: async (data: RegisterData) => {
        set({ isLoading: true });
        try {
          await authService.register(data);
          set({ isLoading: false });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      logout: async () => {
        try {
          await authService.logout();
        } catch (error) {
          console.error('Logout error:', error);
        } finally {
          set({
            user: null,
            token: null,
            isAuthenticated: false
          });
        }
      },

      refreshToken: async () => {
        const { token } = get();
        if (!token) return;

        try {
          const response = await authService.refreshToken();
          set({ token: response.access_token });
        } catch (error) {
          get().logout();
        }
      },

      initialize: async () => {
        const { token } = get();
        if (!token) return;

        try {
          const user = await authService.getCurrentUser();
          set({ 
            user, 
            isAuthenticated: true 
          });
        } catch (error) {
          get().logout();
        }
      },

      updateProfile: async (data: Partial<User>) => {
        try {
          const updatedUser = await authService.updateProfile(data);
          set({ user: updatedUser });
        } catch (error) {
          throw error;
        }
      }
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated
      })
    }
  )
);
```

#### 4.1.2 充电分析状态管理
```typescript
// src/stores/chargingStore.ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { chargingService } from '../services/chargingService';
import { ChargingAnalysis, AnalysisProgress, AnalysisResults } from '../types/charging';

interface ChargingState {
  analyses: ChargingAnalysis[];
  currentAnalysis: ChargingAnalysis | null;
  analysisProgress: AnalysisProgress | null;
  analysisResults: AnalysisResults | null;
  isLoading: boolean;
  
  // Actions
  loadAnalyses: () => Promise<void>;
  createAnalysis: (data: { name: string; file: File }) => Promise<ChargingAnalysis>;
  startAnalysis: (analysisId: string) => Promise<void>;
  updateProgress: (progress: AnalysisProgress) => void;
  loadResults: (analysisId: string) => Promise<void>;
  deleteAnalysis: (analysisId: string) => Promise<void>;
  clearCurrentAnalysis: () => void;
}

export const useChargingStore = create<ChargingState>()(
  devtools(
    (set, get) => ({
      analyses: [],
      currentAnalysis: null,
      analysisProgress: null,
      analysisResults: null,
      isLoading: false,

      loadAnalyses: async () => {
        set({ isLoading: true });
        try {
          const analyses = await chargingService.getAnalyses();
          set({ analyses, isLoading: false });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      createAnalysis: async (data: { name: string; file: File }) => {
        try {
          const analysis = await chargingService.createAnalysis(data);
          set(state => ({
            analyses: [analysis, ...state.analyses],
            currentAnalysis: analysis
          }));
          return analysis;
        } catch (error) {
          throw error;
        }
      },

      startAnalysis: async (analysisId: string) => {
        try {
          await chargingService.startAnalysis(analysisId);
          set(state => ({
            currentAnalysis: state.analyses.find(a => a.id === analysisId) || null
          }));
        } catch (error) {
          throw error;
        }
      },

      updateProgress: (progress: AnalysisProgress) => {
        set({ analysisProgress: progress });
      },

      loadResults: async (analysisId: string) => {
        try {
          const results = await chargingService.getResults(analysisId);
          set({ analysisResults: results });
        } catch (error) {
          throw error;
        }
      },

      deleteAnalysis: async (analysisId: string) => {
        try {
          await chargingService.deleteAnalysis(analysisId);
          set(state => ({
            analyses: state.analyses.filter(a => a.id !== analysisId),
            currentAnalysis: state.currentAnalysis?.id === analysisId ? null : state.currentAnalysis
          }));
        } catch (error) {
          throw error;
        }
      },

      clearCurrentAnalysis: () => {
        set({
          currentAnalysis: null,
          analysisProgress: null,
          analysisResults: null
        });
      }
    })
  )
);
```

### 4.2 自定义 Hooks

#### 4.2.1 WebSocket Hook
```typescript
// src/hooks/useWebSocket.ts
import { useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { useChargingStore } from '../stores/chargingStore';
import { useTrainingStore } from '../stores/trainingStore';
import { message } from 'antd';

export const useWebSocket = () => {
  const socketRef = useRef<Socket | null>(null);
  const { updateProgress, loadResults } = useChargingStore();
  const { updateTrainingProgress } = useTrainingStore();

  const connect = useCallback((endpoint: string) => {
    if (socketRef.current) {
      disconnect();
    }

    socketRef.current = io(endpoint, {
      transports: ['websocket'],
      upgrade: true
    });

    socketRef.current.on('connect', () => {
      console.log('WebSocket connected:', endpoint);
    });

    socketRef.current.on('disconnect', () => {
      console.log('WebSocket disconnected');
    });

    socketRef.current.on('analysis_progress', (data: any) => {
      updateProgress(data);
    });

    socketRef.current.on('training_progress', (data: any) => {
      updateTrainingProgress(data);
    });

    socketRef.current.on('analysis_completed', async (data: any) => {
      message.success('分析完成');
      await loadResults(data.analysis_id);
    });

    socketRef.current.on('error', (error: any) => {
      message.error(`WebSocket错误: ${error.message}`);
    });

    return socketRef.current;
  }, [updateProgress, updateTrainingProgress, loadResults]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
  }, []);

  return { connect, disconnect };
};
```

#### 4.2.2 文件上传 Hook
```typescript
// src/hooks/useFileUpload.ts
import { useState, useCallback } from 'react';
import { uploadService } from '../services/uploadService';
import { message } from 'antd';

interface UseFileUploadOptions {
  maxSize?: number;
  acceptedTypes?: string[];
  onUpload?: (file: File, response: any) => void;
  onError?: (error: Error) => void;
}

export const useFileUpload = (options: UseFileUploadOptions = {}) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const {
    maxSize = 100 * 1024 * 1024, // 100MB
    acceptedTypes = ['.blf', '.csv', '.xlsx'],
    onUpload,
    onError
  } = options;

  const validateFile = useCallback((file: File): boolean => {
    // 检查文件大小
    if (file.size > maxSize) {
      message.error(`文件大小不能超过 ${maxSize / (1024 * 1024)}MB`);
      return false;
    }

    // 检查文件类型
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!acceptedTypes.includes(fileExtension)) {
      message.error(`不支持的文件类型，请上传 ${acceptedTypes.join(', ')} 格式的文件`);
      return false;
    }

    return true;
  }, [maxSize, acceptedTypes]);

  const uploadFile = useCallback(async (file: File) => {
    if (!validateFile(file)) {
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const response = await uploadService.uploadFile(file, (progress) => {
        setUploadProgress(progress);
      });

      message.success('文件上传成功');
      onUpload?.(file, response);
      return response;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '上传失败';
      message.error(errorMessage);
      onError?.(error as Error);
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  }, [validateFile, onUpload, onError]);

  return {
    uploadFile,
    isUploading,
    uploadProgress,
    validateFile
  };
};
```

## 5. UI/UX 设计规范

### 5.1 设计系统

#### 5.1.1 颜色规范
```typescript
// src/styles/colors.ts
export const colors = {
  // 主色调
  primary: {
    50: '#e6f7ff',
    100: '#bae7ff',
    500: '#1890ff',
    600: '#096dd9',
    700: '#0050b3'
  },
  
  // 功能色
  success: '#52c41a',
  warning: '#faad14',
  error: '#ff4d4f',
  info: '#1890ff',
  
  // 中性色
  gray: {
    50: '#fafafa',
    100: '#f5f5f5',
    200: '#f0f0f0',
    500: '#8c8c8c',
    700: '#595959',
    900: '#262626'
  },
  
  // 充电相关色彩
  charging: {
    normal: '#52c41a',
    warning: '#faad14',
    error: '#ff4d4f',
    processing: '#1890ff'
  }
};
```

#### 5.1.2 间距规范
```typescript
// src/styles/spacing.ts
export const spacing = {
  xs: '4px',
  sm: '8px',
  md: '16px',
  lg: '24px',
  xl: '32px',
  xxl: '48px'
};
```

#### 5.1.3 圆角规范
```typescript
// src/styles/borderRadius.ts
export const borderRadius = {
  sm: '4px',
  md: '8px',
  lg: '12px',
  xl: '16px',
  round: '50%'
};
```

### 5.2 响应式设计

#### 5.2.1 断点定义
```typescript
// src/styles/breakpoints.ts
export const breakpoints = {
  xs: '480px',
  sm: '576px',
  md: '768px',
  lg: '992px',
  xl: '1200px',
  xxl: '1600px'
};

export const mediaQueries = {
  mobile: `@media (max-width: ${breakpoints.sm})`,
  tablet: `@media (max-width: ${breakpoints.lg})`,
  desktop: `@media (min-width: ${breakpoints.lg})`,
  largeDesktop: `@media (min-width: ${breakpoints.xl})`
};
```

#### 5.2.2 组件响应式布局
```typescript
// src/components/common/ResponsiveGrid.tsx
import React from 'react';
import { Row, Col } from 'antd';

interface ResponsiveGridProps {
  children: React.ReactNode;
  gutter?: [number, number];
  className?: string;
}

export const ResponsiveGrid: React.FC<ResponsiveGridProps> = ({ 
  children, 
  gutter = [16, 16],
  className 
}) => {
  return (
    <Row gutter={gutter} className={className}>
      {React.Children.map(children, (child, index) => (
        <Col
          key={index}
          xs={24}
          sm={12}
          md={8}
          lg={6}
          xl={6}
          xxl={4}
        >
          {child}
        </Col>
      ))}
    </Row>
  );
};
```

### 5.3 主题定制

#### 5.3.1 Ant Design 主题配置
```typescript
// src/styles/antdTheme.ts
import { ThemeConfig } from 'antd';

export const themeConfig: ThemeConfig = {
  token: {
    colorPrimary: '#1890ff',
    colorSuccess: '#52c41a',
    colorWarning: '#faad14',
    colorError: '#ff4d4f',
    colorInfo: '#1890ff',
    borderRadius: 8,
    fontSize: 14,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial'
  },
  components: {
    Layout: {
      headerBg: '#fff',
      headerHeight: 64,
      siderBg: '#fff'
    },
    Menu: {
      itemBg: 'transparent',
      itemSelectedBg: '#e6f7ff',
      itemHoverBg: '#f5f5f5'
    },
    Card: {
      borderRadius: 12,
      headerBg: '#fafafa'
    },
    Button: {
      borderRadius: 6
    }
  }
};
```

## 6. 权限控制实现

### 6.1 权限钩子

```typescript
// src/hooks/usePermission.ts
import { useMemo } from 'react';
import { useAuthStore } from '../stores/authStore';

interface Permission {
  resource: string;
  action: string;
}

const rolePermissions: Record<string, Permission[]> = {
  user: [
    { resource: 'charging', action: 'read' },
    { resource: 'charging', action: 'create' },
    { resource: 'charging', action: 'update' }
  ],
  admin: [
    { resource: 'charging', action: '*' },
    { resource: 'rag', action: '*' },
    { resource: 'training', action: '*' },
    { resource: 'logs', action: '*' },
    { resource: 'users', action: '*' }
  ]
};

export const usePermission = () => {
  const { user } = useAuthStore();

  const permissions = useMemo(() => {
    if (!user) return [];
    return rolePermissions[user.role] || [];
  }, [user]);

  const hasPermission = (resource: string, action: string): boolean => {
    if (!user) return false;
    
    const userPermissions = rolePermissions[user.role] || [];
    return userPermissions.some(p => 
      (p.resource === resource && (p.action === action || p.action === '*')) ||
      (p.resource === '*' && p.action === '*')
    );
  };

  const hasAnyPermission = (requiredPermissions: Permission[]): boolean => {
    return requiredPermissions.some(permission => 
      hasPermission(permission.resource, permission.action)
    );
  };

  const hasAllPermissions = (requiredPermissions: Permission[]): boolean => {
    return requiredPermissions.every(permission => 
      hasPermission(permission.resource, permission.action)
    );
  };

  return {
    permissions,
    hasPermission,
    hasAnyPermission,
    hasAllPermissions,
    isAdmin: user?.role === 'admin'
  };
};
```

### 6.2 权限组件

```typescript
// src/components/common/PermissionGuard.tsx
import React from 'react';
import { usePermission } from '../../hooks/usePermission';

interface PermissionGuardProps {
  resource: string;
  action: string;
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export const PermissionGuard: React.FC<PermissionGuardProps> = ({
  resource,
  action,
  children,
  fallback = null
}) => {
  const { hasPermission } = usePermission();

  if (!hasPermission(resource, action)) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
};

// 高阶组件版本
export const withPermission = (
  Component: React.ComponentType<any>,
  resource: string,
  action: string
) => {
  return (props: any) => {
    const { hasPermission } = usePermission();

    if (!hasPermission(resource, action)) {
      return null;
    }

    return <Component {...props} />;
  };
};
```

## 7. 实时通信方案

### 7.1 WebSocket 管理器

```typescript
// src/services/websocketService.ts
import { io, Socket } from 'socket.io-client';

class WebSocketManager {
  private sockets: Map<string, Socket> = new Map();
  private eventListeners: Map<string, Set<Function>> = new Map();

  connect(endpoint: string, token: string): Socket {
    const socket = io(endpoint, {
      auth: {
        token
      },
      transports: ['websocket'],
      upgrade: true
    });

    socket.on('connect', () => {
      console.log(`WebSocket connected to ${endpoint}`);
    });

    socket.on('disconnect', () => {
      console.log(`WebSocket disconnected from ${endpoint}`);
    });

    socket.on('error', (error) => {
      console.error(`WebSocket error on ${endpoint}:`, error);
    });

    this.sockets.set(endpoint, socket);
    return socket;
  }

  disconnect(endpoint: string) {
    const socket = this.sockets.get(endpoint);
    if (socket) {
      socket.disconnect();
      this.sockets.delete(endpoint);
    }
  }

  disconnectAll() {
    this.sockets.forEach(socket => socket.disconnect());
    this.sockets.clear();
  }

  subscribe(endpoint: string, event: string, callback: Function) {
    const socket = this.sockets.get(endpoint);
    if (socket) {
      socket.on(event, callback as any);
      
      // 添加到事件监听器集合
      if (!this.eventListeners.has(endpoint)) {
        this.eventListeners.set(endpoint, new Set());
      }
      this.eventListeners.get(endpoint)?.add(callback);
    }
  }

  unsubscribe(endpoint: string, event: string, callback: Function) {
    const socket = this.sockets.get(endpoint);
    if (socket) {
      socket.off(event, callback as any);
      this.eventListeners.get(endpoint)?.delete(callback);
    }
  }

  emit(endpoint: string, event: string, data: any) {
    const socket = this.sockets.get(endpoint);
    if (socket) {
      socket.emit(event, data);
    }
  }
}

export const websocketManager = new WebSocketManager();
```

### 7.2 进度追踪组件

```typescript
// src/components/common/ProgressTracker.tsx
import React from 'react';
import { Progress, Card, List, Avatar } from 'antd';
import { 
  LoadingOutlined, 
  CheckCircleOutlined, 
  ExclamationCircleOutlined 
} from '@ant-design/icons';

interface ProgressStep {
  id: string;
  title: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  description?: string;
  icon?: React.ReactNode;
}

interface ProgressTrackerProps {
  steps: ProgressStep[];
  currentStep: string;
  overallProgress: number;
}

export const ProgressTracker: React.FC<ProgressTrackerProps> = ({
  steps,
  currentStep,
  overallProgress
}) => {
  const getStepIcon = (step: ProgressStep) => {
    switch (step.status) {
      case 'processing':
        return <LoadingOutlined spin style={{ color: '#1890ff' }} />;
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'error':
        return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <Avatar size="small">{step.id}</Avatar>;
    }
  };

  return (
    <Card title="任务进度" size="small">
      <Progress 
        percent={overallProgress} 
        strokeColor="#1890ff"
        trailColor="#f0f0f0"
      />
      
      <List
        dataSource={steps}
        renderItem={(step) => (
          <List.Item className={`progress-step ${step.id === currentStep ? 'active' : ''}`}>
            <List.Item.Meta
              avatar={getStepIcon(step)}
              title={step.title}
              description={step.description}
            />
            {step.status === 'processing' && (
              <LoadingOutlined spin style={{ color: '#1890ff' }} />
            )}
          </List.Item>
        )}
      />
    </Card>
  );
};
```

## 8. 性能优化

### 8.1 代码分割

```typescript
// src/components/common/LazyComponent.tsx
import React, { Suspense } from 'react';
import { Spin } from 'antd';

export const LazyComponent = <T extends React.ComponentType<any>>(
  loader: () => Promise<{ default: T }>,
  fallback?: React.ReactNode
) => {
  const LazyComponent = React.lazy(loader);
  
  return (props: React.ComponentProps<T>) => (
    <Suspense fallback={fallback || <Spin size="large" />}>
      <LazyComponent {...props} />
    </Suspense>
  );
};

// 使用示例
export const LazyChargingAnalysisPage = LazyComponent(
  () => import('../../pages/charging/ChargingAnalysisPage')
);
```

### 8.2 虚拟化列表

```typescript
// src/components/common/VirtualList.tsx
import React from 'react';
import { FixedSizeList as List } from 'react-window';
import { AutoSizer } from 'react-virtualized-auto-sizer';

interface VirtualListProps<T> {
  items: T[];
  itemHeight: number;
  renderItem: (item: T, index: number) => React.ReactNode;
  className?: string;
}

export function VirtualList<T>({ 
  items, 
  itemHeight, 
  renderItem, 
  className 
}: VirtualListProps<T>) {
  return (
    <AutoSizer>
      {({ height, width }) => (
        <List
          height={height}
          width={width}
          itemCount={items.length}
          itemSize={itemHeight}
          className={className}
        >
          {({ index, style }) => (
            <div style={style}>
              {renderItem(items[index], index)}
            </div>
          )}
        </List>
      )}
    </AutoSizer>
  );
}
```

### 8.3 缓存策略

```typescript
// src/utils/cacheManager.ts
class CacheManager {
  private cache = new Map<string, { data: any; expiry: number }>();
  private readonly defaultTTL = 5 * 60 * 1000; // 5分钟

  set(key: string, data: any, ttl: number = this.defaultTTL) {
    const expiry = Date.now() + ttl;
    this.cache.set(key, { data, expiry });
  }

  get(key: string) {
    const item = this.cache.get(key);
    if (!item) return null;

    if (Date.now() > item.expiry) {
      this.cache.delete(key);
      return null;
    }

    return item.data;
  }

  delete(key: string) {
    this.cache.delete(key);
  }

  clear() {
    this.cache.clear();
  }

  cleanup() {
    const now = Date.now();
    for (const [key, item] of this.cache.entries()) {
      if (now > item.expiry) {
        this.cache.delete(key);
      }
    }
  }
}

export const cacheManager = new CacheManager();

// 定期清理过期缓存
setInterval(() => cacheManager.cleanup(), 60000);
```

## 9. 错误处理

### 9.1 错误边界

```typescript
// src/components/common/ErrorBoundary.tsx (增强版)
import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Result, Button, Card } from 'antd';

interface Props {
  children: ReactNode;
  fallback?: (error: Error, errorInfo: ErrorInfo) => ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Component error:', error, errorInfo);
    
    // 发送错误到监控服务
    this.reportError(error, errorInfo);
    
    this.setState({ errorInfo });
  }

  private reportError = (error: Error, errorInfo: ErrorInfo) => {
    // 这里可以发送到错误监控服务
    console.log('Error reported:', { error, errorInfo });
  };

  private handleReload = () => {
    window.location.reload();
  };

  private handleReset = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback(this.state.error!, this.state.errorInfo!);
      }

      return (
        <div style={{ padding: '24px' }}>
          <Card>
            <Result
              status="500"
              title="组件错误"
              subTitle={
                <div>
                  <p>抱歉，组件出现了错误。</p>
                  {process.env.NODE_ENV === 'development' && (
                    <pre style={{ marginTop: '16px', fontSize: '12px' }}>
                      {this.state.error?.stack}
                    </pre>
                  )}
                </div>
              }
              extra={[
                <Button key="reset" onClick={this.handleReset}>
                  重试
                </Button>,
                <Button 
                  key="reload" 
                  type="primary" 
                  onClick={this.handleReload}
                >
                  刷新页面
                </Button>
              ]}
            />
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}
```

### 9.2 异步错误处理

```typescript
// src/utils/errorHandler.ts
import { message } from 'antd';

interface ErrorContext {
  component?: string;
  action?: string;
  metadata?: Record<string, any>;
}

export class ErrorHandler {
  static handle(error: unknown, context?: ErrorContext) {
    let errorMessage = '未知错误';
    let errorCode = 'UNKNOWN_ERROR';

    if (error instanceof Error) {
      errorMessage = error.message;
    } else if (typeof error === 'string') {
      errorMessage = error;
    }

    // 错误码映射
    if (errorMessage.includes('网络')) {
      errorCode = 'NETWORK_ERROR';
    } else if (errorMessage.includes('权限')) {
      errorCode = 'PERMISSION_ERROR';
    } else if (errorMessage.includes('认证')) {
      errorCode = 'AUTH_ERROR';
    }

    // 记录错误
    this.logError(error, context);

    // 用户提示
    message.error(errorMessage);

    return { errorCode, errorMessage };
  }

  private static logError(error: unknown, context?: ErrorContext) {
    console.error('Error caught:', {
      error,
      context,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      userAgent: navigator.userAgent
    });
  }
}

// 使用装饰器
export const withErrorHandler = (
  component: string,
  action?: string
) => {
  return function <T extends (...args: any[]) => Promise<any>>(target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;

    descriptor.value = async function (...args: any[]) {
      try {
        return await method.apply(this, args);
      } catch (error) {
        ErrorHandler.handle(error, { component, action });
        throw error;
      }
    };
  };
};
```

这个前端React架构设计文档提供了完整的技术规范，包括组件设计、状态管理、权限控制、实时通信和性能优化等各个方面。