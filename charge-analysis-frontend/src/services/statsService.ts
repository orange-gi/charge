import { apiRequest } from '../lib/api';

export interface SystemStats {
  totalAnalyses: number;
  completedAnalyses: number;
  activeUsers: number;
  knowledgeDocuments: number;
}

export interface RecentActivity {
  id: number;
  type: string;
  user: string;
  action: string;
  time: string;
  timestamp: string;
}

interface BackendAnalysis {
  id: number;
  name: string;
  status: string;
  created_at: string;
  updated_at: string;
  user_id: number;
}

interface BackendCollectionSummary {
  document_count: number;
}

const DEFAULT_STATS: SystemStats = {
  totalAnalyses: 0,
  completedAnalyses: 0,
  activeUsers: 0,
  knowledgeDocuments: 0
};

export const statsService = {
  async getSystemStats(token: string | null): Promise<SystemStats> {
    if (!token) {
      return DEFAULT_STATS;
    }

    try {
      const analysesPayload = await apiRequest<{ items: BackendAnalysis[] }>('/api/analyses', { token });
      const analyses = analysesPayload.items || [];

      let knowledgeDocuments = 0;
      try {
        const collections = await apiRequest<BackendCollectionSummary[]>('/api/rag/collections', { token });
        knowledgeDocuments = collections.reduce((total, item) => total + (item.document_count || 0), 0);
      } catch {
        knowledgeDocuments = 0;
      }

      const activeUsers = new Set(analyses.map(item => item.user_id)).size || 1;
      const completedAnalyses = analyses.filter(item => item.status === 'completed').length;

      return {
        totalAnalyses: analyses.length,
        completedAnalyses,
        activeUsers,
        knowledgeDocuments
      };
    } catch (error) {
      console.error('获取系统统计失败:', error);
      return DEFAULT_STATS;
    }
  },

  async getRecentActivities(
    token: string | null,
    currentUserName: string,
    limit: number = 10
  ): Promise<RecentActivity[]> {
    if (!token) {
      return [];
    }

    try {
      const analysesPayload = await apiRequest<{ items: BackendAnalysis[] }>('/api/analyses', { token });
      const analyses = (analysesPayload.items || [])
        .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
        .slice(0, limit);

      return analyses.map((analysis) => {
        const action = analysis.status === 'completed' ? '完成了充电分析' : '提交了充电数据';
        const timestamp = analysis.updated_at || analysis.created_at;
        return {
          id: analysis.id,
          type: '充电分析',
          user: currentUserName,
          action: `${action}「${analysis.name}」`,
          time: getTimeAgo(new Date(timestamp)),
          timestamp
        };
      });
    } catch (error) {
      console.error('获取最近活动失败:', error);
      return [];
    }
  }
};

function getTimeAgo(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return '刚刚';
  if (diffMins < 60) return `${diffMins}分钟前`;
  if (diffHours < 24) return `${diffHours}小时前`;
  if (diffDays < 7) return `${diffDays}天前`;

  return date.toLocaleDateString('zh-CN');
}
