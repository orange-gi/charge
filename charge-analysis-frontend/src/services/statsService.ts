import { querySupabase } from '../lib/supabase';

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

export const statsService = {
  /**
   * 获取系统统计数据
   */
  async getSystemStats(): Promise<SystemStats> {
    try {
      // 并行查询所有统计数据
      const [
        totalAnalysesData,
        completedAnalysesData,
        activeUsersData,
        knowledgeDocsData
      ] = await Promise.all([
        // 总分析次数
        querySupabase('charging_analyses', 'GET'),
        
        // 已完成分析
        querySupabase('charging_analyses', 'GET', {
          filter: { status: 'completed' }
        }),
        
        // 活跃用户（最近7天有活动的用户）
        querySupabase('users', 'GET'),
        
        // 知识库文档数
        querySupabase('knowledge_documents', 'GET')
      ]);

      return {
        totalAnalyses: totalAnalysesData.length,
        completedAnalyses: completedAnalysesData.length,
        activeUsers: activeUsersData.length,
        knowledgeDocuments: knowledgeDocsData.length
      };
    } catch (error) {
      console.error('获取系统统计失败:', error);
      // 返回默认值
      return {
        totalAnalyses: 0,
        completedAnalyses: 0,
        activeUsers: 0,
        knowledgeDocuments: 0
      };
    }
  },

  /**
   * 获取最近活动记录
   */
  async getRecentActivities(limit: number = 10): Promise<RecentActivity[]> {
    try {
      // 查询审计日志获取最近活动
      const auditLogs = await querySupabase('audit_logs', 'GET', {
        order: { column: 'timestamp', ascending: false },
        limit
      });

      // 获取相关用户信息
      const userIds = [...new Set(auditLogs.map((log: any) => log.user_id).filter((id: any) => id))];
      let users: any[] = [];
      
      if (userIds.length > 0) {
        users = await Promise.all(
          userIds.map((userId: number) => 
            querySupabase('users', 'GET', {
              filter: { id: userId },
              single: true
            }).catch(() => null)
          )
        );
      }

      const userMap = new Map(
        users.filter(u => u).map((user: any) => [user.id, user.username || user.email])
      );

      // 转换为活动记录
      const activities: RecentActivity[] = auditLogs.map((log: any) => {
        const userName = userMap.get(log.user_id) || '系统';
        const actionText = getActionText(log.action, log.resource_type);
        const timeAgo = getTimeAgo(new Date(log.timestamp));

        return {
          id: log.id,
          type: getActivityType(log.resource_type),
          user: userName,
          action: actionText,
          time: timeAgo,
          timestamp: log.timestamp
        };
      });

      return activities;
    } catch (error) {
      console.error('获取最近活动失败:', error);
      return [];
    }
  }
};

/**
 * 获取活动类型显示文本
 */
function getActivityType(resourceType: string): string {
  const typeMap: Record<string, string> = {
    'charging_analysis': '充电分析',
    'training_task': '训练任务',
    'knowledge_document': 'RAG查询',
    'rag_query': 'RAG查询',
    'user': '用户管理'
  };
  
  return typeMap[resourceType] || '系统操作';
}

/**
 * 获取操作文本
 */
function getActionText(action: string, resourceType: string): string {
  const actionMap: Record<string, string> = {
    'CREATE': '创建了',
    'UPDATE': '更新了',
    'DELETE': '删除了',
    'COMPLETE': '完成了',
    'START': '启动了'
  };

  const resourceMap: Record<string, string> = {
    'charging_analysis': '充电数据分析',
    'training_task': '新的训练任务',
    'knowledge_document': '知识库文档',
    'rag_query': '知识库文档',
    'user': '用户信息'
  };

  const actionText = actionMap[action] || '操作了';
  const resourceText = resourceMap[resourceType] || resourceType;

  return `${actionText}${resourceText}`;
}

/**
 * 计算时间差
 */
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
