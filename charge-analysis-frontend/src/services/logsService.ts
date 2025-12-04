import { querySupabase } from '../lib/supabase';

export interface SystemLog {
  id: number;
  level: string;
  module: string;
  message: string;
  loggerName?: string;
  functionName?: string;
  lineNumber?: number;
  filePath?: string;
  userId?: number;
  requestId?: string;
  sessionId?: string;
  metadata?: string;
  ipAddress?: string;
  userAgent?: string;
  timestamp: string;
}

export interface AuditLog {
  id: number;
  action: string;
  resourceType: string;
  resourceId?: string;
  oldValues?: string;
  newValues?: string;
  userId?: number;
  userName?: string;
  ipAddress?: string;
  userAgent?: string;
  timestamp: string;
}

export interface LogFilters {
  level?: string;
  module?: string;
  startDate?: string;
  endDate?: string;
  search?: string;
}

export const logsService = {
  /**
   * 获取系统日志
   */
  async getSystemLogs(
    page: number = 1,
    pageSize: number = 20,
    filters?: LogFilters
  ): Promise<{ logs: SystemLog[]; total: number }> {
    try {
      const offset = (page - 1) * pageSize;
      
      // 构建过滤条件
      const filter: Record<string, any> = {};
      if (filters?.level) {
        filter.level = filters.level;
      }
      if (filters?.module) {
        filter.module = filters.module;
      }

      // 查询日志
      const logs = await querySupabase('system_logs', 'GET', {
        filter,
        order: { column: 'timestamp', ascending: false },
        limit: pageSize
      });

      // 获取总数（简化版，实际应该单独查询count）
      const total = logs.length;

      return {
        logs: logs.map((log: any) => ({
          id: log.id,
          level: log.level,
          module: log.module,
          message: log.message,
          loggerName: log.logger_name,
          functionName: log.function_name,
          lineNumber: log.line_number,
          filePath: log.file_path,
          userId: log.user_id,
          requestId: log.request_id,
          sessionId: log.session_id,
          metadata: log.metadata,
          ipAddress: log.ip_address,
          userAgent: log.user_agent,
          timestamp: log.timestamp
        })),
        total
      };
    } catch (error) {
      console.error('获取系统日志失败:', error);
      return { logs: [], total: 0 };
    }
  },

  /**
   * 获取审计日志
   */
  async getAuditLogs(
    page: number = 1,
    pageSize: number = 20,
    filters?: { action?: string; resourceType?: string }
  ): Promise<{ logs: AuditLog[]; total: number }> {
    try {
      const offset = (page - 1) * pageSize;
      
      // 构建过滤条件
      const filter: Record<string, any> = {};
      if (filters?.action) {
        filter.action = filters.action;
      }
      if (filters?.resourceType) {
        filter.resource_type = filters.resourceType;
      }

      // 查询审计日志
      const logs = await querySupabase('audit_logs', 'GET', {
        filter,
        order: { column: 'timestamp', ascending: false },
        limit: pageSize
      });

      // 获取用户信息
      const userIds = [...new Set(logs.map((log: any) => log.user_id).filter((id: any) => id))];
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

      const total = logs.length;

      return {
        logs: logs.map((log: any) => ({
          id: log.id,
          action: log.action,
          resourceType: log.resource_type,
          resourceId: log.resource_id,
          oldValues: log.old_values,
          newValues: log.new_values,
          userId: log.user_id,
          userName: userMap.get(log.user_id),
          ipAddress: log.ip_address,
          userAgent: log.user_agent,
          timestamp: log.timestamp
        })),
        total
      };
    } catch (error) {
      console.error('获取审计日志失败:', error);
      return { logs: [], total: 0 };
    }
  },

  /**
   * 获取日志级别列表
   */
  getLogLevels(): string[] {
    return ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];
  },

  /**
   * 获取模块列表
   */
  async getModules(): Promise<string[]> {
    try {
      const logs = await querySupabase('system_logs', 'GET', {
        limit: 100
      });

      const modules = [...new Set(logs.map((log: any) => log.module))] as string[];
      return modules.filter((m: string) => m);
    } catch (error) {
      console.error('获取模块列表失败:', error);
      return [];
    }
  },

  /**
   * 获取操作类型列表
   */
  getActionTypes(): string[] {
    return ['CREATE', 'UPDATE', 'DELETE', 'LOGIN', 'LOGOUT', 'UPLOAD', 'DOWNLOAD'];
  },

  /**
   * 获取资源类型列表
   */
  getResourceTypes(): string[] {
    return [
      'charging_analysis',
      'training_task',
      'knowledge_document',
      'rag_query',
      'user',
      'model_version'
    ];
  }
};
