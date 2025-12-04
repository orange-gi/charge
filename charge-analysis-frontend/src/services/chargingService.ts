import { apiRequest } from '../lib/api';

export interface ChargingAnalysis {
  id: number;
  name: string;
  description?: string | null;
  filePath: string;
  fileSize?: number | null;
  fileType?: string | null;
  status: string;
  progress: number;
  resultData?: string | null;
  errorMessage?: string | null;
  userId: number;
  createdAt: string;
  updatedAt: string;
  startedAt?: string | null;
  completedAt?: string | null;
}

export interface AnalysisResult {
  id: number;
  analysisId: number;
  resultType: string;
  title: string;
  content: string;
  confidenceScore?: number | null;
  metadata?: string | null;
  createdAt: string;
}

interface BackendAnalysis {
  id: number;
  name: string;
  description?: string | null;
  file_path: string;
  file_size?: number | null;
  file_type?: string | null;
  status: string;
  progress: number;
  result_data?: string | null;
  error_message?: string | null;
  user_id: number;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

interface BackendAnalysisResult {
  id: number;
  analysis_id: number;
  result_type: string;
  title: string;
  content: string;
  confidence_score?: number | null;
  metadata?: string | null;
  created_at: string;
}

export const chargingService = {
  async uploadFile(
    file: File,
    token: string,
    analysisName?: string,
    description?: string
  ): Promise<ChargingAnalysis> {
    const formData = new FormData();
    formData.append('file', file);
    if (analysisName) {
      formData.append('analysis_name', analysisName);
    }
    if (description) {
      formData.append('description', description);
    }

    const data = await apiRequest<BackendAnalysis>('/api/analyses/upload', {
      method: 'POST',
      token,
      body: formData
    });

    return transformAnalysisData(data);
  },

  async startAnalysis(analysisId: number, token: string): Promise<void> {
    await apiRequest(`/api/analyses/${analysisId}/run`, {
      method: 'POST',
      token
    });
  },

  async getAnalysis(analysisId: number, token: string): Promise<ChargingAnalysis> {
    const data = await apiRequest<BackendAnalysis>(`/api/analyses/${analysisId}`, {
      token
    });
    return transformAnalysisData(data);
  },

  async getUserAnalyses(token: string): Promise<ChargingAnalysis[]> {
    const payload = await apiRequest<{ items: BackendAnalysis[] }>('/api/analyses', {
      token
    });
    return (payload.items || []).map(transformAnalysisData);
  },

  async getAnalysisResults(analysisId: number, token: string): Promise<AnalysisResult[]> {
    const data = await apiRequest<{ analysis: BackendAnalysis; results: BackendAnalysisResult[] }>(
      `/api/analyses/${analysisId}/results`,
      { token }
    );
    return (data.results || []).map(transformResultData);
  },

  async deleteAnalysis(analysisId: number, token: string): Promise<void> {
    await apiRequest(`/api/analyses/${analysisId}`, {
      method: 'DELETE',
      token
    });
  }
};

function transformAnalysisData(data: BackendAnalysis): ChargingAnalysis {
  return {
    id: data.id,
    name: data.name,
    description: data.description,
    filePath: data.file_path,
    fileSize: data.file_size,
    fileType: data.file_type,
    status: data.status,
    progress: data.progress,
    resultData: data.result_data,
    errorMessage: data.error_message,
    userId: data.user_id,
    createdAt: data.created_at,
    updatedAt: data.updated_at,
    startedAt: data.started_at,
    completedAt: data.completed_at
  };
}

function transformResultData(data: BackendAnalysisResult): AnalysisResult {
  return {
    id: data.id,
    analysisId: data.analysis_id,
    resultType: data.result_type,
    title: data.title,
    content: data.content,
    confidenceScore: data.confidence_score,
    metadata: data.metadata,
    createdAt: data.created_at
  };
}
