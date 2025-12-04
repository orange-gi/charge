import { callEdgeFunction, querySupabase } from '../lib/supabase';

interface ChargingAnalysis {
  id: number;
  name: string;
  description?: string;
  filePath: string;
  fileSize: number;
  fileType: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  resultData?: string;
  errorMessage?: string;
  userId: number;
  createdAt: string;
  updatedAt: string;
  startedAt?: string;
  completedAt?: string;
}

interface AnalysisResult {
  id: number;
  analysisId: number;
  resultType: string;
  title: string;
  content: string;
  confidenceScore?: number;
  metadata?: string;
  createdAt: string;
}

export const chargingService = {
  // Upload file and create analysis
  async uploadFile(
    file: File,
    userId: number,
    token: string,
    analysisName?: string,
    description?: string
  ): Promise<{ analysisId: number; publicUrl: string }> {
    // Convert file to base64
    const base64Data = await fileToBase64(file);

    const response = await callEdgeFunction(
      'fileUpload',
      {
        fileData: base64Data,
        fileName: file.name,
        fileSize: file.size,
        userId,
        analysisName: analysisName || file.name,
        description
      },
      token
    );

    return {
      analysisId: response.data.analysisId,
      publicUrl: response.data.publicUrl
    };
  },

  // Start analysis
  async startAnalysis(analysisId: number, userId: number, token: string): Promise<void> {
    await callEdgeFunction(
      'chargingAnalysis',
      {
        analysisId,
        userId
      },
      token
    );
  },

  // Get analysis by ID
  async getAnalysis(analysisId: number): Promise<ChargingAnalysis> {
    const data = await querySupabase('charging_analyses', 'GET', {
      filter: { id: analysisId },
      single: true
    });

    return transformAnalysisData(data);
  },

  // Get user's analyses
  async getUserAnalyses(userId: number, limit: number = 50): Promise<ChargingAnalysis[]> {
    const data = await querySupabase('charging_analyses', 'GET', {
      filter: { user_id: userId },
      order: { column: 'created_at', ascending: false },
      limit
    });

    return data.map(transformAnalysisData);
  },

  // Get analysis results
  async getAnalysisResults(analysisId: number): Promise<AnalysisResult[]> {
    const data = await querySupabase('analysis_results', 'GET', {
      filter: { analysis_id: analysisId },
      order: { column: 'created_at', ascending: true }
    });

    return data.map(transformResultData);
  },

  // Delete analysis
  async deleteAnalysis(analysisId: number): Promise<void> {
    await querySupabase('charging_analyses', 'DELETE', {
      filter: { id: analysisId }
    });
  }
};

// Helper functions
function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      resolve(reader.result as string);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function transformAnalysisData(data: any): ChargingAnalysis {
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

function transformResultData(data: any): AnalysisResult {
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
