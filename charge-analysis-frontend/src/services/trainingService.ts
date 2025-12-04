import { apiRequest } from '../lib/api';

interface DatasetResponse {
  id: number;
  sampleCount: number;
}

interface TrainingTaskStatus {
  id: number;
  name: string;
  status: string;
  progress: number;
  metrics?: Record<string, unknown> | null;
}

export const trainingService = {
  async uploadDataset(
    name: string,
    file: File,
    token: string,
    options?: {
      description?: string;
      datasetType?: string;
    }
  ): Promise<DatasetResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);
    if (options?.description) {
      formData.append('description', options.description);
    }
    if (options?.datasetType) {
      formData.append('dataset_type', options.datasetType);
    }

    const data = await apiRequest<{ dataset_id: number; sample_count: number }>(
      '/api/training/datasets',
      {
        method: 'POST',
        token,
        body: formData
      }
    );

    return { id: data.dataset_id, sampleCount: data.sample_count };
  },

  async createTrainingTask(
    name: string,
    datasetId: number,
    modelType: string,
    hyperparameters: Record<string, unknown>,
    token: string,
    description?: string
  ): Promise<{ id: number; status: string }> {
    const data = await apiRequest<{ task_id: number; status: string }>('/api/training/tasks', {
      method: 'POST',
      token,
      body: {
        name,
        description,
        dataset_id: datasetId,
        model_type: modelType,
        hyperparameters
      }
    });

    return { id: data.task_id, status: data.status };
  },

  async startTraining(taskId: number, token: string): Promise<void> {
    await apiRequest(`/api/training/tasks/${taskId}/start`, {
      method: 'POST',
      token
    });
  },

  async getTrainingStatus(taskId: number, token: string): Promise<TrainingTaskStatus> {
    const data = await apiRequest<{
      id: number;
      name: string;
      status: string;
      progress: number;
      metrics?: Record<string, unknown> | null;
    }>(`/api/training/tasks/${taskId}` , {
      token
    });

    return {
      id: data.id,
      name: data.name,
      status: data.status,
      progress: data.progress,
      metrics: data.metrics
    };
  },

  async createModelVersion(
    payload: {
      name: string;
      modelType: string;
      version: string;
      modelPath: string;
      config?: string;
      metrics?: string;
    },
    token: string
  ): Promise<{ modelVersionId: number }> {
    const formData = new FormData();
    formData.append('name', payload.name);
    formData.append('model_type', payload.modelType);
    formData.append('version', payload.version);
    formData.append('model_path', payload.modelPath);
    if (payload.config) {
      formData.append('config', payload.config);
    }
    if (payload.metrics) {
      formData.append('metrics', payload.metrics);
    }

    const data = await apiRequest<{ model_version_id: number }>('/api/training/models', {
      method: 'POST',
      token,
      body: formData
    });

    return { modelVersionId: data.model_version_id };
  }
};
