import { apiRequest } from '../lib/api';

export interface DatasetResponse {
  id: number;
  sampleCount: number;
}

export type TrainingStatus = 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface TrainingConfigPayload {
  name: string;
  baseModel: string;
  modelPath: string;
  adapterType: 'lora';
  modelSize: '1.5b' | '7b';
  datasetStrategy: string;
  hyperparameters?: Record<string, unknown>;
  notes?: string | null;
}

export interface TrainingConfig extends TrainingConfigPayload {
  id: number;
  createdAt: string;
  updatedAt: string;
}

export interface TrainingTaskDetail {
  id: number;
  name: string;
  status: TrainingStatus;
  progress: number;
  modelSize: '1.5b' | '7b';
  adapterType: 'lora';
  datasetId: number | null;
  configId: number | null;
  currentEpoch?: number | null;
  totalEpochs?: number | null;
  metrics?: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface TrainingLogEntry {
  id: number;
  logLevel: string;
  message: string;
  createdAt: string;
}

export interface TrainingMetricPoint {
  epoch: number;
  step: number;
  loss?: number;
  accuracy?: number;
  learningRate?: number;
  gpuMemory?: number;
  createdAt: string;
}

export interface TrainingEvaluation {
  id: number;
  taskId: number;
  evaluationType: string;
  evaluator?: string | null;
  metrics: Record<string, unknown>;
  recommendedPlan: string;
  notes?: string | null;
  createdAt: string;
}

const toTrainingConfig = (raw: any): TrainingConfig => ({
  id: raw.id,
  name: raw.name,
  baseModel: raw.base_model,
  modelPath: raw.model_path,
  adapterType: raw.adapter_type,
  modelSize: raw.model_size,
  datasetStrategy: raw.dataset_strategy,
  hyperparameters: raw.hyperparameters ?? {},
  notes: raw.notes,
  createdAt: raw.created_at,
  updatedAt: raw.updated_at
});

const toTaskDetail = (raw: any): TrainingTaskDetail => ({
  id: raw.id,
  name: raw.name,
  status: raw.status,
  progress: raw.progress,
  modelSize: raw.model_size,
  adapterType: raw.adapter_type,
  datasetId: raw.dataset_id ?? null,
  configId: raw.config_id ?? null,
  currentEpoch: raw.current_epoch ?? null,
  totalEpochs: raw.total_epochs ?? null,
  metrics: raw.metrics ?? {},
  createdAt: raw.created_at,
  updatedAt: raw.updated_at
});

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

  async fetchConfigs(token: string): Promise<TrainingConfig[]> {
    const data = await apiRequest<any[]>('/api/training/configs', { token });
    return data.map(toTrainingConfig);
  },

  async saveConfig(payload: TrainingConfigPayload, token: string): Promise<TrainingConfig> {
    const data = await apiRequest<any>('/api/training/configs', {
      method: 'POST',
      token,
      body: {
        name: payload.name,
        base_model: payload.baseModel,
        model_path: payload.modelPath,
        adapter_type: payload.adapterType,
        model_size: payload.modelSize,
        dataset_strategy: payload.datasetStrategy,
        hyperparameters: payload.hyperparameters,
        notes: payload.notes
      }
    });
    return toTrainingConfig(data);
  },

  async updateConfig(configId: number, payload: TrainingConfigPayload, token: string): Promise<TrainingConfig> {
    const data = await apiRequest<any>(`/api/training/configs/${configId}`, {
      method: 'PUT',
      token,
      body: {
        name: payload.name,
        base_model: payload.baseModel,
        model_path: payload.modelPath,
        adapter_type: payload.adapterType,
        model_size: payload.modelSize,
        dataset_strategy: payload.datasetStrategy,
        hyperparameters: payload.hyperparameters,
        notes: payload.notes
      }
    });
    return toTrainingConfig(data);
  },

  async listTasks(token: string): Promise<TrainingTaskDetail[]> {
    const data = await apiRequest<{ items: any[] }>('/api/training/tasks', { token });
    return data.items.map(toTaskDetail);
  },

  async createTrainingTask(
    payload: {
      name: string;
      description?: string;
      datasetId: number;
      configId?: number | null;
      modelType: string;
      modelSize: '1.5b' | '7b';
      hyperparameters: Record<string, unknown>;
    },
    token: string
  ): Promise<{ id: number; status: TrainingStatus }> {
    const data = await apiRequest<{ task_id: number; status: TrainingStatus }>('/api/training/tasks', {
      method: 'POST',
      token,
      body: {
        name: payload.name,
        description: payload.description,
        dataset_id: payload.datasetId,
        config_id: payload.configId,
        model_type: payload.modelType,
        model_size: payload.modelSize,
        hyperparameters: payload.hyperparameters,
        adapter_type: 'lora'
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

  async getTask(taskId: number, token: string): Promise<TrainingTaskDetail> {
    const data = await apiRequest<any>(`/api/training/tasks/${taskId}`, { token });
    return toTaskDetail(data);
  },

  async getTaskLogs(taskId: number, token: string): Promise<TrainingLogEntry[]> {
    const data = await apiRequest<any[]>(`/api/training/tasks/${taskId}/logs`, { token });
    return data.map((item) => ({
      id: item.id,
      logLevel: item.log_level,
      message: item.message,
      createdAt: item.created_at
    }));
  },

  async getTaskMetrics(taskId: number, token: string): Promise<TrainingMetricPoint[]> {
    const data = await apiRequest<any[]>(`/api/training/tasks/${taskId}/metrics`, { token });
    return data.map((item) => ({
      epoch: item.epoch,
      step: item.step,
      loss: item.loss,
      accuracy: item.accuracy,
      learningRate: item.learning_rate,
      gpuMemory: item.gpu_memory,
      createdAt: item.created_at
    }));
  },

  async evaluateTask(
    taskId: number,
    payload: {
      evaluationType: string;
      metrics: Record<string, unknown>;
      recommendedPlan: string;
      notes?: string;
    },
    token: string
  ): Promise<TrainingEvaluation> {
    const data = await apiRequest<any>(`/api/training/tasks/${taskId}/evaluate`, {
      method: 'POST',
      token,
      body: {
        evaluation_type: payload.evaluationType,
        metrics: payload.metrics,
        recommended_plan: payload.recommendedPlan,
        notes: payload.notes
      }
    });
    return {
      id: data.id,
      taskId: data.task_id,
      evaluator: data.evaluator,
      evaluationType: data.evaluation_type,
      metrics: data.metrics,
      recommendedPlan: data.recommended_plan,
      notes: data.notes,
      createdAt: data.created_at
    };
  },

  async getTaskEvaluation(taskId: number, token: string): Promise<TrainingEvaluation | null> {
    const data = await apiRequest<any | null>(`/api/training/tasks/${taskId}/evaluation`, { token });
    if (!data) {
      return null;
    }
    return {
      id: data.id,
      taskId: data.task_id,
      evaluator: data.evaluator,
      evaluationType: data.evaluation_type,
      metrics: data.metrics,
      recommendedPlan: data.recommended_plan,
      notes: data.notes,
      createdAt: data.created_at
    };
  },

  async publishModel(
    taskId: number,
    payload: {
      version: string;
      targetEnvironment: string;
      endpointUrl?: string;
      notes?: string;
      setDefault?: boolean;
    },
    token: string
  ): Promise<{ modelVersionId: number; version: string; endpointUrl?: string }> {
    const data = await apiRequest<{ model_version_id: number; version: string; endpoint_url?: string }>(
      `/api/training/tasks/${taskId}/publish`,
      {
        method: 'POST',
        token,
        body: {
          version: payload.version,
          target_environment: payload.targetEnvironment,
          endpoint_url: payload.endpointUrl,
          notes: payload.notes,
          set_default: payload.setDefault ?? false
        }
      }
    );

    return {
      modelVersionId: data.model_version_id,
      version: data.version,
      endpointUrl: data.endpoint_url
    };
  }
};
