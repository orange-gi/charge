import { callEdgeFunction } from '../lib/supabase';

export const trainingService = {
  // Upload dataset
  async uploadDataset(
    name: string,
    file: File,
    userId: number,
    token: string,
    description?: string,
    datasetType?: string
  ) {
    const base64Data = await fileToBase64(file);

    const response = await callEdgeFunction(
      'trainingManagement',
      {
        action: 'upload_dataset',
        name,
        description,
        datasetType: datasetType || 'standard',
        fileData: base64Data,
        fileName: file.name,
        userId
      },
      token
    );

    return response.data.dataset;
  },

  // Create training task
  async createTrainingTask(
    name: string,
    datasetId: number,
    modelType: string,
    hyperparameters: any,
    userId: number,
    token: string,
    description?: string
  ) {
    const response = await callEdgeFunction(
      'trainingManagement',
      {
        action: 'create_training_task',
        name,
        description,
        datasetId,
        modelType,
        hyperparameters,
        userId
      },
      token
    );

    return response.data.task;
  },

  // Start training
  async startTraining(taskId: number, token: string) {
    const response = await callEdgeFunction(
      'trainingManagement',
      {
        action: 'start_training',
        taskId
      },
      token
    );

    return response.data;
  },

  // Get training status
  async getTrainingStatus(taskId: number, token: string) {
    const response = await callEdgeFunction(
      'trainingManagement',
      {
        action: 'get_training_status',
        taskId
      },
      token
    );

    return response.data.task;
  },

  // Create model version
  async createModelVersion(
    name: string,
    modelType: string,
    version: string,
    modelPath: string,
    config: any,
    metrics: any,
    userId: number,
    token: string
  ) {
    const response = await callEdgeFunction(
      'trainingManagement',
      {
        action: 'create_model_version',
        name,
        modelType,
        version,
        modelPath,
        config,
        metrics,
        userId
      },
      token
    );

    return response.data.modelVersion;
  }
};

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
