import { apiRequest } from '../lib/api';

export interface KnowledgeCollection {
  id: number;
  name: string;
  description?: string | null;
  documentCount: number;
  embeddingModel: string;
  collectionType: string;
  isActive: boolean;
  createdBy?: number | null;
  createdAt: string;
  updatedAt: string;
}

export interface KnowledgeDocument {
  id: number;
  collectionId: number;
  filename: string;
  filePath: string;
  fileSize?: number | null;
  fileType?: string | null;
  content?: string | null;
  chunkCount: number;
  metaInfo?: string | null;
  uploadStatus: string;
  processingError?: string | null;
  uploadedBy?: number | null;
  createdAt: string;
  updatedAt: string;
}

export interface RAGQueryRecord {
  id: number;
  collectionId: number;
  queryText: string;
  resultCount: number;
  responseText?: string | null;
  userId?: number | null;
  queryTimeMs?: number | null;
  createdAt: string;
}

export interface RagDocumentLog {
  id: number;
  documentId: number;
  logLevel: string;
  message: string;
  createdAt: string;
}

interface BackendCollection {
  id: number;
  name: string;
  description?: string | null;
  document_count: number;
  embedding_model: string;
  collection_type: string;
  is_active: boolean;
  created_by?: number | null;
  created_at: string;
  updated_at: string;
}

interface BackendDocument {
  id: number;
  collection_id: number;
  filename: string;
  file_path: string;
  file_size?: number | null;
  file_type?: string | null;
  content?: string | null;
  chunk_count: number;
  meta_info?: string | null;
  upload_status: string;
  processing_error?: string | null;
  uploaded_by?: number | null;
  created_at: string;
  updated_at: string;
}

interface BackendQueryRecord {
  id: number;
  collection_id: number;
  query_text: string;
  result_count: number;
  response_text?: string | null;
  user_id?: number | null;
  query_time_ms?: number | null;
  created_at: string;
}

interface BackendRagDocumentLog {
  id: number;
  document_id: number;
  log_level: string;
  message: string;
  created_at: string;
}

export const ragService = {
  async createCollection(name: string, description: string | undefined, token: string): Promise<KnowledgeCollection> {
    const data = await apiRequest<BackendCollection>('/api/rag/collections', {
      method: 'POST',
      token,
      body: {
        name,
        description
      }
    });
    return transformCollectionData(data);
  },

  async getCollections(token: string): Promise<KnowledgeCollection[]> {
    const data = await apiRequest<BackendCollection[]>('/api/rag/collections', {
      token
    });
    return data.map(transformCollectionData);
  },

  async getDocuments(collectionId: number, token: string): Promise<KnowledgeDocument[]> {
    const data = await apiRequest<BackendDocument[]>(`/api/rag/collections/${collectionId}/documents`, {
      token
    });
    return data.map(transformDocumentData);
  },

  async getDocument(collectionId: number, documentId: number, token: string): Promise<KnowledgeDocument> {
    const data = await apiRequest<BackendDocument>(`/api/rag/collections/${collectionId}/documents/${documentId}`, {
      token
    });
    return transformDocumentData(data);
  },

  async getDocumentLogs(collectionId: number, documentId: number, token: string, limit: number = 200): Promise<RagDocumentLog[]> {
    const data = await apiRequest<BackendRagDocumentLog[]>(
      `/api/rag/collections/${collectionId}/documents/${documentId}/logs?limit=${limit}`,
      { token }
    );
    return data.map((item) => ({
      id: item.id,
      documentId: item.document_id,
      logLevel: item.log_level,
      message: item.message,
      createdAt: item.created_at
    }));
  },

  async uploadDocument(
    collectionId: number,
    file: File,
    token: string,
    options?: { overwrite?: boolean }
  ): Promise<KnowledgeDocument> {
    const formData = new FormData();
    formData.append('file', file);
    if (options?.overwrite) {
      formData.append('overwrite', 'true');
    }

    const data = await apiRequest<BackendDocument>(`/api/rag/collections/${collectionId}/documents`, {
      method: 'POST',
      token,
      body: formData
    });

    return transformDocumentData(data);
  },

  async query(
    collectionId: number,
    queryText: string,
    token: string
  ): Promise<{ response: string; documents: any[]; queryTime: number }> {
    const response = await apiRequest<{ response: string; documents: any[]; query_time: number }>(
      '/api/rag/query',
      {
        method: 'POST',
        token,
        body: {
          collection_id: collectionId,
          query: queryText
        }
      }
    );

    return {
      response: response.response,
      documents: response.documents,
      queryTime: response.query_time
    };
  },

  async getQueryHistory(collectionId: number, token: string, limit: number = 50): Promise<RAGQueryRecord[]> {
    const data = await apiRequest<BackendQueryRecord[]>(
      `/api/rag/collections/${collectionId}/queries?limit=${limit}`,
      { token }
    );
    return data.map(transformQueryData);
  }
};

function transformCollectionData(data: BackendCollection): KnowledgeCollection {
  return {
    id: data.id,
    name: data.name,
    description: data.description,
    documentCount: data.document_count,
    embeddingModel: data.embedding_model,
    collectionType: data.collection_type,
    isActive: data.is_active,
    createdBy: data.created_by,
    createdAt: data.created_at,
    updatedAt: data.updated_at
  };
}

function transformDocumentData(data: BackendDocument): KnowledgeDocument {
  return {
    id: data.id,
    collectionId: data.collection_id,
    filename: data.filename,
    filePath: data.file_path,
    fileSize: data.file_size,
    fileType: data.file_type,
    content: data.content,
    chunkCount: data.chunk_count,
    metaInfo: data.meta_info,
    uploadStatus: data.upload_status,
    processingError: data.processing_error,
    uploadedBy: data.uploaded_by,
    createdAt: data.created_at,
    updatedAt: data.updated_at
  };
}

function transformQueryData(data: BackendQueryRecord): RAGQueryRecord {
  return {
    id: data.id,
    collectionId: data.collection_id,
    queryText: data.query_text,
    resultCount: data.result_count,
    responseText: data.response_text,
    userId: data.user_id,
    queryTimeMs: data.query_time_ms,
    createdAt: data.created_at
  };
}
