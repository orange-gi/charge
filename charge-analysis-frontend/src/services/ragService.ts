import { callEdgeFunction, querySupabase } from '../lib/supabase';

interface KnowledgeCollection {
  id: number;
  name: string;
  description?: string;
  collectionType: string;
  documentCount: number;
  embeddingModel: string;
  isActive: boolean;
  createdBy?: number;
  createdAt: string;
  updatedAt: string;
}

interface KnowledgeDocument {
  id: number;
  collectionId: number;
  filename: string;
  filePath: string;
  fileSize: number;
  fileType?: string;
  content?: string;
  chunkCount: number;
  metadata?: string;
  uploadStatus: string;
  processingError?: string;
  uploadedBy?: number;
  createdAt: string;
  updatedAt: string;
}

interface RAGQuery {
  id: number;
  collectionId: number;
  queryText: string;
  resultCount: number;
  responseText?: string;
  userId?: number;
  queryTimeMs?: number;
  createdAt: string;
}

export const ragService = {
  // Create collection
  async createCollection(
    name: string,
    description: string,
    userId: number,
    token: string
  ): Promise<KnowledgeCollection> {
    const response = await callEdgeFunction(
      'ragQuery',
      {
        action: 'create_collection',
        name,
        description,
        userId
      },
      token
    );

    return transformCollectionData(response.data.collection);
  },

  // Get all collections
  async getCollections(): Promise<KnowledgeCollection[]> {
    const data = await querySupabase('knowledge_collections', 'GET', {
      order: { column: 'created_at', ascending: false }
    });

    return data.map(transformCollectionData);
  },

  // Get collection by ID
  async getCollection(collectionId: number): Promise<KnowledgeCollection> {
    const data = await querySupabase('knowledge_collections', 'GET', {
      filter: { id: collectionId },
      single: true
    });

    return transformCollectionData(data);
  },

  // Upload document
  async uploadDocument(
    collectionId: number,
    file: File,
    userId: number,
    token: string
  ): Promise<KnowledgeDocument> {
    const base64Data = await fileToBase64(file);

    const response = await callEdgeFunction(
      'ragQuery',
      {
        action: 'upload',
        collectionId,
        documentFile: base64Data,
        documentName: file.name,
        userId
      },
      token
    );

    return transformDocumentData(response.data.document);
  },

  // Get documents in collection
  async getDocuments(collectionId: number): Promise<KnowledgeDocument[]> {
    const data = await querySupabase('knowledge_documents', 'GET', {
      filter: { collection_id: collectionId },
      order: { column: 'created_at', ascending: false }
    });

    return data.map(transformDocumentData);
  },

  // Query RAG
  async query(
    collectionId: number,
    query: string,
    userId: number,
    token: string
  ): Promise<{ response: string; documents: any[]; queryTime: number }> {
    const response = await callEdgeFunction(
      'ragQuery',
      {
        action: 'query',
        collectionId,
        query,
        userId
      },
      token
    );

    return response.data;
  },

  // Get query history
  async getQueryHistory(collectionId: number, limit: number = 50): Promise<RAGQuery[]> {
    const data = await querySupabase('rag_queries', 'GET', {
      filter: { collection_id: collectionId },
      order: { column: 'created_at', ascending: false },
      limit
    });

    return data.map(transformQueryData);
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

function transformCollectionData(data: any): KnowledgeCollection {
  return {
    id: data.id,
    name: data.name,
    description: data.description,
    collectionType: data.collection_type,
    documentCount: data.document_count,
    embeddingModel: data.embedding_model,
    isActive: data.is_active,
    createdBy: data.created_by,
    createdAt: data.created_at,
    updatedAt: data.updated_at
  };
}

function transformDocumentData(data: any): KnowledgeDocument {
  return {
    id: data.id,
    collectionId: data.collection_id,
    filename: data.filename,
    filePath: data.file_path,
    fileSize: data.file_size,
    fileType: data.file_type,
    content: data.content,
    chunkCount: data.chunk_count,
    metadata: data.metadata,
    uploadStatus: data.upload_status,
    processingError: data.processing_error,
    uploadedBy: data.uploaded_by,
    createdAt: data.created_at,
    updatedAt: data.updated_at
  };
}

function transformQueryData(data: any): RAGQuery {
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
