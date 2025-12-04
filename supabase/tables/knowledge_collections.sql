CREATE TABLE knowledge_collections (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    collection_type VARCHAR(20) DEFAULT 'document' NOT NULL,
    chroma_collection_id VARCHAR(255) UNIQUE,
    document_count INTEGER DEFAULT 0,
    embedding_model VARCHAR(100) DEFAULT 'bge-base-zh-v1.5',
    is_active BOOLEAN DEFAULT true NOT NULL,
    created_by INTEGER,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now() NOT NULL
);