CREATE TABLE knowledge_documents (
    id SERIAL PRIMARY KEY,
    collection_id INTEGER NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT,
    file_type VARCHAR(50),
    content TEXT,
    chunk_count INTEGER DEFAULT 0,
    metadata TEXT,
    upload_status VARCHAR(20) DEFAULT 'uploading' NOT NULL,
    processing_error TEXT,
    uploaded_by INTEGER,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now() NOT NULL
);