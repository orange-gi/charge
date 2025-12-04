CREATE TABLE rag_queries (
    id SERIAL PRIMARY KEY,
    collection_id INTEGER NOT NULL,
    query_text TEXT NOT NULL,
    result_count INTEGER DEFAULT 0,
    response_text TEXT,
    user_id INTEGER,
    query_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL
);