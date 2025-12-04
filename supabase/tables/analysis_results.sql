CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    analysis_id INTEGER NOT NULL,
    result_type VARCHAR(50) NOT NULL,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    confidence_score FLOAT,
    metadata TEXT,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL
);