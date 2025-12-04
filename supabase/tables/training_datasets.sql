CREATE TABLE training_datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    dataset_type VARCHAR(50) DEFAULT 'standard',
    file_path VARCHAR(500),
    sample_count INTEGER DEFAULT 0,
    metadata TEXT,
    is_public BOOLEAN DEFAULT false NOT NULL,
    created_by INTEGER,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now() NOT NULL
);