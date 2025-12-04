CREATE TABLE charging_analyses (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT,
    file_type VARCHAR(20) DEFAULT 'blf',
    status VARCHAR(50) DEFAULT 'pending' NOT NULL,
    progress FLOAT DEFAULT 0.0,
    result_data TEXT,
    error_message TEXT,
    user_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now() NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);