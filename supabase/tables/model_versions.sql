CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) DEFAULT 'flow_control',
    version VARCHAR(50) NOT NULL,
    model_path VARCHAR(500) NOT NULL,
    config TEXT,
    metrics TEXT,
    is_active BOOLEAN DEFAULT false NOT NULL,
    is_default BOOLEAN DEFAULT false NOT NULL,
    created_by INTEGER,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL
);