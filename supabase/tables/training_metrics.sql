CREATE TABLE training_metrics (
    id SERIAL PRIMARY KEY,
    task_id INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    step INTEGER NOT NULL,
    loss FLOAT,
    accuracy FLOAT,
    learning_rate FLOAT,
    gpu_memory FLOAT,
    custom_metrics TEXT,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL
);