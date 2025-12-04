CREATE TABLE system_logs (
    id SERIAL PRIMARY KEY,
    level VARCHAR(20) NOT NULL,
    module VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    logger_name VARCHAR(100),
    function_name VARCHAR(100),
    line_number INTEGER,
    file_path VARCHAR(255),
    user_id INTEGER,
    request_id VARCHAR(100),
    session_id VARCHAR(100),
    metadata TEXT,
    ip_address VARCHAR(45),
    user_agent TEXT,
    timestamp TIMESTAMPTZ DEFAULT now() NOT NULL
);