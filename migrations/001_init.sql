CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    placeholder_message_id BIGINT NOT NULL,
    input_file_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('QUEUED','RUNNING','COMPLETED','FAILED','CANCELLED')),
    error TEXT,
    runpod_request_id TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_chat ON jobs(chat_id);
CREATE INDEX IF NOT EXISTS idx_jobs_runpod_request ON jobs(runpod_request_id);
